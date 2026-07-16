"""Causal streaming inference for Personal VAD 2.0."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import Tensor
from torch.nn import functional as F

from features import FeatureFrame, PvadFeatureConfig, PvadFeatureExtractor
from model.pvad2 import Pvad2, PvadStreamingState


@dataclass(frozen=True)
class StreamingPvadOutput:
  logits: Tensor
  probabilities: Tensor
  frames: tuple[FeatureFrame, ...]
  cumulative_rtf: float | None


def _empty_stack_caches(x, layers):
  attention = tuple(
      x.new_zeros((x.size(0), layer.self_attn.h, 0,
                   2 * layer.self_attn.d_k))
      for layer in layers)
  convolution = tuple(
      x.new_zeros((x.size(0), layer.size, 0)) for layer in layers)
  return attention, convolution


def _streaming_attention_mask(model, batch_size, chunk_size, cache_size,
                              offset, device):
  if cache_size > offset:
    raise ValueError('Attention cache is longer than the processed prefix.')
  query_positions = torch.arange(
      offset, offset + chunk_size, device=device).view(chunk_size, 1)
  key_positions = torch.arange(
      offset - cache_size, offset + chunk_size, device=device).view(
          1, cache_size + chunk_size)
  mask = key_positions <= query_positions
  if model.left_context is not None and model.left_context >= 0:
    mask = mask & ((query_positions - key_positions) <= model.left_context)
  return mask.unsqueeze(0).expand(batch_size, -1, -1)


def _trim_attention_cache(cache, left_context):
  if left_context is None or left_context < 0:
    return cache
  if left_context == 0:
    return cache[:, :, :0]
  return cache[:, :, -left_context:]


def _validate_caches(attention, convolution, layers, batch_size):
  if len(attention) != len(layers) or len(convolution) != len(layers):
    raise ValueError('Streaming cache count does not match model layers.')
  for att_cache, cnn_cache in zip(attention, convolution):
    if att_cache.size(0) != batch_size or cnn_cache.size(0) != batch_size:
      raise ValueError('Streaming batch size cannot change without reset.')


def _run_stack_chunk(model, x, layers, attention, convolution, pos_enc,
                     offset):
  _validate_caches(attention, convolution, layers, x.size(0))
  next_attention = []
  next_convolution = []
  pad_mask = torch.ones(
      (x.size(0), 1, x.size(1)), dtype=torch.bool, device=x.device)

  for layer, att_cache, cnn_cache in zip(
      layers, attention, convolution):
    cache_size = att_cache.size(2)
    mask = _streaming_attention_mask(
        model, x.size(0), x.size(1), cache_size, offset, x.device)
    pos_emb = pos_enc.position_encoding(
        offset - cache_size, cache_size + x.size(1)).to(
            device=x.device, dtype=x.dtype)
    x, _, new_att_cache, new_cnn_cache = layer(
        x, mask, pos_emb, mask_pad=pad_mask,
        att_cache=att_cache, cnn_cache=cnn_cache)
    next_attention.append(
        _trim_attention_cache(new_att_cache, model.left_context))
    next_convolution.append(new_cnn_cache)

  return x, tuple(next_attention), tuple(next_convolution)


def forward_feature_chunk(model: Pvad2, inputs: Tensor,
                          embedding: Tensor | None = None,
                          state: PvadStreamingState | None = None):
  """Run one non-empty chunk of already stacked 512-d acoustic features."""

  if model.training:
    raise ValueError('Streaming inference requires model.eval().')
  if model.subsampling_type != 'linear':
    raise ValueError(
        'Streaming feature chunks require linear model subsampling because '
        'PvadFeatureExtractor already applies factor-three subsampling.')
  if inputs.dim() != 3 or inputs.size(-1) != model.input_dim:
    raise ValueError(
        f'inputs must have shape (B, T, {model.input_dim}).')
  if inputs.size(1) == 0:
    raise ValueError('inputs must contain at least one feature frame.')

  offset = state.offset if state is not None else 0
  input_mask = torch.ones(
      (inputs.size(0), 1, inputs.size(1)),
      dtype=torch.bool, device=inputs.device)
  enc_inputs, _, _ = model.subsample(inputs, input_mask, offset=offset)
  spk_inputs, _, _ = model.speaker_subsample(
      inputs, input_mask, offset=offset)

  if state is None:
    enc_att, enc_cnn = _empty_stack_caches(enc_inputs, model.encoder)
    spk_att, spk_cnn = _empty_stack_caches(
        spk_inputs, model.speaker_pre_net)
  else:
    enc_att = state.encoder_att_caches
    enc_cnn = state.encoder_cnn_caches
    spk_att = state.speaker_att_caches
    spk_cnn = state.speaker_cnn_caches

  enc_outputs, enc_att, enc_cnn = _run_stack_chunk(
      model, enc_inputs, model.encoder, enc_att, enc_cnn,
      model.pos_enc, offset)
  spk_outputs, spk_att, spk_cnn = _run_stack_chunk(
      model, spk_inputs, model.speaker_pre_net, spk_att, spk_cnn,
      model.speaker_pos_enc, offset)

  ref_embedding = model._normalize_embedding(
      embedding, inputs.size(0), inputs.device, inputs.dtype)
  if ref_embedding.size(1) == 1:
    ref_embedding = ref_embedding.expand(-1, spk_outputs.size(1), -1)
  elif ref_embedding.size(1) != spk_outputs.size(1):
    raise ValueError(
        'Time-varying embedding length must match the current chunk.')

  cos_sim = F.cosine_similarity(
      spk_outputs, ref_embedding, dim=-1).unsqueeze(-1)
  gammas = 1.0 + model.gamma_module(cos_sim)
  betas = model.beta_module(cos_sim)
  logits = model.classifier(model.film(enc_outputs, gammas, betas))
  next_state = PvadStreamingState(
      offset=offset + logits.size(1),
      encoder_att_caches=enc_att,
      encoder_cnn_caches=enc_cnn,
      speaker_att_caches=spk_att,
      speaker_cnn_caches=spk_cnn)
  return logits, next_state


class PvadStreamingAdapter:
  """Stateful mono-audio to three-class probability adapter."""

  def __init__(self, model: Pvad2, embedding: Tensor | None,
               feature_config: PvadFeatureConfig | None = None,
               device: torch.device | str | None = None,
               measure_rtf: bool = False):
    if device is not None:
      model = model.to(device)
    self.model = model.eval()
    self.device = next(model.parameters()).device
    self.feature_extractor = PvadFeatureExtractor(
        feature_config, device=self.device)
    self.measure_rtf = measure_rtf
    self.embedding = self._prepare_embedding(embedding)
    self.reset()

  def _prepare_embedding(self, embedding):
    if embedding is None:
      return None
    embedding = torch.as_tensor(
        embedding, dtype=torch.float32, device=self.device)
    if embedding.dim() == 1:
      embedding = embedding.unsqueeze(0)
    if (embedding.dim() != 2
        or embedding.size(1) != self.model.speaker_embedding_dim):
      raise ValueError(
          'embedding must have shape (speaker_embedding_dim,) or '
          '(1, speaker_embedding_dim).')
    if embedding.size(0) != 1:
      raise ValueError('The audio adapter currently supports batch size one.')
    return embedding

  def reset(self):
    self.feature_extractor.reset()
    self.state = None
    self.audio_samples = 0
    self.feature_frames = 0
    self.processing_seconds = 0.0

  def _synchronize(self):
    if self.device.type == 'cuda':
      torch.cuda.synchronize(self.device)

  def feed_audio(self, waveform_chunk: Tensor) -> StreamingPvadOutput:
    waveform_chunk = torch.as_tensor(waveform_chunk)
    if waveform_chunk.dim() not in (1, 2):
      raise ValueError('waveform_chunk must be mono or channel-first.')
    self.audio_samples += waveform_chunk.shape[-1]

    started = None
    if self.measure_rtf:
      self._synchronize()
      started = perf_counter()

    features = self.feature_extractor.feed(waveform_chunk)
    if features.size(0):
      with torch.inference_mode():
        logits, self.state = forward_feature_chunk(
            self.model, features.unsqueeze(0), self.embedding, self.state)
      logits = logits.squeeze(0)
      probabilities = logits.softmax(dim=-1)
    else:
      logits = torch.empty(
          (0, self.model.num_classes), device=self.device)
      probabilities = torch.empty_like(logits)

    if self.measure_rtf:
      self._synchronize()
      self.processing_seconds += perf_counter() - started

    start_frame = self.feature_frames
    frames = tuple(self.feature_extractor.frame_timing(
        logits.size(0), start_index=start_frame))
    self.feature_frames += logits.size(0)
    duration = (
        self.audio_samples / self.feature_extractor.config.sample_rate)
    rtf = None
    if self.measure_rtf and duration > 0:
      rtf = self.processing_seconds / duration
    return StreamingPvadOutput(
        logits=logits, probabilities=probabilities,
        frames=frames, cumulative_rtf=rtf)
