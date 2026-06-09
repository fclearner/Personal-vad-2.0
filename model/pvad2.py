import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .attention import RelPositionalEncoding
from .conformer import ConformerBlock
from .film import FiLM
from .subsampling import Conv2dSubsampling3, LinearNoSubsampling
from .utils import make_pad_mask


class Pvad2(nn.Module):
  """Personal VAD 2.0 style frame classifier.

  The default configuration follows the public paper setup: 512-dim stacked
  acoustic features, 64-dim Conformer layers, 4 encoder layers, a 2-layer
  speaker pre-net, cosine-similarity FiLM conditioning, and 3 frame classes
  (target speech, non-target speech, non-speech).
  """

  def __init__(
      self,
      input_dim: int = 512,
      encoder_dim: int = 64,
      speaker_embedding_dim: int = 64,
      num_classes: int = 3,
      num_encoder_layers: int = 4,
      num_speaker_layers: int = 2,
      num_attention_heads: int = 8,
      linear_units: int = 64,
      dropout_rate: float = 0.1,
      attention_dropout_rate: float = 0.0,
      conv_kernel_size: int = 7,
      causal: bool = True,
      left_context: int | None = 31,
      subsampling: str = 'linear',
  ):
    super(Pvad2, self).__init__()
    self.input_dim = input_dim
    self.encoder_dim = encoder_dim
    self.speaker_embedding_dim = speaker_embedding_dim
    self.num_classes = num_classes
    self.left_context = left_context
    self.subsampling_type = subsampling
    self.model_config = {
        'input_dim': input_dim,
        'encoder_dim': encoder_dim,
        'speaker_embedding_dim': speaker_embedding_dim,
        'num_classes': num_classes,
        'num_encoder_layers': num_encoder_layers,
        'num_speaker_layers': num_speaker_layers,
        'num_attention_heads': num_attention_heads,
        'linear_units': linear_units,
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'conv_kernel_size': conv_kernel_size,
        'causal': causal,
        'left_context': left_context,
        'subsampling': subsampling,
    }

    self.pos_enc = RelPositionalEncoding(
        d_model=encoder_dim, dropout_rate=dropout_rate)
    self.speaker_pos_enc = RelPositionalEncoding(
        d_model=encoder_dim, dropout_rate=dropout_rate)
    self.subsample = self._build_subsampling(
        input_dim, encoder_dim, dropout_rate, self.pos_enc, subsampling)
    self.speaker_subsample = self._build_subsampling(
        input_dim, encoder_dim, dropout_rate, self.speaker_pos_enc,
        subsampling)

    self.encoder = nn.ModuleList([
        self._build_conformer_block(
            encoder_dim, num_attention_heads, linear_units, dropout_rate,
            attention_dropout_rate, conv_kernel_size, causal)
        for _ in range(num_encoder_layers)
    ])
    self.speaker_pre_net = nn.ModuleList([
        self._build_conformer_block(
            encoder_dim, num_attention_heads, linear_units, dropout_rate,
            attention_dropout_rate, conv_kernel_size, causal)
        for _ in range(num_speaker_layers)
    ])

    if speaker_embedding_dim == encoder_dim:
      self.speaker_embedding_proj = nn.Identity()
    else:
      self.speaker_embedding_proj = nn.Linear(speaker_embedding_dim,
                                              encoder_dim)

    self.gamma_module = nn.Linear(1, encoder_dim, bias=True)
    self.beta_module = nn.Linear(1, encoder_dim, bias=True)
    self.film = FiLM()
    self.classifier = nn.Linear(encoder_dim, num_classes, bias=True)
    self.reset_parameters()

  @staticmethod
  def _build_conformer_block(size, attention_heads, linear_units,
                             dropout_rate, attention_dropout_rate,
                             conv_kernel_size, causal):
    return ConformerBlock(size=size,
                          attention_heads=attention_heads,
                          linear_units=linear_units,
                          dropout_rate=dropout_rate,
                          attention_dropout_rate=attention_dropout_rate,
                          cnn_module_kernel=conv_kernel_size,
                          causal=causal)

  @staticmethod
  def _build_subsampling(input_dim, encoder_dim, dropout_rate, pos_enc,
                         subsampling):
    if subsampling == 'linear':
      return LinearNoSubsampling(input_dim, encoder_dim, dropout_rate, pos_enc)
    if subsampling == 'conv2d3':
      return Conv2dSubsampling3(input_dim, encoder_dim, dropout_rate, pos_enc)
    raise ValueError(f'Unsupported subsampling mode: {subsampling}')

  def reset_parameters(self):
    for module in self.modules():
      if module in (self.gamma_module, self.beta_module):
        continue
      if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    for module in self.modules():
      for name, param in module.named_parameters(recurse=False):
        if name in ('pos_bias_u', 'pos_bias_v'):
          nn.init.xavier_uniform_(param)

    nn.init.zeros_(self.gamma_module.weight)
    nn.init.zeros_(self.gamma_module.bias)
    nn.init.zeros_(self.beta_module.weight)
    nn.init.zeros_(self.beta_module.bias)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  @classmethod
  def load_model(cls, path):
    package = torch.load(path, map_location=lambda storage, loc: storage)
    return cls.load_model_from_package(package)

  @classmethod
  def load_model_from_package(cls, package):
    model = cls(**package.get('model_config', {}))
    model.load_state_dict(package['state_dict'])
    return model

  @staticmethod
  def serialize(model, optimizer=None, epoch=0, tr_loss=None, cv_loss=None):
    package = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'model_config': getattr(model, 'model_config', {}),
    }
    if optimizer is not None:
      package['optim_dict'] = optimizer.state_dict()
    if tr_loss is not None:
      package['tr_loss'] = tr_loss
      package['cv_loss'] = cv_loss
    return package

  def update_dropout(self, dropout_p):
    for module in self.modules():
      if isinstance(module, nn.Dropout):
        module.p = dropout_p

  def _subsample_lengths(self, input_lengths):
    if self.subsampling_type == 'linear':
      return input_lengths
    if self.subsampling_type == 'conv2d3':
      return torch.div(input_lengths - 2, 3, rounding_mode='floor').clamp_min(0)
    raise ValueError(f'Unsupported subsampling mode: {self.subsampling_type}')

  def _make_attention_mask(self, lengths, max_len):
    valid_keys = ~make_pad_mask(lengths, max_len)
    if self.left_context is None:
      return valid_keys.unsqueeze(1)

    steps = torch.arange(max_len, device=lengths.device)
    query = steps.view(max_len, 1)
    key = steps.view(1, max_len)
    context = key <= query
    if self.left_context >= 0:
      context = context & ((query - key) <= self.left_context)
    return context.unsqueeze(0) & valid_keys.unsqueeze(1)

  def _normalize_lengths(self, inputs, input_lengths):
    if input_lengths is None:
      return torch.full((inputs.size(0),), inputs.size(1),
                        dtype=torch.long, device=inputs.device)
    return input_lengths.to(device=inputs.device, dtype=torch.long)

  def _normalize_embedding(self, embedding, batch_size, device, dtype):
    if embedding is None:
      embedding = torch.zeros(batch_size, self.speaker_embedding_dim,
                              device=device, dtype=dtype)
    else:
      embedding = embedding.to(device=device, dtype=dtype)

    if embedding.dim() == 2:
      embedding = self.speaker_embedding_proj(embedding).unsqueeze(1)
    elif embedding.dim() == 3:
      embedding = self.speaker_embedding_proj(embedding)
    else:
      raise ValueError('embedding must have shape (B, E) or (B, T, E).')
    return embedding

  def _run_stack(self, x, pos_emb, lengths, layers):
    max_len = x.size(1)
    pad_mask = ~make_pad_mask(lengths, max_len).unsqueeze(1)
    attn_mask = self._make_attention_mask(lengths, max_len)

    for layer in layers:
      x, _, _, _ = layer(x, attn_mask, pos_emb, mask_pad=pad_mask)

    return x.masked_fill(~pad_mask.transpose(1, 2), 0.0), pad_mask

  def forward(self,
              inputs: Tensor,
              embedding: Tensor | None = None,
              input_lengths: Tensor | None = None,
              return_lengths: bool = False):
    """Run frame-level Personal VAD inference.

    Args:
      inputs: Acoustic features with shape ``(batch, time, input_dim)``.
      embedding: Speaker embedding with shape ``(batch, speaker_dim)``. Passing
        ``None`` or a zero vector represents enrollment-less inference.
      input_lengths: Valid frame counts before padding.
      return_lengths: When true, return ``(logits, output_lengths)``.
    """
    if inputs.dim() != 3:
      raise ValueError('inputs must have shape (B, T, D).')
    if inputs.size(-1) != self.input_dim:
      raise ValueError(
          f'Expected input dim {self.input_dim}, got {inputs.size(-1)}.')

    input_lengths = self._normalize_lengths(inputs, input_lengths)
    input_mask = ~make_pad_mask(input_lengths, inputs.size(1)).unsqueeze(1)

    enc_inputs, enc_pos_emb, _ = self.subsample(inputs, input_mask)
    spk_inputs, spk_pos_emb, _ = self.speaker_subsample(inputs, input_mask)
    output_lengths = self._subsample_lengths(input_lengths).clamp_max(
        enc_inputs.size(1))

    enc_outputs, enc_pad_mask = self._run_stack(
        enc_inputs, enc_pos_emb, output_lengths, self.encoder)
    spk_outputs, _ = self._run_stack(
        spk_inputs, spk_pos_emb, output_lengths, self.speaker_pre_net)

    ref_embedding = self._normalize_embedding(
        embedding, inputs.size(0), inputs.device, inputs.dtype)
    if ref_embedding.size(1) == 1:
      ref_embedding = ref_embedding.expand(-1, spk_outputs.size(1), -1)
    elif ref_embedding.size(1) != spk_outputs.size(1):
      raise ValueError(
          'Time-varying embedding length must match model output length.')

    cos_sim = F.cosine_similarity(spk_outputs, ref_embedding, dim=-1)
    cos_sim = cos_sim.unsqueeze(-1)
    gammas = 1.0 + self.gamma_module(cos_sim)
    betas = self.beta_module(cos_sim)
    outputs = self.film(enc_outputs, gammas, betas)
    logits = self.classifier(outputs)
    logits = logits.masked_fill(~enc_pad_mask.transpose(1, 2), 0.0)

    if return_lengths:
      return logits, output_lengths
    return logits


PVAD2 = Pvad2
