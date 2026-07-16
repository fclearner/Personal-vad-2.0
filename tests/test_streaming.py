from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import PvadFeatureExtractor
from model.pvad2 import Pvad2
from streaming import PvadStreamingAdapter, forward_feature_chunk


def _small_model():
  return Pvad2(
      num_encoder_layers=2, num_speaker_layers=1,
      dropout_rate=0.0, attention_dropout_rate=0.0).eval()


def test_cached_feature_chunks_match_offline():
  torch.manual_seed(13)
  model = _small_model()
  features = torch.randn(1, 47, 512)
  embedding = torch.randn(1, 64)
  with torch.inference_mode():
    offline = model(features, embedding)

  state = None
  chunks = []
  start = 0
  sizes = [1, 5, 17, 3, 11]
  chunk_index = 0
  while start < features.size(1):
    end = min(features.size(1), start + sizes[chunk_index % len(sizes)])
    logits, state = forward_feature_chunk(
        model, features[:, start:end], embedding, state)
    chunks.append(logits)
    start = end
    chunk_index += 1

  streamed = torch.cat(chunks, dim=1)
  torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=2e-5)
  assert state.offset == features.size(1)
  for cache in state.encoder_att_caches + state.speaker_att_caches:
    assert cache.size(2) <= model.left_context
  for cache in state.encoder_cnn_caches + state.speaker_cnn_caches:
    assert cache.size(2) == 6


def test_audio_adapter_matches_offline_and_reports_rtf():
  torch.manual_seed(19)
  waveform = torch.randn(21317) * 0.05
  embedding = torch.randn(64)
  model = _small_model()
  features = PvadFeatureExtractor().extract(waveform)
  with torch.inference_mode():
    offline = model(features.unsqueeze(0), embedding.unsqueeze(0)).squeeze(0)

  adapter = PvadStreamingAdapter(
      model, embedding, device='cpu', measure_rtf=True)
  outputs = []
  frame_indices = []
  position = 0
  sizes = [79, 480, 997, 13, 2048]
  index = 0
  last_output = None
  while position < waveform.numel():
    size = sizes[index % len(sizes)]
    last_output = adapter.feed_audio(waveform[position:position + size])
    if last_output.logits.size(0):
      outputs.append(last_output.logits)
      frame_indices.extend(frame.index for frame in last_output.frames)
    position += size
    index += 1

  streamed = torch.cat(outputs)
  torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=2e-5)
  assert adapter.feature_frames == features.size(0)
  assert frame_indices == list(range(features.size(0)))
  assert last_output.cumulative_rtf is not None
  assert last_output.cumulative_rtf > 0.0


def test_streaming_requires_eval_mode():
  model = _small_model().train()
  try:
    forward_feature_chunk(model, torch.randn(1, 1, 512))
  except ValueError as error:
    assert 'model.eval' in str(error)
  else:
    raise AssertionError('training-mode streaming must be rejected')


if __name__ == '__main__':
  test_cached_feature_chunks_match_offline()
  test_audio_adapter_matches_offline_and_reports_rtf()
  test_streaming_requires_eval_mode()
  print('streaming tests ok')
