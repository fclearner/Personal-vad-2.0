from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import (PvadFeatureConfig, PvadFeatureExtractor,
                      labels_from_intervals)


def test_paper_dimensions_and_timing():
  config = PvadFeatureConfig()
  extractor = PvadFeatureExtractor(config)
  features = extractor.extract(torch.linspace(-0.5, 0.5, config.sample_rate))
  assert config.output_dim == 512
  assert config.output_shift_ms == 30.0
  assert config.receptive_field_ms == 62.0
  assert extractor.mel_filter.sum(dim=0).gt(0).all()
  assert features.shape == (32, 512)
  timing = extractor.frame_timing(features.size(0))
  assert (timing[0].stack_start_sample,
          timing[0].decision_start_sample,
          timing[0].decision_end_sample) == (0, 480, 992)
  assert timing[1].decision_start_sample - timing[0].decision_start_sample == 480
  timing_slice = extractor.frame_timing(2, start_index=2)
  assert [frame.index for frame in timing_slice] == [2, 3]
  assert timing_slice == extractor.frame_timing(4)[2:]


def test_streaming_matches_offline_for_irregular_chunks():
  waveform = torch.randn(24173, generator=torch.Generator().manual_seed(7))
  offline = PvadFeatureExtractor().extract(waveform)
  streaming = PvadFeatureExtractor()
  outputs = []
  position = 0
  sizes = [1, 79, 160, 511, 997, 13, 2048]
  index = 0
  while position < waveform.numel():
    size = sizes[index % len(sizes)]
    output = streaming.feed(waveform[position:position + size])
    if output.numel():
      outputs.append(output)
    position += size
    index += 1
  streamed = torch.cat(outputs)
  assert streamed.shape == offline.shape
  torch.testing.assert_close(streamed, offline, rtol=1e-5, atol=1e-5)


def test_identity_labels_use_target_priority():
  extractor = PvadFeatureExtractor()
  frames = extractor.frame_timing(4)
  target = [(frames[1].decision_start_sample,
             frames[2].decision_end_sample)]
  non_target = [(frames[0].decision_start_sample,
                 frames[1].decision_end_sample)]
  labels = labels_from_intervals(frames, target, non_target)
  assert labels.tolist() == [1, 0, 0, 2]

  quarter_end = (frames[0].decision_start_sample
                 + (frames[0].decision_end_sample
                    - frames[0].decision_start_sample) // 4)
  duplicate_quarter = [(frames[0].decision_start_sample, quarter_end)] * 2
  assert labels_from_intervals(frames[:1], duplicate_quarter, []).item() == 2

if __name__ == '__main__':
  test_paper_dimensions_and_timing()
  test_streaming_matches_offline_for_irregular_chunks()
  test_identity_labels_use_target_priority()
  print('feature tests ok')
