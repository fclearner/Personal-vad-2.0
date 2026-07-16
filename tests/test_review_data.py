from collections import Counter
from pathlib import Path
import random
import sys
import tempfile

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_review_set import (
    DEV_SCENARIOS, SAMPLE_RATE, TRAIN_SCENARIOS, _active_rms,
    _add_noise, _write_pcm16_and_reload, scenario_plan, simulated_farfield_rir,
    split_speakers, transform_intervals)


def test_scenario_plan_is_exact_and_deterministic():
  first = scenario_plan(17)
  second = scenario_plan(17)
  assert first == second
  assert len(first['train']) == 80
  assert len(first['dev']) == 20
  assert Counter(first['train']) == Counter(TRAIN_SCENARIOS)
  assert Counter(first['dev']) == Counter(DEV_SCENARIOS)


def test_speaker_split_and_interval_transform():
  train, dev = split_speakers(
      [f'S{index:04d}' for index in range(2, 14)], train_count=8)
  assert train == [f'S{index:04d}' for index in range(2, 10)]
  assert dev == [f'S{index:04d}' for index in range(10, 14)]
  assert not set(train) & set(dev)

  transformed = transform_intervals(
      [(100, 300), (400, 900)], crop_start=200, length=500,
      offset=50, tail_samples=20)
  assert transformed == [(50, 170), (250, 570)]


def test_farfield_rir_and_active_snr():
  rir = simulated_farfield_rir()
  assert rir.numel() == round(0.18 * SAMPLE_RATE)
  torch.testing.assert_close(
      rir.square().sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)
  assert torch.count_nonzero(rir) == 4

  mixture = torch.zeros(2000)
  mixture[200:1200] = 0.1
  intervals = [(200, 1200)]
  noise = torch.randn(4000, generator=torch.Generator().manual_seed(3))
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'noise.wav'
    sf.write(
        str(path), noise.numpy(), SAMPLE_RATE, subtype='FLOAT')
    mixed, metadata = _add_noise(
        mixture, path, random.Random(5), snr_db=0.0,
        signal_intervals=intervals)
  noise_component = mixed - mixture
  signal_rms = _active_rms(mixture, intervals)
  noise_rms = _active_rms(noise_component, intervals)
  assert abs(float(signal_rms / noise_rms) - 1.0) < 0.05
  assert metadata['snr_db'] == 0.0


def test_features_are_computed_from_persisted_pcm16():
  from features import PvadFeatureExtractor, load_audio

  waveform = torch.linspace(-0.99, 0.99, SAMPLE_RATE)
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'mixture.wav'
    stored = _write_pcm16_and_reload(path, waveform)
    generated = PvadFeatureExtractor().extract(stored)
    reconstructed = PvadFeatureExtractor().extract(load_audio(path))
  torch.testing.assert_close(generated, reconstructed, rtol=0.0, atol=0.0)


if __name__ == '__main__':
  test_scenario_plan_is_exact_and_deterministic()
  test_speaker_split_and_interval_transform()
  test_farfield_rir_and_active_snr()
  test_features_are_computed_from_persisted_pcm16()
  print('review data tests ok')
