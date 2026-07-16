from collections import Counter
import json
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
    _add_noise, _write_pcm16_and_reload, discover_tts_wavs,
    load_source_partitions,
    scaled_scenario_counts, scenario_plan, simulated_farfield_rir,
    speaker_pair_plan, split_speakers, transform_intervals)


def test_scenario_plan_is_exact_and_deterministic():
  first = scenario_plan(17)
  second = scenario_plan(17)
  assert first == second
  assert len(first['train']) == 80
  assert len(first['dev']) == 20
  assert Counter(first['train']) == Counter(TRAIN_SCENARIOS)
  assert Counter(first['dev']) == Counter(DEV_SCENARIOS)

  pilot = scenario_plan(17, train_samples=800, dev_samples=200)
  assert len(pilot['train']) == 800
  assert len(pilot['dev']) == 200
  assert Counter(pilot['train']) == {
      scenario: count * 10 for scenario, count in TRAIN_SCENARIOS.items()}
  assert Counter(pilot['dev']) == {
      scenario: count * 10 for scenario, count in DEV_SCENARIOS.items()}
  assert sum(scaled_scenario_counts(TRAIN_SCENARIOS, 83).values()) == 83


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


def test_speaker_pair_plan_is_deterministic_and_diverse():
  speakers = [f'S{index:04d}' for index in range(2, 12)]
  first = speaker_pair_plan(speakers, 25, seed=17)
  second = speaker_pair_plan(speakers, 25, seed=17)
  assert first == second
  assert all(target != non_target for target, non_target in first)
  assert {target for target, _ in first[:10]} == set(speakers)
  target_counts = Counter(target for target, _ in first)
  assert set(target_counts.values()) == {2, 3}
  assert len(set(first)) >= 20


def test_farfield_rir_and_active_snr():
  rir, metadata = simulated_farfield_rir(random.Random(17))
  repeated, repeated_metadata = simulated_farfield_rir(random.Random(17))
  different, _ = simulated_farfield_rir(random.Random(18))
  torch.testing.assert_close(rir, repeated, rtol=0.0, atol=0.0)
  assert metadata == repeated_metadata
  assert not torch.equal(rir, different)
  assert 1.5 <= metadata['distance_m'] <= 5.0
  assert 0.15 <= metadata['rt60_seconds'] <= 0.65
  assert 0 <= metadata['label_tail_samples'] < rir.numel()
  torch.testing.assert_close(
      rir.square().sum(), torch.tensor(1.0), rtol=1e-6, atol=1e-6)
  assert torch.count_nonzero(rir) > metadata['early_reflections']

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


def test_source_manifest_partitions_are_loaded_exactly():
  speakers = {f'S{index:04d}': () for index in range(2, 7)}
  payload = {
      'partitions': {
          'train': ['S0004', 'S0002'],
          'dev': ['S0005'],
          'test': ['S0003'],
      }}
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'speaker_split.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    selected, partitions = load_source_partitions(path, speakers)
  assert set(selected) == {'S0002', 'S0003', 'S0004', 'S0005'}
  assert partitions == payload['partitions']


def test_tts_sources_are_split_by_train_and_dev():
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for split in ('train', 'dev'):
      (root / split).mkdir()
      (root / split / f'response-{split}.wav').touch()
    sources = discover_tts_wavs(root)
  assert [path.name for path in sources['train']] == ['response-train.wav']
  assert [path.name for path in sources['dev']] == ['response-dev.wav']


if __name__ == '__main__':
  test_scenario_plan_is_exact_and_deterministic()
  test_speaker_split_and_interval_transform()
  test_farfield_rir_and_active_snr()
  test_features_are_computed_from_persisted_pcm16()
  print('review data tests ok')
