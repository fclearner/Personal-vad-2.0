from pathlib import Path
import sys
import tempfile

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prepare_speaker_manifest import (
    build_utterance_records, discover_speaker_wavs, partition_speakers,
    partition_speakers_balanced, select_speakers)


def _speaker(root, name, utterances=6):
  directory = root / name
  directory.mkdir(parents=True)
  waveform = torch.linspace(-0.2, 0.2, 320).numpy()
  for index in range(utterances):
    sf.write(
        str(directory / f'{name}-{index}.wav'), waveform, 16000,
        subtype='PCM_16')


def test_multiple_roots_and_deterministic_partitions():
  with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    first = base / 'first'
    second = base / 'second'
    for root in (first, second):
      root.mkdir()
    for index in range(2, 7):
      _speaker(first, f'S{index:04d}')
    for index in range(7, 12):
      _speaker(second, f'S{index:04d}')

    speakers = discover_speaker_wavs([first, second])
    one = partition_speakers(speakers, 6, 2, 2, seed=17)
    two = partition_speakers(speakers, 6, 2, 2, seed=17)
    assert one == two
    assert {split: len(names) for split, names in one.items()} == {
        'train': 6, 'dev': 2, 'test': 2}
    assert not set(one['train']) & set(one['dev'])
    assert not set(one['train']) & set(one['test'])
    assert not set(one['dev']) & set(one['test'])

    records = build_utterance_records(
        speakers, one, enrollment_utterances=2)
    assert len(records) == 60
    assert sum(row['role'] == 'enrollment' for row in records) == 20
    assert sum(row['eligible_for_mixing'] for row in records) == 32
    assert not any(
        row['eligible_for_mixing'] for row in records
        if row['split'] == 'test')

    no_test = partition_speakers(speakers, 8, 2, 0, seed=17)
    assert no_test['test'] == []


def test_duplicate_speaker_across_roots_is_rejected():
  with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    first = base / 'first'
    second = base / 'second'
    first.mkdir()
    second.mkdir()
    _speaker(first, 'S0002')
    _speaker(second, 'S0002')
    try:
      discover_speaker_wavs([first, second])
    except ValueError as error:
      assert 'Duplicate speaker' in str(error)
    else:
      raise AssertionError('Duplicate speaker was not rejected.')


def test_balanced_partition_preserves_groups_in_every_split():
  speakers = {f'S{index:04d}': () for index in range(2, 12)}
  groups = {
      speaker: 'M' if index < 7 else 'F'
      for index, speaker in enumerate(sorted(speakers), 2)}
  partitions = partition_speakers_balanced(
      speakers, groups, 6, 2, 2, seed=17)
  for split, expected_per_group in (
      ('train', 3), ('dev', 1), ('test', 1)):
    counts = {
        group: sum(groups[speaker] == group for speaker in partitions[split])
        for group in ('M', 'F')}
    assert counts == {'M': expected_per_group, 'F': expected_per_group}
  assert not set(partitions['train']) & set(partitions['dev'])
  assert not set(partitions['train']) & set(partitions['test'])
  assert not set(partitions['dev']) & set(partitions['test'])


def test_explicit_speaker_selection_is_exact():
  speakers = {'S0002': ('a.wav',), 'S0003': ('b.wav',)}
  assert select_speakers(speakers, ['S0003']) == {
      'S0003': ('b.wav',)}
  try:
    select_speakers(speakers, ['S9999'])
  except ValueError as error:
    assert 'not discovered' in str(error)
  else:
    raise AssertionError('Missing selected speaker was not rejected.')
