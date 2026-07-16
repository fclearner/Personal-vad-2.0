"""Build deterministic speaker-disjoint source manifests without copying audio."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf


def discover_speaker_wavs(wav_roots, min_utterances=6):
  """Return speaker -> sorted wav paths across non-overlapping roots."""

  speakers = {}
  roots = [Path(root).resolve() for root in wav_roots]
  for root in roots:
    if not root.is_dir():
      raise ValueError(f'Wav root does not exist: {root}')
    for directory in sorted(root.glob('S*')):
      if not directory.is_dir():
        continue
      wavs = tuple(sorted(path.resolve() for path in directory.glob('*.wav')))
      if not wavs:
        continue
      if len(wavs) < min_utterances:
        raise ValueError(
            f'{directory.name} has {len(wavs)} wavs; '
            f'at least {min_utterances} are required.')
      if directory.name in speakers:
        raise ValueError(
            f'Duplicate speaker {directory.name} across wav roots.')
      speakers[directory.name] = wavs
  if not speakers:
    raise ValueError('No speaker wav directories were found.')
  return speakers


def partition_speakers(speakers, train_count, dev_count, test_count, seed):
  """Deterministically partition every speaker exactly once."""

  names = sorted(speakers)
  counts = {
      'train': train_count,
      'dev': dev_count,
      'test': test_count,
  }
  if train_count <= 0 or dev_count <= 0 or test_count < 0:
    raise ValueError(
        'train/dev counts must be positive and test count non-negative.')
  if sum(counts.values()) != len(names):
    raise ValueError(
        f'Speaker counts sum to {sum(counts.values())}, '
        f'but {len(names)} speakers were discovered.')
  random.Random(seed).shuffle(names)
  train_end = train_count
  dev_end = train_end + dev_count
  partitions = {
      'train': sorted(names[:train_end]),
      'dev': sorted(names[train_end:dev_end]),
      'test': sorted(names[dev_end:]),
  }
  flattened = [speaker for split in partitions.values() for speaker in split]
  if len(flattened) != len(set(flattened)):
    raise AssertionError('Speaker partition leakage detected.')
  return partitions


def build_utterance_records(speakers, partitions, enrollment_utterances=2,
                            sample_rate=16000):
  """Read audio headers and mark disjoint enrollment/current utterances."""

  if enrollment_utterances <= 0:
    raise ValueError('enrollment_utterances must be positive.')
  records = []
  seen_paths = set()
  for split in ('train', 'dev', 'test'):
    for speaker in partitions[split]:
      wavs = speakers[speaker]
      if len(wavs) <= enrollment_utterances:
        raise ValueError(
            f'{speaker} has no current audio after enrollment selection.')
      for index, path in enumerate(wavs):
        path_string = str(path)
        if path_string in seen_paths:
          raise AssertionError(f'Duplicate audio path: {path}')
        seen_paths.add(path_string)
        info = sf.info(path_string)
        if info.samplerate != sample_rate:
          raise ValueError(
              f'{path} has sample rate {info.samplerate}, expected {sample_rate}.')
        if info.channels != 1:
          raise ValueError(f'{path} has {info.channels} channels, expected mono.')
        role = 'enrollment' if index < enrollment_utterances else 'current'
        records.append({
            'id': path.stem,
            'speaker_id': speaker,
            'split': split,
            'role': role,
            'path': path_string,
            'frames': info.frames,
            'duration_seconds': info.frames / info.samplerate,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'subtype': info.subtype,
            'eligible_for_mixing': split != 'test' and role == 'current',
        })
  return records


def _write_jsonl(path, rows):
  with path.open('w', encoding='utf-8') as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_manifests(output_dir, wav_roots, partitions, records, seed,
                    enrollment_utterances):
  output_dir = Path(output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  split_payload = {
      'seed': seed,
      'wav_roots': [str(Path(root).resolve()) for root in wav_roots],
      'enrollment_utterances_per_speaker': enrollment_utterances,
      'partitions': partitions,
  }
  (output_dir / 'speaker_split.json').write_text(
      json.dumps(split_payload, ensure_ascii=False, indent=2),
      encoding='utf-8')
  _write_jsonl(output_dir / 'utterances.jsonl', records)
  for split in ('train', 'dev', 'test'):
    _write_jsonl(
        output_dir / f'{split}.jsonl',
        [record for record in records if record['split'] == split])

  role_counts = Counter(
      (record['split'], record['role']) for record in records)
  durations = defaultdict(float)
  for record in records:
    durations[record['split']] += record['duration_seconds']
  summary = {
      'speakers': {
          split: len(names) for split, names in partitions.items()},
      'utterances': {
          f'{split}/{role}': count
          for (split, role), count in sorted(role_counts.items())},
      'duration_hours': {
          split: durations[split] / 3600.0
          for split in ('train', 'dev', 'test')},
      'speaker_overlap': {
          'train_dev': sorted(
              set(partitions['train']) & set(partitions['dev'])),
          'train_test': sorted(
              set(partitions['train']) & set(partitions['test'])),
          'dev_test': sorted(
              set(partitions['dev']) & set(partitions['test'])),
      },
      'eligible_mixing_utterances': sum(
          record['eligible_for_mixing'] for record in records),
  }
  (output_dir / 'summary.json').write_text(
      json.dumps(summary, ensure_ascii=False, indent=2),
      encoding='utf-8')
  return summary


def parse_args():
  parser = argparse.ArgumentParser(
      description='Create speaker-disjoint Personal VAD source manifests.')
  parser.add_argument(
      '--wav-root', action='append', required=True,
      help='Speaker-directory root; repeat for additional non-overlapping roots.')
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--train-speakers', type=int, default=80)
  parser.add_argument('--dev-speakers', type=int, default=10)
  parser.add_argument('--test-speakers', type=int, default=10)
  parser.add_argument('--enrollment-utterances', type=int, default=2)
  parser.add_argument('--min-utterances', type=int, default=6)
  parser.add_argument('--sample-rate', type=int, default=16000)
  parser.add_argument('--seed', type=int, default=20260716)
  return parser.parse_args()


def main():
  args = parse_args()
  speakers = discover_speaker_wavs(args.wav_root, args.min_utterances)
  partitions = partition_speakers(
      speakers, args.train_speakers, args.dev_speakers,
      args.test_speakers, args.seed)
  records = build_utterance_records(
      speakers, partitions, args.enrollment_utterances, args.sample_rate)
  summary = write_manifests(
      args.output_dir, args.wav_root, partitions, records, args.seed,
      args.enrollment_utterances)
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
