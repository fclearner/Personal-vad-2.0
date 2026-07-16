"""Build deterministic speaker-disjoint source manifests without copying audio."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from extract_aishell_webdataset import read_speaker_info


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


def select_speakers(speakers, selected_names):
  """Select an explicit speaker subset without copying source audio."""

  selected_names = list(selected_names)
  if len(selected_names) != len(set(selected_names)):
    raise ValueError('Speaker selection contains duplicates.')
  missing = sorted(set(selected_names) - set(speakers))
  if missing:
    raise ValueError(f'Selected speakers were not discovered: {missing}')
  if not selected_names:
    raise ValueError('Speaker selection is empty.')
  return {name: speakers[name] for name in sorted(selected_names)}


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


def partition_speakers_balanced(
    speakers, speaker_groups, train_count, dev_count, test_count, seed):
  """Partition equal-sized groups evenly across train/dev/test."""

  names = sorted(speakers)
  missing = sorted(set(names) - set(speaker_groups))
  if missing:
    raise ValueError(f'Missing speaker group metadata: {missing}')
  grouped = defaultdict(list)
  for name in names:
    grouped[speaker_groups[name]].append(name)
  if len(grouped) < 2:
    raise ValueError('Balanced partitioning requires at least two groups.')
  group_sizes = {group: len(members) for group, members in grouped.items()}
  if len(set(group_sizes.values())) != 1:
    raise ValueError(f'Speaker groups are not equal-sized: {group_sizes}')
  counts = {'train': train_count, 'dev': dev_count, 'test': test_count}
  if train_count <= 0 or dev_count <= 0 or test_count < 0:
    raise ValueError(
        'train/dev counts must be positive and test count non-negative.')
  if sum(counts.values()) != len(names):
    raise ValueError(
        f'Speaker counts sum to {sum(counts.values())}, '
        f'but {len(names)} speakers were discovered.')
  for split, count in counts.items():
    if count % len(grouped):
      raise ValueError(
          f'{split} count {count} is not divisible by {len(grouped)} groups.')

  partitions = {split: [] for split in counts}
  for group_index, group in enumerate(sorted(grouped)):
    members = sorted(grouped[group])
    random.Random(seed + group_index).shuffle(members)
    offset = 0
    for split, count in counts.items():
      take = count // len(grouped)
      partitions[split].extend(members[offset:offset + take])
      offset += take
    if offset != len(members):
      raise AssertionError(f'Unassigned speakers in group {group}.')
  for split in partitions:
    partitions[split].sort()
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
                    enrollment_utterances, speaker_groups=None):
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
  if speaker_groups is not None:
    summary['speaker_groups'] = {
        split: dict(sorted(Counter(
            speaker_groups[speaker] for speaker in names).items()))
        for split, names in partitions.items()
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
  parser.add_argument(
      '--speaker-info',
      help='Official AISHELL speaker.info; balances discovered M/F groups.')
  parser.add_argument(
      '--speaker-list',
      help='Optional text file with one selected speaker ID per line.')
  return parser.parse_args()


def main():
  args = parse_args()
  speakers = discover_speaker_wavs(args.wav_root, args.min_utterances)
  if args.speaker_list:
    selected_names = [
        line.strip() for line in Path(args.speaker_list).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')]
    speakers = select_speakers(speakers, selected_names)
  speaker_groups = None
  if args.speaker_info:
    metadata = read_speaker_info(args.speaker_info)
    speaker_groups = {
        speaker: metadata[speaker]
        for speaker in speakers if speaker in metadata}
    partitions = partition_speakers_balanced(
        speakers, speaker_groups, args.train_speakers, args.dev_speakers,
        args.test_speakers, args.seed)
  else:
    partitions = partition_speakers(
        speakers, args.train_speakers, args.dev_speakers,
        args.test_speakers, args.seed)
  records = build_utterance_records(
      speakers, partitions, args.enrollment_utterances, args.sample_rate)
  summary = write_manifests(
      args.output_dir, args.wav_root, partitions, records, args.seed,
      args.enrollment_utterances, speaker_groups)
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
