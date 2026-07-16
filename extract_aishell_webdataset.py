"""Safely extract selected AISHELL-1 WAVs by speaker metadata."""

import argparse
import hashlib
import json
import os
import re
import tarfile
from collections import Counter
from pathlib import Path


WAV_NAME = re.compile(r'^BAC009(S\d{4})W\d+\.wav$')


def read_speaker_info(path):
  """Read official AISHELL speaker.info as Sxxxx -> M/F."""

  speakers = {}
  for line_number, line in enumerate(
      Path(path).read_text(encoding='utf-8').splitlines(), 1):
    fields = line.split()
    if len(fields) != 2 or not re.fullmatch(r'\d{4}', fields[0]):
      raise ValueError(f'Invalid speaker.info line {line_number}: {line!r}')
    speaker = f'S{fields[0]}'
    gender = fields[1].upper()
    if gender not in {'M', 'F'}:
      raise ValueError(f'Invalid gender on line {line_number}: {gender!r}')
    if speaker in speakers:
      raise ValueError(f'Duplicate speaker metadata: {speaker}')
    speakers[speaker] = gender
  if not speakers:
    raise ValueError('speaker.info is empty.')
  return speakers


def _sha256(path):
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    while chunk := handle.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _copy_member(source, destination, expected_size):
  """Copy one tar member atomically, verifying an existing identical file."""

  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.part')
  digest = hashlib.sha256()
  written = 0
  try:
    with temporary.open('xb') as handle:
      while chunk := source.read(1024 * 1024):
        handle.write(chunk)
        digest.update(chunk)
        written += len(chunk)
    if written != expected_size:
      raise ValueError(
          f'{destination.name}: wrote {written} bytes, expected {expected_size}.')
    if destination.exists():
      if destination.stat().st_size != written or _sha256(destination) != digest.hexdigest():
        raise ValueError(f'Existing file differs: {destination}')
      temporary.unlink()
      return False
    os.replace(temporary, destination)
    return True
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def extract_wavs(archives, speaker_info, output_root, genders=('F',)):
  """Extract only selected-gender WAV members into speaker directories."""

  metadata = read_speaker_info(speaker_info)
  requested = {gender.upper() for gender in genders}
  if not requested or not requested <= {'M', 'F'}:
    raise ValueError(f'Invalid gender filter: {sorted(requested)}')
  output_root = Path(output_root).resolve()
  totals = Counter()
  speaker_counts = Counter()
  archive_rows = []
  for archive in [Path(path).resolve() for path in archives]:
    if not archive.is_file():
      raise ValueError(f'Archive does not exist: {archive}')
    counts = Counter()
    with tarfile.open(archive, mode='r:gz') as handle:
      for member in handle:
        if not member.isfile():
          counts['non_file_ignored'] += 1
          continue
        if member.name.endswith('.json'):
          counts['json_ignored'] += 1
          continue
        match = WAV_NAME.fullmatch(member.name)
        if not match:
          raise ValueError(
              f'Unexpected non-JSON member in {archive.name}: {member.name!r}')
        speaker = match.group(1)
        if speaker not in metadata:
          raise ValueError(f'Missing speaker metadata for {speaker}')
        if metadata[speaker] not in requested:
          counts['gender_excluded'] += 1
          continue
        source = handle.extractfile(member)
        if source is None:
          raise ValueError(f'Could not read member: {member.name}')
        destination = output_root / speaker / member.name
        created = _copy_member(source, destination, member.size)
        counts['wav_extracted' if created else 'existing_verified'] += 1
        speaker_counts[speaker] += 1
    totals.update(counts)
    archive_rows.append({'archive': str(archive), **dict(sorted(counts.items()))})
  return {
      'output_root': str(output_root),
      'speaker_info': str(Path(speaker_info).resolve()),
      'gender_filter': sorted(requested),
      'archives': archive_rows,
      'totals': dict(sorted(totals.items())),
      'speakers': dict(sorted(speaker_counts.items())),
  }


def parse_args():
  parser = argparse.ArgumentParser(
      description='Extract gender-filtered AISHELL WAVs without using text.')
  parser.add_argument('--archive', action='append', required=True)
  parser.add_argument('--speaker-info', required=True)
  parser.add_argument('--output-root', required=True)
  parser.add_argument('--gender', action='append', choices=('M', 'F'), default=[])
  parser.add_argument('--report')
  return parser.parse_args()


def main():
  args = parse_args()
  report = extract_wavs(
      args.archive, args.speaker_info, args.output_root,
      genders=args.gender or ('F',))
  if args.report:
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
  print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
