"""Evaluate a Personal VAD checkpoint on a recipe-backed manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import PvadManifestDataset
from metrics import SlicedClassificationAccumulator, event_metrics
from model.pvad2 import Pvad2


CLASS_EVENT_TYPES = {
    0: 'target',
    1: 'non_target',
    2: 'non_speech',
}


def parse_args():
  parser = argparse.ArgumentParser(
      description='Evaluate frame and control-event Personal VAD metrics.')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--manifest', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--speaker-embedding-dim', type=int, default=None)
  parser.add_argument('--target-threshold', type=float, default=0.5)
  parser.add_argument('--top-errors', type=int, default=50)
  parser.add_argument('--include-curves', action='store_true')
  return parser.parse_args()


def _read_jsonl(path):
  records = []
  with Path(path).open('r', encoding='utf-8') as handle:
    for line in handle:
      line = line.strip()
      if line:
        records.append(json.loads(line))
  return records


def _resolve_recipe_path(record):
  path = Path(record['recipe'])
  if not path.is_absolute():
    path = Path(record['_base_dir']) / path
  return path


def load_recipes(dataset):
  """Load and index every recipe file referenced by a manifest."""
  paths = {_resolve_recipe_path(record) for record in dataset.records}
  indexed = {}
  for path in sorted(paths):
    for recipe in _read_jsonl(path):
      sample_id = recipe['id']
      if sample_id in indexed:
        raise ValueError(f'Duplicate recipe id: {sample_id}')
      indexed[sample_id] = recipe
  missing = [
      record.get('id') for record in dataset.records
      if record.get('id') not in indexed]
  if missing:
    raise ValueError(f'Missing recipes for manifest ids: {missing[:5]}')
  return indexed


def sample_tags(record):
  """Return stable, non-exclusive classification/event slice tags."""
  tags = [record.get('scenario')]
  value = record.get('tags', ())
  tags.extend((value,) if isinstance(value, str) else value)
  return tuple(dict.fromkeys(str(tag) for tag in tags if tag))


def contiguous_label_events(labels, tags, tts_playback=False, offset=0):
  """Convert exclusive frame labels into event intervals.

  TTS-only frames are evaluated as playback interruptions, not as generic
  third-party speech. Target-labelled overlap remains a target event by design.
  """
  labels = np.asarray(labels, dtype=np.int64).reshape(-1)
  if labels.size == 0:
    return []
  changes = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
  events = []
  for start, end in zip(changes[:-1], changes[1:]):
    class_id = int(labels[start])
    if class_id not in CLASS_EVENT_TYPES:
      continue
    event_type = CLASS_EVENT_TYPES[class_id]
    if tts_playback and event_type == 'non_target':
      event_type = 'tts_playback'
    events.append({
        'start_frame': int(offset + start),
        'end_frame': int(offset + end),
        'event_type': event_type,
        'tags': list(tags),
    })
  return events


def _error_summary(sample_id, labels, predictions, target_scores, tags):
  labels = np.asarray(labels)
  predictions = np.asarray(predictions)
  errors = labels != predictions
  target = labels == 0
  false_target = (labels != 0) & (predictions == 0)
  missed_target = target & (predictions != 0)
  return {
      'id': sample_id,
      'tags': list(tags),
      'frames': int(labels.size),
      'frame_errors': int(errors.sum()),
      'frame_error_rate': float(errors.mean()) if labels.size else None,
      'false_target_frames': int(false_target.sum()),
      'missed_target_frames': int(missed_target.sum()),
      'max_target_score': float(np.max(target_scores)) if labels.size else None,
  }


def evaluate(model, dataset, recipes, device, target_threshold=0.5,
             include_curves=False, top_errors=50):
  model = model.to(device).eval()
  accumulator = SlicedClassificationAccumulator(
      num_classes=model.num_classes, target_class=0)
  all_target_scores = []
  events = []
  errors = []
  frame_offset = 0
  frame_shift_ms = None

  with torch.inference_mode():
    for index, item in enumerate(dataset):
      record = dataset.records[index]
      recipe = recipes[item['id']]
      configured_shift = (
          float(recipe['feature_config']['frame_shift_ms'])
          * int(recipe['feature_config']['subsample_factor']))
      if frame_shift_ms is None:
        frame_shift_ms = configured_shift
      elif frame_shift_ms != configured_shift:
        raise ValueError('All recipes must use the same output frame shift.')

      features = item['features'].unsqueeze(0).to(device)
      embedding = item['embedding'].unsqueeze(0).to(device)
      lengths = torch.tensor([item['length']], device=device)
      logits, output_lengths = model(
          features, embedding, lengths, return_lengths=True)
      output_length = int(output_lengths[0])
      logits = logits[0, :output_length]
      labels = item['labels'][:output_length].cpu().numpy()
      probabilities = logits.softmax(dim=-1).cpu().numpy()
      predictions = probabilities.argmax(axis=-1)
      target_scores = probabilities[:, 0]
      tags = sample_tags(record)

      accumulator.update(
          labels, predictions, target_scores, [tags] * labels.size)
      all_target_scores.append(target_scores)
      events.extend(contiguous_label_events(
          labels, tags, tts_playback='tts_playback' in tags,
          offset=frame_offset))
      errors.append(_error_summary(
          item['id'], labels, predictions, target_scores, tags))
      frame_offset += labels.size

  if frame_shift_ms is None:
    raise ValueError('Cannot evaluate an empty dataset.')
  scores = np.concatenate(all_target_scores)
  errors.sort(key=lambda row: (
      row['frame_error_rate'], row['false_target_frames']), reverse=True)
  return {
      'checkpoint_epoch': None,
      'samples': len(dataset),
      'frames': int(scores.size),
      'frame_shift_ms': frame_shift_ms,
      'target_threshold': float(target_threshold),
      'classification': accumulator.compute(include_curves),
      'events': event_metrics(
          scores, events, frame_shift_ms=frame_shift_ms,
          target_threshold=target_threshold),
      'top_errors': errors[:top_errors],
  }


def main():
  args = parse_args()
  if not 0.0 <= args.target_threshold <= 1.0:
    raise ValueError('--target-threshold must be in [0, 1].')
  if args.top_errors < 0:
    raise ValueError('--top-errors must be non-negative.')

  checkpoint = torch.load(args.checkpoint, map_location='cpu')
  model = Pvad2.load_model_from_package(checkpoint)
  embedding_dim = args.speaker_embedding_dim or model.speaker_embedding_dim
  dataset = PvadManifestDataset(
      args.manifest, speaker_embedding_dim=embedding_dim)
  recipes = load_recipes(dataset)
  report = evaluate(
      model, dataset, recipes, torch.device(args.device),
      target_threshold=args.target_threshold,
      include_curves=args.include_curves, top_errors=args.top_errors)
  report['checkpoint'] = str(Path(args.checkpoint).resolve())
  report['checkpoint_epoch'] = checkpoint.get('epoch')
  report['manifest'] = str(Path(args.manifest).resolve())
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write('\n')


if __name__ == '__main__':
  main()
