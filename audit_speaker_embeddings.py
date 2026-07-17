"""Audit CAM++ enrollment/current speaker separation for generated recipes."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from speaker_backends.modelscope_export import (
    aggregate_embeddings, l2_normalize, load_pipeline)


def parse_args():
  parser = argparse.ArgumentParser(
      description='Audit CAM++ enrollment stability and speaker separation.')
  parser.add_argument('--recipes', required=True)
  parser.add_argument('--campplus-model', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--split', default='dev')
  parser.add_argument('--current-per-speaker', type=int, default=2)
  parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
  return parser.parse_args()


def read_recipes(path):
  recipes = []
  with Path(path).open('r', encoding='utf-8') as handle:
    for line in handle:
      line = line.strip()
      if line:
        recipes.append(json.loads(line))
  return recipes


def collect_speaker_audio(recipes, split, current_per_speaker):
  """Collect disjoint enrollment/current paths from one recipe split."""
  if current_per_speaker <= 0:
    raise ValueError('current_per_speaker must be positive.')
  enrollment = {}
  current = defaultdict(list)
  for recipe in recipes:
    if recipe.get('split') != split:
      continue
    target = recipe['target_speaker']
    paths = tuple(recipe['enrollment_paths'])
    if target in enrollment and enrollment[target] != paths:
      raise ValueError(f'Inconsistent enrollment paths for {target}.')
    enrollment[target] = paths
    for component in recipe.get('components', ()):
      speaker = component.get('speaker_id')
      if (component.get('role') not in {'target', 'non_target'}
          or not speaker or speaker == 'qwen3_omni_tts'):
        continue
      source_path = component['source_path']
      if source_path not in current[speaker]:
        current[speaker].append(source_path)

  if not enrollment:
    raise ValueError(f'No enrollment speakers found for split {split}.')
  missing = sorted(set(enrollment) - set(current))
  if missing:
    raise ValueError(f'No current audio found for speakers: {missing}')

  selected = {}
  for speaker in sorted(enrollment):
    enrollment_paths = enrollment[speaker]
    current_paths = tuple(current[speaker][:current_per_speaker])
    if set(enrollment_paths) & set(current_paths):
      raise ValueError(f'Enrollment/current leakage for {speaker}.')
    selected[speaker] = {
        'enrollment': enrollment_paths,
        'current': current_paths,
    }
  return selected


def summarize_scores(values):
  values = np.asarray(values, dtype=np.float64)
  if values.size == 0:
    return {'count': 0}
  percentiles = np.percentile(values, [0, 5, 50, 95, 100])
  return {
      'count': int(values.size),
      'min': float(percentiles[0]),
      'p05': float(percentiles[1]),
      'median': float(percentiles[2]),
      'p95': float(percentiles[3]),
      'max': float(percentiles[4]),
      'mean': float(values.mean()),
  }


def embedding_metrics(enrollment_embeddings, current_embeddings):
  """Compute verification distributions and closed-set top-1 retrieval."""
  speakers = sorted(enrollment_embeddings)
  if set(speakers) != set(current_embeddings):
    raise ValueError('Enrollment and current speaker sets must match.')
  if len(speakers) < 2:
    raise ValueError('At least two speakers are required for separation audit.')
  aggregates = np.stack([
      aggregate_embeddings(enrollment_embeddings[speaker])
      for speaker in speakers])

  same_scores = []
  impostor_scores = []
  enrollment_pair_scores = []
  top1_correct = 0
  top1_margins = []
  current_count = 0
  per_speaker = {}
  for speaker_index, speaker in enumerate(speakers):
    enrollments = np.stack([
        l2_normalize(vector) for vector in enrollment_embeddings[speaker]])
    currents = np.stack([
        l2_normalize(vector) for vector in current_embeddings[speaker]])
    if enrollments.shape[0] >= 2:
      enrollment_pair_scores.append(float(enrollments[0] @ enrollments[1]))
    speaker_same = []
    for vector in currents:
      scores = aggregates @ vector
      same = float(scores[speaker_index])
      other = np.delete(scores, speaker_index)
      same_scores.append(same)
      speaker_same.append(same)
      impostor_scores.extend(float(value) for value in other)
      prediction = int(np.argmax(scores))
      top1_correct += prediction == speaker_index
      top1_margins.append(same - float(np.max(other)))
      current_count += 1
    per_speaker[speaker] = {
        'current_utterances': int(currents.shape[0]),
        'aggregate_to_current': summarize_scores(speaker_same),
    }

  impostor_p99 = float(np.percentile(impostor_scores, 99))
  return {
      'speakers': len(speakers),
      'current_utterances': current_count,
      'enrollment_pair_cosine': summarize_scores(enrollment_pair_scores),
      'same_aggregate_to_current_cosine': summarize_scores(same_scores),
      'impostor_aggregate_to_current_cosine': summarize_scores(impostor_scores),
      'same_at_or_below_impostor_p99_rate': float(
          np.mean(np.asarray(same_scores) <= impostor_p99)),
      'closed_set_top1_accuracy': float(top1_correct / current_count),
      'closed_set_top1_margin': summarize_scores(top1_margins),
      'per_speaker': per_speaker,
  }


def _extract_embedding(pipeline, audio_path):
  result = pipeline([str(audio_path)], output_emb=True)
  matrix = np.asarray(result['embs'], dtype=np.float32)
  if matrix.ndim != 2 or matrix.shape[0] != 1:
    raise RuntimeError(
        f'Unexpected CAM++ embedding shape for {audio_path}: {matrix.shape}')
  return l2_normalize(matrix[0])


def main():
  args = parse_args()
  recipes_path = Path(args.recipes).resolve()
  model_path = Path(args.campplus_model).resolve()
  selected = collect_speaker_audio(
      read_recipes(recipes_path), args.split, args.current_per_speaker)
  pipeline = load_pipeline(str(model_path), device=args.device)
  enrollment_embeddings = {}
  current_embeddings = {}
  sources = {}
  for speaker, paths in selected.items():
    enrollment_embeddings[speaker] = [
        _extract_embedding(pipeline, path) for path in paths['enrollment']]
    current_embeddings[speaker] = [
        _extract_embedding(pipeline, path) for path in paths['current']]
    sources[speaker] = {
        'enrollment': list(paths['enrollment']),
        'current': list(paths['current']),
    }

  report = embedding_metrics(enrollment_embeddings, current_embeddings)
  report.update({
      'recipes': str(recipes_path),
      'split': args.split,
      'campplus_model': str(model_path),
      'device': args.device,
      'sources': sources,
  })
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write('\n')


if __name__ == '__main__':
  main()
