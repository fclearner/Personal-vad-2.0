"""Calibrate the target-speech control state machine on a dev manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset import PvadManifestDataset
from evaluate import load_recipes, sample_tags
from features import FeatureFrame, PvadFeatureConfig
from model.pvad2 import Pvad2
from postprocessing import (
    TargetSpeechStateMachine, TargetSpeechStateMachineConfig)


def parse_args():
  parser = argparse.ArgumentParser(
      description='Dev-only calibration for Personal VAD target speech FSM.')
  parser.add_argument('--checkpoint', required=True)
  parser.add_argument('--manifest', required=True)
  parser.add_argument('--output', required=True)
  parser.add_argument('--device', default='cpu')
  parser.add_argument('--activation-threshold', type=float, required=True)
  parser.add_argument(
      '--activation-continue-thresholds', type=float, nargs='+', default=None,
      help=(
          'Candidate thresholds used only after a frame crosses the strong '
          'activation threshold. Defaults to the activation threshold.'))
  parser.add_argument('--min-activation-frames', type=int, required=True)
  parser.add_argument(
      '--release-thresholds', type=float, nargs='+',
      default=(0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50))
  parser.add_argument(
      '--min-release-frames', type=int, nargs='+',
      default=(3, 5, 8, 12, 16, 24, 32))
  parser.add_argument('--max-release-p95-ms', type=float, default=900.0)
  return parser.parse_args()


def _ratio(numerator, denominator):
  return float(numerator / denominator) if denominator else None


def _percentile(values, percentile):
  return float(np.percentile(values, percentile)) if values else None


def _label_events(labels, class_id):
  labels = np.asarray(labels, dtype=np.int64).reshape(-1)
  mask = labels == class_id
  boundaries = np.flatnonzero(
      np.r_[False, mask, False][1:] != np.r_[False, mask, False][:-1])
  return tuple(
      (int(start), int(end))
      for start, end in zip(boundaries[::2], boundaries[1::2]))


def _frame_timing(count, feature_config):
  shift = feature_config.frame_shift_samples
  frames = []
  for index in range(count):
    stack_base = index * feature_config.subsample_factor
    decision_base = stack_base + feature_config.stack_frames - 1
    decision_start = decision_base * shift
    frames.append(FeatureFrame(
        index=index,
        stack_start_sample=stack_base * shift,
        decision_start_sample=decision_start,
        decision_end_sample=(
            decision_start + feature_config.frame_length_samples)))
  return frames


def _evaluate_sample(sample, config, feature_config):
  labels = np.asarray(sample['labels'], dtype=np.int64).reshape(-1)
  probabilities = np.asarray(
      sample['probabilities'], dtype=np.float64)
  if probabilities.shape != (labels.size, 3):
    raise ValueError(
        f"Sample {sample.get('id')} probabilities and labels do not align.")
  processor = TargetSpeechStateMachine(config)
  output = processor.process(
      probabilities, _frame_timing(labels.size, feature_config))
  active = np.asarray(
      [decision.target_active for decision in output.decisions], dtype=bool)
  transitions = output.transitions

  target_events = _label_events(labels, 0)
  detected_events = 0
  detection_latencies = []
  release_eligible = 0
  released_events = 0
  release_latencies = []
  fragmentation_excess = 0
  for start, end in target_events:
    detected = np.flatnonzero(active[start:end])
    if not detected.size:
      continue
    detected_events += 1
    detection_latencies.append(
        float(detected[0] * feature_config.output_shift_ms))
    starts_inside = sum(
        transition.target_activated
        and start <= transition.frame_index < end
        for transition in transitions)
    fragmentation_excess += max(0, starts_inside - 1)
    if end >= labels.size:
      continue
    release_eligible += 1
    released = np.flatnonzero(~active[end:])
    if released.size:
      released_events += 1
      release_latencies.append(
          float(released[0] * feature_config.output_shift_ms))

  false_activation_transitions = sum(
      transition.target_activated
      and labels[transition.frame_index] != 0
      for transition in transitions)
  return {
      'frames': int(labels.size),
      'target_frames': int(np.count_nonzero(labels == 0)),
      'target_active_frames': int(np.count_nonzero(active & (labels == 0))),
      'non_target_frames': int(np.count_nonzero(labels == 1)),
      'non_target_active_frames': int(np.count_nonzero(active & (labels == 1))),
      'non_speech_frames': int(np.count_nonzero(labels == 2)),
      'non_speech_active_frames': int(np.count_nonzero(active & (labels == 2))),
      'target_events': len(target_events),
      'target_events_detected': detected_events,
      'detection_latencies_ms': detection_latencies,
      'release_eligible_events': release_eligible,
      'released_events': released_events,
      'release_latencies_ms': release_latencies,
      'fragmentation_excess': fragmentation_excess,
      'false_activation_transitions': false_activation_transitions,
  }


def _merge_sample_metrics(rows):
  total_keys = (
      'frames', 'target_frames', 'target_active_frames',
      'non_target_frames', 'non_target_active_frames',
      'non_speech_frames', 'non_speech_active_frames',
      'target_events', 'target_events_detected',
      'release_eligible_events', 'released_events',
      'fragmentation_excess', 'false_activation_transitions')
  totals = {
      key: sum(int(row[key]) for row in rows)
      for key in total_keys}
  detection_latencies = [
      value for row in rows for value in row['detection_latencies_ms']]
  release_latencies = [
      value for row in rows for value in row['release_latencies_ms']]
  return {
      **totals,
      'target_frame_recall': _ratio(
          totals['target_active_frames'], totals['target_frames']),
      'non_target_active_rate': _ratio(
          totals['non_target_active_frames'], totals['non_target_frames']),
      'non_speech_active_rate': _ratio(
          totals['non_speech_active_frames'], totals['non_speech_frames']),
      'target_event_recall': _ratio(
          totals['target_events_detected'], totals['target_events']),
      'detection_latency_p50_ms': _percentile(detection_latencies, 50),
      'detection_latency_p95_ms': _percentile(detection_latencies, 95),
      'release_success_rate': _ratio(
          totals['released_events'], totals['release_eligible_events']),
      'release_latency_p50_ms': _percentile(release_latencies, 50),
      'release_latency_p95_ms': _percentile(release_latencies, 95),
  }


def evaluate_candidate(samples, config, feature_config=None, include_slices=False):
  """Evaluate one state-machine configuration without joining sample streams."""
  feature_config = feature_config or PvadFeatureConfig()
  rows = [
      _evaluate_sample(sample, config, feature_config)
      for sample in samples]
  result = {
      'config': config.to_dict(),
      'metrics': _merge_sample_metrics(rows),
  }
  if include_slices:
    tags = sorted({
        tag for sample in samples for tag in sample.get('tags', ())})
    result['slices'] = {}
    for tag in tags:
      selected = [
          row for sample, row in zip(samples, rows)
          if tag in sample.get('tags', ())]
      result['slices'][tag] = _merge_sample_metrics(selected)
  return result


def _metric(candidate, name, default):
  value = candidate['metrics'].get(name)
  return default if value is None else value


def select_candidate(candidates, max_release_p95_ms):
  """Recall-first selection under an explicit endpoint-latency constraint."""
  if not candidates:
    raise ValueError('At least one calibration candidate is required.')
  if max_release_p95_ms <= 0:
    raise ValueError('max_release_p95_ms must be positive.')

  best_event_recall = max(
      _metric(candidate, 'target_event_recall', -1.0)
      for candidate in candidates)
  recall_matched = [
      candidate for candidate in candidates
      if _metric(candidate, 'target_event_recall', -1.0)
      >= best_event_recall - 1e-12]
  best_release_success = max(
      _metric(candidate, 'release_success_rate', 1.0)
      for candidate in recall_matched)
  release_matched = [
      candidate for candidate in recall_matched
      if _metric(candidate, 'release_success_rate', 1.0)
      >= best_release_success - 1e-12]
  feasible = [
      candidate for candidate in release_matched
      if (_metric(candidate, 'release_latency_p95_ms', 0.0)
          <= max_release_p95_ms)]
  meets_constraint = bool(feasible)
  pool = feasible or release_matched

  def ranking_key(candidate):
    config = candidate['config']
    return (
        -_metric(candidate, 'target_frame_recall', -1.0),
        _metric(candidate, 'fragmentation_excess', float('inf')),
        _metric(candidate, 'non_target_active_rate', float('inf')),
        _metric(candidate, 'non_speech_active_rate', float('inf')),
        _metric(candidate, 'release_latency_p95_ms', float('inf')),
        int(config['min_release_frames']),
        -float(config['release_threshold']),
    )

  selected = min(pool, key=ranking_key)
  return selected, {
      'meets_release_latency_constraint': meets_constraint,
      'max_release_p95_ms': float(max_release_p95_ms),
      'best_target_event_recall': float(best_event_recall),
      'best_release_success_rate': float(best_release_success),
      'feasible_candidates': len(feasible),
      'candidate_count': len(candidates),
  }


def collect_model_outputs(model, dataset, device):
  model = model.to(device).eval()
  samples = []
  with torch.inference_mode():
    for index, item in enumerate(dataset):
      features = item['features'].unsqueeze(0).to(device)
      embedding = item['embedding'].unsqueeze(0).to(device)
      lengths = torch.tensor([item['length']], device=device)
      logits, output_lengths = model(
          features, embedding, lengths, return_lengths=True)
      output_length = int(output_lengths[0])
      samples.append({
          'id': item['id'],
          'labels': item['labels'][:output_length].cpu().numpy(),
          'probabilities': (
              logits[0, :output_length].softmax(dim=-1).cpu().numpy()),
          'tags': sample_tags(dataset.records[index]),
      })
  return samples


def _feature_config_from_recipes(recipes):
  configs = {
      json.dumps(recipe['feature_config'], sort_keys=True)
      for recipe in recipes.values()}
  if len(configs) != 1:
    raise ValueError('All dev recipes must use one feature configuration.')
  return PvadFeatureConfig(**json.loads(configs.pop()))


def main():
  args = parse_args()
  if args.min_activation_frames <= 0:
    raise ValueError('--min-activation-frames must be positive.')
  if args.max_release_p95_ms <= 0:
    raise ValueError('--max-release-p95-ms must be positive.')

  checkpoint_path = Path(args.checkpoint).resolve()
  manifest_path = Path(args.manifest).resolve()
  checkpoint = torch.load(checkpoint_path, map_location='cpu')
  model = Pvad2.load_model_from_package(checkpoint)
  dataset = PvadManifestDataset(
      manifest_path, speaker_embedding_dim=model.speaker_embedding_dim)
  split_values = {
      record.get('split') for record in dataset.records if record.get('split')}
  if split_values and split_values != {'dev'}:
    raise ValueError(
        f'Calibration manifest must contain only dev records: {split_values}')
  recipes = load_recipes(dataset)
  feature_config = _feature_config_from_recipes(recipes)
  samples = collect_model_outputs(model, dataset, torch.device(args.device))

  candidates = []
  activation_continue_thresholds = (
      args.activation_continue_thresholds or (args.activation_threshold,))
  for activation_continue_threshold in activation_continue_thresholds:
    for release_threshold in args.release_thresholds:
      if release_threshold > activation_continue_threshold:
        continue
      for min_release_frames in args.min_release_frames:
        config = TargetSpeechStateMachineConfig(
            activation_threshold=args.activation_threshold,
            activation_continue_threshold=activation_continue_threshold,
            release_threshold=release_threshold,
            min_activation_frames=args.min_activation_frames,
            min_release_frames=min_release_frames,
            sample_rate=feature_config.sample_rate)
        candidates.append(evaluate_candidate(samples, config, feature_config))
  selected, selection = select_candidate(
      candidates, args.max_release_p95_ms)
  selected_with_slices = evaluate_candidate(
      samples,
      TargetSpeechStateMachineConfig.from_dict(selected['config']),
      feature_config,
      include_slices=True)

  report = {
      'schema_version': 2,
      'status': 'dev_calibrated_control_candidate',
      'selection_objective': (
          'target_event_recall_then_release_success_then_target_frame_recall'),
      'checkpoint': str(checkpoint_path),
      'checkpoint_epoch': checkpoint.get('epoch'),
      'dev_manifest': str(manifest_path),
      'feature_config': feature_config.to_dict(),
      'target_speech_state_machine': selected['config'],
      'selected_metrics': selected_with_slices['metrics'],
      'selected_slices': selected_with_slices['slices'],
      'selection': selection,
      'candidates': candidates,
      'constraints': {
          'dev_only': True,
          'test_set_used': False,
          'online_actions_enabled': False,
      },
      'notes': [
          'Personal VAD identity remains separate from valid/invalid and '
          'semantic-end.',
          'Runtime control must remain disabled until replay and service smoke '
          'tests pass.',
      ],
  }
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open('w', encoding='utf-8') as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write('\n')


if __name__ == '__main__':
  main()
