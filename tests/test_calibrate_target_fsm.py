from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calibrate_target_fsm import evaluate_candidate, select_candidate
from features import PvadFeatureConfig
from postprocessing import TargetSpeechStateMachineConfig


def _probabilities(target_scores):
  return np.asarray([
      [score, (1.0 - score) * 0.5, (1.0 - score) * 0.5]
      for score in target_scores], dtype=np.float64)


def _config(min_release_frames):
  return TargetSpeechStateMachineConfig(
      activation_threshold=0.55,
      release_threshold=0.25,
      min_activation_frames=2,
      min_release_frames=min_release_frames)


def test_candidate_metrics_keep_samples_as_independent_streams():
  samples = [
      {
          'id': 'target',
          'labels': np.asarray([0, 0, 0, 0, 0, 2, 2, 2]),
          'probabilities': _probabilities(
              [0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1]),
          'tags': ('target_only',),
      },
      {
          'id': 'third-party',
          'labels': np.asarray([1, 1, 1, 2]),
          'probabilities': _probabilities([0.1, 0.2, 0.1, 0.1]),
          'tags': ('nearfield',),
      },
  ]
  report = evaluate_candidate(
      samples, _config(min_release_frames=2),
      PvadFeatureConfig(), include_slices=True)
  metrics = report['metrics']
  assert metrics['target_event_recall'] == 1.0
  assert metrics['target_frame_recall'] == 0.8
  assert metrics['fragmentation_excess'] == 0
  assert metrics['false_activation_transitions'] == 0
  assert metrics['release_success_rate'] == 1.0
  assert report['slices']['nearfield']['non_target_active_rate'] == 0.0


def test_recall_first_selection_preserves_target_through_short_dropout():
  samples = [{
      'id': 'dropout',
      'labels': np.asarray([0, 0, 0, 0, 0, 2, 2, 2]),
      'probabilities': _probabilities(
          [0.8, 0.8, 0.1, 0.8, 0.8, 0.1, 0.1, 0.1]),
      'tags': ('target_only',),
  }]
  short = evaluate_candidate(samples, _config(min_release_frames=1))
  hangover = evaluate_candidate(samples, _config(min_release_frames=2))
  selected, selection = select_candidate(
      [short, hangover], max_release_p95_ms=900.0)
  assert selected['config']['min_release_frames'] == 2
  assert selection['meets_release_latency_constraint']
  assert selection['feasible_candidates'] == 2


def test_selection_reports_when_release_constraint_is_unmet():
  candidate = {
      'config': _config(min_release_frames=2).to_dict(),
      'metrics': {
          'target_event_recall': 1.0,
          'target_frame_recall': 1.0,
          'release_success_rate': 1.0,
          'release_latency_p95_ms': 1200.0,
          'fragmentation_excess': 0,
          'non_target_active_rate': 0.0,
          'non_speech_active_rate': 0.0,
      },
  }
  selected, selection = select_candidate(
      [candidate], max_release_p95_ms=900.0)
  assert selected is candidate
  assert not selection['meets_release_latency_constraint']


def test_recall_tolerance_prefers_safer_candidate_inside_recall_band():
  best_recall = {
      'config': _config(min_release_frames=2).to_dict(),
      'metrics': {
          'target_event_recall': 1.0,
          'target_frame_recall': 0.9113,
          'release_success_rate': 1.0,
          'release_latency_p95_ms': 600.0,
          'fragmentation_excess': 0,
          'non_target_active_rate': 0.2124,
          'non_speech_active_rate': 0.0276,
      },
  }
  safer = {
      'config': {
          **_config(min_release_frames=2).to_dict(),
          'activation_continue_threshold': 0.35,
      },
      'metrics': {
          **best_recall['metrics'],
          'target_frame_recall': 0.9106,
          'non_target_active_rate': 0.2104,
          'non_speech_active_rate': 0.0273,
      },
  }
  exact, _ = select_candidate(
      [best_recall, safer], max_release_p95_ms=900.0)
  tolerant, selection = select_candidate(
      [best_recall, safer],
      max_release_p95_ms=900.0,
      target_frame_recall_tolerance=0.001)
  assert exact is best_recall
  assert tolerant is safer
  assert selection['recall_tolerant_candidates'] == 2
  assert np.isclose(selection['target_frame_recall_floor'], 0.9103)
