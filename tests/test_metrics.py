from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import (ClassificationAccumulator,
                     SlicedClassificationAccumulator, event_metrics)


def test_three_class_metrics_and_target_curves():
  labels = np.array([0, 0, 1, 1, 2, 2])
  predictions = np.array([0, 1, 1, 2, 2, 0])
  scores = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.0])
  accumulator = ClassificationAccumulator(num_score_bins=100)
  accumulator.update(labels, predictions, scores)
  result = accumulator.compute(include_curves=True)
  assert result['confusion_matrix'] == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
  assert result['accuracy'] == 0.5
  assert result['macro_f1'] == 0.5
  assert result['per_class']['target']['support'] == 2
  assert result['target_vs_rest']['average_precision'] == 1.0
  assert result['target_vs_rest']['roc_auc'] == 1.0


def test_nonexclusive_slice_metrics():
  accumulator = SlicedClassificationAccumulator(num_score_bins=10)
  accumulator.update(
      labels=[0, 1, 2], predictions=[0, 0, 2],
      target_scores=[0.9, 0.7, 0.1],
      slice_tags=[('nearfield',), ('nearfield', 'wham'), ('wham',)])
  result = accumulator.compute()
  assert result['overall']['frames'] == 3
  assert result['slices']['nearfield']['frames'] == 2
  assert result['slices']['wham']['frames'] == 2


def test_event_control_metrics_and_slices():
  scores = np.zeros(20)
  scores[3] = 0.9
  scores[7] = 0.8
  scores[12] = 0.7
  events = [
      {'event_type': 'target', 'start_frame': 2, 'end_frame': 6,
       'tags': ['nearfield']},
      {'event_type': 'non_target', 'start_frame': 6, 'end_frame': 10,
       'tags': ['nearfield', 'wham']},
      {'event_type': 'tts_playback', 'start_frame': 10, 'end_frame': 14,
       'tags': ['tts_playback']},
      {'event_type': 'target', 'start_frame': 14, 'end_frame': 19,
       'tags': ['farfield']},
  ]
  result = event_metrics(
      scores, events, frame_shift_ms=30.0, duration_seconds=0.6)
  overall = result['overall']
  assert overall['target_event_recall'] == 0.5
  assert overall['third_party_false_activation_rate'] == 1.0
  assert overall['false_interruption_rate'] == 1.0
  assert overall['detection_latency_p50_ms'] == 30.0
  assert overall['detection_latency_p95_ms'] == 30.0
  assert overall['false_activations'] == 2
  assert result['slices']['farfield']['target_event_recall'] == 0.0
  assert result['slices']['wham']['third_party_false_activation_rate'] == 1.0


if __name__ == '__main__':
  test_three_class_metrics_and_target_curves()
  test_nonexclusive_slice_metrics()
  test_event_control_metrics_and_slices()
  print('metrics tests ok')
