from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import format_metric, selection_score


def test_selection_score_directions_and_missing_target():
  metrics = {
      'loss': 0.25,
      'macro_f1': 0.75,
      'per_class': {
          'target': {'f1': 0.8},
          'non_target': {'f1': 0.7},
          'non_speech': {'f1': 0.75},
      },
  }
  assert selection_score(metrics, 'loss') == -0.25
  assert selection_score(metrics, 'macro_f1') == 0.75
  assert selection_score(metrics, 'target_f1') == 0.8
  assert format_metric(0.8) == '0.8000'
  assert format_metric(None) == 'n/a'

  metrics['per_class']['target']['f1'] = None
  assert selection_score(metrics, 'target_f1') == -math.inf


if __name__ == '__main__':
  test_selection_score_directions_and_missing_target()
  print('train metrics tests ok')
