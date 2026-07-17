from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train import (epoch_checkpoint_name, format_metric, selection_score,
                   summarize_runtime)


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


def test_runtime_summary_reports_throughput_and_memory():
  result = summarize_runtime(
      elapsed_seconds=2.0, steps=4, examples=8, frames=1600,
      peak_allocated_bytes=128 * 1024 ** 2,
      peak_reserved_bytes=256 * 1024 ** 2)
  assert result['steps_per_second'] == 2.0
  assert result['examples_per_second'] == 4.0
  assert result['frames_per_second'] == 800.0
  assert result['peak_memory_allocated_mb'] == 128.0
  assert result['peak_memory_reserved_mb'] == 256.0


def test_epoch_checkpoint_name_is_stable_and_rejects_invalid_epochs():
  assert epoch_checkpoint_name(3) == 'epoch-0003.pt'
  try:
    epoch_checkpoint_name(0)
  except ValueError:
    pass
  else:
    raise AssertionError('Non-positive epochs must be rejected.')


if __name__ == '__main__':
  test_selection_score_directions_and_missing_target()
  print('train metrics tests ok')
