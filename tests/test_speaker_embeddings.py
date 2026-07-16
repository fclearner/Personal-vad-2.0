from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speaker_backends.modelscope_export import (aggregate_embeddings,
                                                l2_normalize)


def test_l2_normalize_rejects_degenerate_vectors():
  np.testing.assert_allclose(l2_normalize([3.0, 4.0]), [0.6, 0.8])
  for vector in ([], [0.0, 0.0], [np.nan, 1.0]):
    try:
      l2_normalize(vector)
    except ValueError:
      pass
    else:
      raise AssertionError(f'Expected ValueError for {vector}')


def test_aggregate_normalizes_before_and_after_mean():
  aggregate = aggregate_embeddings([[2.0, 0.0], [20.0, 0.0]])
  np.testing.assert_allclose(aggregate, [1.0, 0.0])
  balanced = aggregate_embeddings([[1.0, 0.0], [0.0, 1.0]])
  np.testing.assert_allclose(
      balanced, [2 ** -0.5, 2 ** -0.5], rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  test_l2_normalize_rejects_degenerate_vectors()
  test_aggregate_normalizes_before_and_after_mean()
  print('speaker embedding tests ok')
