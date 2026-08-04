from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.pilot4k_speaker_aware_inference import (
    CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, aggregate_enrollment_embeddings,
    load_checkpoint, run_inference, sha256_file)


def test_checkpoint_and_speaker_aware_example(tmp_path):
  assert sha256_file(DEFAULT_CHECKPOINT) == CHECKPOINT_SHA256
  embedding_paths = []
  for index in range(2):
    path = tmp_path / f'enrollment_{index}.npy'
    vector = np.zeros(192, dtype=np.float32)
    vector[index] = 1.0
    np.save(path, vector)
    embedding_paths.append(path)

  embedding = aggregate_enrollment_embeddings(embedding_paths)
  torch.testing.assert_close(embedding.norm(), torch.tensor(1.0))

  device = torch.device('cpu')
  model, package = load_checkpoint(DEFAULT_CHECKPOINT, device)
  features = torch.randn(12, 512, generator=torch.Generator().manual_seed(17))
  result = run_inference(
      model, features, embedding, list(package['class_names']), device,
      include_frames=True)

  assert result['frame_count'] == 12
  assert result['frame_shift_ms'] == 30.0
  assert len(result['frame_predictions']) == 12
  assert sum(segment['frames'] for segment in result['segments']) == 12
  for frame in result['frame_predictions']:
    assert abs(sum(frame['probabilities']) - 1.0) < 1e-5


if __name__ == '__main__':
  with tempfile.TemporaryDirectory() as directory:
    test_checkpoint_and_speaker_aware_example(Path(directory))
  print('pilot4k example test ok')
