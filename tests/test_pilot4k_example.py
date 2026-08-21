import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.pilot4k_speaker_aware_inference import (
    CALIBRATION_SHA256, CHECKPOINT_SHA256, DEFAULT_CALIBRATION,
    DEFAULT_CHECKPOINT, aggregate_enrollment_embeddings, load_calibration,
    load_checkpoint, run_inference, sha256_file)


def test_transfer_manifest_checksums():
  manifest = json.loads(
      (ROOT / 'TRANSFER_MANIFEST.json').read_text(encoding='utf-8'))
  for record in manifest['files']:
    path = ROOT / record['path']
    assert path.stat().st_size == record['size_bytes']
    assert sha256_file(path) == record['sha256']


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
  target_fsm_config, calibration, calibration_sha256 = load_calibration(
      DEFAULT_CALIBRATION)
  assert calibration_sha256 == CALIBRATION_SHA256
  assert calibration['constraints']['test_set_used'] is False
  features = torch.randn(12, 512, generator=torch.Generator().manual_seed(17))
  result = run_inference(
      model, features, embedding, list(package['class_names']), device,
      include_frames=True, target_fsm_config=target_fsm_config)

  assert result['frame_count'] == 12
  assert result['frame_shift_ms'] == 30.0
  assert len(result['frame_predictions']) == 12
  assert sum(segment['frames'] for segment in result['segments']) == 12
  assert result['target_speech_fsm']['config']['activation_threshold'] == 0.55
  assert 'segments' in result['target_speech_fsm']
  for frame in result['frame_predictions']:
    assert abs(sum(frame['probabilities']) - 1.0) < 1e-5


if __name__ == '__main__':
  with tempfile.TemporaryDirectory() as directory:
    test_checkpoint_and_speaker_aware_example(Path(directory))
  print('pilot4k example test ok')
