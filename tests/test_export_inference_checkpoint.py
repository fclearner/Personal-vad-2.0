from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_inference_checkpoint import (
    EXPORT_KEYS, export_checkpoint, load_training_checkpoint, sha256_file)
from model.pvad2 import Pvad2


def test_export_checkpoint_removes_training_only_fields(tmp_path):
  model = Pvad2(speaker_embedding_dim=192)
  source = Pvad2.serialize(model, epoch=7, tr_loss=0.2, cv_loss=0.3)
  source.update({
      'optim_dict': {'state': {}, 'param_groups': []},
      'args': {
          'train_manifest': '/private/data/train.jsonl',
          'output_dir': '/private/runs/pvad2',
      },
      'selection_metric': 'target_f1',
      'best_selection_score': 0.75,
      'valid_metrics': {
          'loss': 0.3,
          'macro_f1': 0.7,
          'per_class': {'target': {'f1': 0.75}},
      },
  })
  source_path = tmp_path / 'best.pt'
  output_path = tmp_path / 'best_inference.pt'
  torch.save(source, source_path)

  report = export_checkpoint(source_path, output_path)
  exported = load_training_checkpoint(output_path)

  assert set(exported) == EXPORT_KEYS
  assert exported['source_checkpoint_sha256'] == sha256_file(source_path)
  assert exported['selection_score'] == 0.75
  assert report['output_sha256'] == sha256_file(output_path)
  assert 'args' not in exported
  assert 'optim_dict' not in exported
  assert 'valid_metrics' not in exported
  restored = Pvad2.load_model_from_package(exported)
  assert restored.speaker_embedding_dim == 192
