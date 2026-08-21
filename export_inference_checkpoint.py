#!/usr/bin/env python3
"""Export a training checkpoint as a minimal inference-only package."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from model.pvad2 import Pvad2


CLASS_NAMES = ('target', 'non_target', 'non_speech')
EXPORT_FORMAT_VERSION = 1
EXPORT_KEYS = {
    'state_dict', 'model_config', 'epoch', 'selection_metric',
    'selection_score', 'class_names', 'source_checkpoint_sha256',
    'export_format_version',
}


def parse_args():
  parser = argparse.ArgumentParser(
      description='Strip optimizer, paths, and training metadata from best.pt.')
  parser.add_argument('--checkpoint', required=True,
                      help='Trusted training checkpoint produced by train.py.')
  parser.add_argument('--output', required=True,
                      help='Destination for the inference-only .pt package.')
  return parser.parse_args()


def sha256_file(path: str | Path) -> str:
  """Return the lowercase SHA256 digest of one file."""

  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def load_training_checkpoint(path: str | Path):
  """Load a trusted torch checkpoint with the safest supported API."""

  try:
    return torch.load(path, map_location='cpu', weights_only=True)
  except TypeError:  # PyTorch 1.x compatibility.
    return torch.load(path, map_location='cpu')


def _derived_selection_score(checkpoint, selection_metric):
  metrics = checkpoint.get('valid_metrics') or checkpoint.get('train_metrics')
  if not metrics:
    raise ValueError(
        'Checkpoint has no best_selection_score or train/validation metrics.')
  if selection_metric == 'target_f1':
    return float(metrics['per_class']['target']['f1'])
  if selection_metric == 'macro_f1':
    return float(metrics['macro_f1'])
  if selection_metric == 'loss':
    return -float(metrics['loss'])
  raise ValueError(f'Unsupported selection metric: {selection_metric}')


def build_inference_package(checkpoint, source_sha256):
  """Build and validate the public, inference-only checkpoint dictionary."""

  required = {'state_dict', 'model_config', 'epoch'}
  missing = required - set(checkpoint)
  if missing:
    raise ValueError(f'Training checkpoint is missing: {sorted(missing)}')

  model_config = dict(checkpoint['model_config'])
  model = Pvad2(**model_config)
  model.load_state_dict(checkpoint['state_dict'])
  if model.num_classes != len(CLASS_NAMES):
    raise ValueError(
        f'Expected {len(CLASS_NAMES)} output classes, got {model.num_classes}.')

  selection_metric = checkpoint.get('selection_metric', 'target_f1')
  selection_score = checkpoint.get('best_selection_score')
  if selection_score is None:
    selection_score = _derived_selection_score(checkpoint, selection_metric)
  selection_score = float(selection_score)
  if not math.isfinite(selection_score):
    raise ValueError('Selection score must be finite.')

  package = {
      'state_dict': {
          name: value.detach().cpu()
          for name, value in checkpoint['state_dict'].items()
      },
      'model_config': model_config,
      'epoch': int(checkpoint['epoch']),
      'selection_metric': selection_metric,
      'selection_score': selection_score,
      'class_names': list(CLASS_NAMES),
      'source_checkpoint_sha256': source_sha256,
      'export_format_version': EXPORT_FORMAT_VERSION,
  }
  if set(package) != EXPORT_KEYS:
    raise AssertionError('Inference package whitelist changed unexpectedly.')
  return package


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path):
  """Export one trusted training checkpoint and return its public metadata."""

  checkpoint_path = Path(checkpoint_path).resolve()
  output_path = Path(output_path).resolve()
  source_sha256 = sha256_file(checkpoint_path)
  package = build_inference_package(
      load_training_checkpoint(checkpoint_path), source_sha256)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(package, output_path)
  return {
      'source_checkpoint_sha256': source_sha256,
      'output_sha256': sha256_file(output_path),
      'output_size_bytes': output_path.stat().st_size,
      'epoch': package['epoch'],
      'selection_metric': package['selection_metric'],
      'selection_score': package['selection_score'],
      'exported_keys': sorted(package),
  }


def main():
  args = parse_args()
  report = export_checkpoint(args.checkpoint, args.output)
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
