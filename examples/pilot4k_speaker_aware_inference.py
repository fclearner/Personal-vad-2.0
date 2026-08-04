#!/usr/bin/env python3
"""Run the sanitized pilot4k speaker-aware Personal VAD checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import PvadFeatureExtractor, load_audio
from model.pvad2 import Pvad2


DEFAULT_CHECKPOINT = (
    ROOT / 'checkpoints' / 'pilot4k_speaker_aware_epoch38'
    / 'best_inference.pt')
CHECKPOINT_SHA256 = (
    '142419ebd37fb0a160acc3c75c0571cf8d0c420f2535fb450a5cbb3f7add3753')
PACKAGE_KEYS = {
    'state_dict', 'model_config', 'epoch', 'selection_metric',
    'selection_score', 'class_names', 'source_checkpoint_sha256',
    'export_format_version',
}


def parse_args():
  parser = argparse.ArgumentParser(
      description='Run the pilot4k speaker-aware Personal VAD example.')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
      '--audio', help='Current 16 kHz mono audio (other rates are resampled).')
  source.add_argument(
      '--features', help='Precomputed float32 NumPy array with shape (T, 512).')
  parser.add_argument(
      '--embedding', action='append', required=True,
      help='CAM++ .npy enrollment embedding. Repeat for multiple clips.')
  parser.add_argument('--checkpoint', default=str(DEFAULT_CHECKPOINT))
  parser.add_argument('--device', default='cpu')
  parser.add_argument(
      '--output', help='Optional JSON output path. Defaults to stdout.')
  parser.add_argument(
      '--include-frames', action='store_true',
      help='Include every frame posterior in addition to merged segments.')
  return parser.parse_args()


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def load_checkpoint(path: str | Path, device: torch.device):
  path = Path(path)
  actual_sha256 = sha256_file(path)
  if actual_sha256 != CHECKPOINT_SHA256:
    raise ValueError(
        f'Checkpoint SHA256 mismatch: expected {CHECKPOINT_SHA256}, '
        f'got {actual_sha256}.')
  try:
    package = torch.load(path, map_location=device, weights_only=True)
  except TypeError:  # PyTorch 1.x compatibility.
    package = torch.load(path, map_location=device)
  if set(package) != PACKAGE_KEYS:
    raise ValueError(
        f'Checkpoint field mismatch; missing '
        f'{sorted(PACKAGE_KEYS - set(package))}, extra '
        f'{sorted(set(package) - PACKAGE_KEYS)}.')
  model = Pvad2.load_model_from_package(package).to(device).eval()
  return model, package


def load_features(path: str | Path) -> torch.Tensor:
  features = np.load(path, allow_pickle=False)
  if isinstance(features, np.lib.npyio.NpzFile):
    if len(features.files) != 1:
      raise ValueError('Feature .npz must contain exactly one array.')
    features = features[features.files[0]]
  features = torch.as_tensor(features, dtype=torch.float32)
  if features.dim() != 2 or features.size(1) != 512:
    raise ValueError(
        f'Features must have shape (T, 512), got {tuple(features.shape)}.')
  return features


def aggregate_enrollment_embeddings(paths) -> torch.Tensor:
  embeddings = []
  for path in paths:
    array = np.load(path, allow_pickle=False)
    embedding = torch.as_tensor(array, dtype=torch.float32).squeeze()
    if embedding.shape != (192,):
      raise ValueError(
          f'CAM++ embedding {path} must have shape (192,), '
          f'got {tuple(embedding.shape)}.')
    if not torch.isfinite(embedding).all() or embedding.norm() <= 0:
      raise ValueError(f'CAM++ embedding {path} is not a finite non-zero vector.')
    embeddings.append(F.normalize(embedding, dim=0))
  mean_embedding = torch.stack(embeddings).mean(dim=0)
  if mean_embedding.norm() <= 0:
    raise ValueError('The mean CAM++ enrollment embedding is zero.')
  return F.normalize(mean_embedding, dim=0)


def merge_predictions(probabilities, frames, class_names, sample_rate=16000):
  labels = probabilities.argmax(dim=-1).tolist()
  segments = []
  start = 0
  for end in range(1, len(labels) + 1):
    if end < len(labels) and labels[end] == labels[start]:
      continue
    class_id = labels[start]
    segments.append({
        'class_id': class_id,
        'class_name': class_names[class_id],
        'start_ms': round(
            frames[start].decision_start_sample * 1000.0 / sample_rate, 3),
        'end_ms': round(
            frames[end - 1].decision_end_sample * 1000.0 / sample_rate, 3),
        'frames': end - start,
        'mean_class_probability': round(float(
            probabilities[start:end, class_id].mean()), 6),
    })
    start = end
  return labels, segments


def run_inference(model, features, embedding, class_names, device,
                  include_frames=False):
  if features.size(0) == 0:
    raise ValueError('No feature frames were produced.')
  extractor = PvadFeatureExtractor(device='cpu')
  frames = extractor.frame_timing(features.size(0))
  with torch.inference_mode():
    logits = model(
        features.unsqueeze(0).to(device),
        embedding.unsqueeze(0).to(device))[0]
    probabilities = logits.softmax(dim=-1).cpu()
  labels, segments = merge_predictions(probabilities, frames, class_names)
  result = {
      'frame_count': len(frames),
      'frame_shift_ms': extractor.config.output_shift_ms,
      'classes': {str(index): name
                  for index, name in enumerate(class_names)},
      'segments': segments,
  }
  if include_frames:
    result['frame_predictions'] = [{
        'index': index,
        'class_id': labels[index],
        'class_name': class_names[labels[index]],
        'decision_start_ms': round(
            frame.decision_start_sample * 1000.0
            / extractor.config.sample_rate, 3),
        'decision_end_ms': round(
            frame.decision_end_sample * 1000.0
            / extractor.config.sample_rate, 3),
        'probabilities': [round(float(value), 7)
                          for value in probabilities[index]],
    } for index, frame in enumerate(frames)]
  return result


def main():
  args = parse_args()
  device = torch.device(args.device)
  model, package = load_checkpoint(args.checkpoint, device)
  embedding = aggregate_enrollment_embeddings(args.embedding)
  if args.features:
    features = load_features(args.features)
    input_kind = 'precomputed_features'
  else:
    waveform = load_audio(args.audio)
    features = PvadFeatureExtractor().extract(waveform)
    input_kind = 'audio'

  class_names = list(package['class_names'])
  result = run_inference(
      model, features, embedding, class_names, device,
      include_frames=args.include_frames)
  result.update({
      'checkpoint_sha256': CHECKPOINT_SHA256,
      'checkpoint_epoch': int(package['epoch']),
      'selection_metric': package['selection_metric'],
      'selection_score': float(package['selection_score']),
      'input_kind': input_kind,
      'speaker_embedding_backend': 'CAM++',
      'speaker_embedding_dimension': int(
          package['model_config']['speaker_embedding_dim']),
  })
  rendered = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
  if args.output:
    Path(args.output).write_text(rendered, encoding='utf-8')
  else:
    print(rendered, end='')


if __name__ == '__main__':
  main()
