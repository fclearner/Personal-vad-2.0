import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_IGNORE_INDEX = -100


def _resolve_path(base_dir, value):
  path = Path(value)
  if path.is_absolute():
    return path
  return base_dir / path


def _pick_record_value(record, names, default=None):
  for name in names:
    if name in record and record[name] not in (None, ''):
      return record[name]
  return default


def _load_numpy(path, key=None):
  import numpy as np

  array = np.load(path)
  if isinstance(array, np.lib.npyio.NpzFile):
    selected_key = key or array.files[0]
    array = array[selected_key]
  return torch.as_tensor(array)


def load_tensor(path, key=None):
  path = Path(path)
  suffix = path.suffix.lower()
  if suffix in {'.pt', '.pth'}:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict):
      if key is not None:
        obj = obj[key]
      else:
        for candidate in ('features', 'feature', 'x', 'labels', 'label',
                          'y', 'embedding', 'speaker_embedding'):
          if candidate in obj:
            obj = obj[candidate]
            break
        else:
          raise KeyError(f'No tensor key found in {path}.')
    return torch.as_tensor(obj)
  if suffix in {'.npy', '.npz'}:
    return _load_numpy(path, key)
  raise ValueError(f'Unsupported tensor file type: {path}')


def read_manifest(path):
  path = Path(path)
  base_dir = path.parent
  if path.suffix.lower() == '.jsonl':
    records = []
    with path.open('r', encoding='utf-8') as handle:
      for line_no, line in enumerate(handle, 1):
        line = line.strip()
        if not line:
          continue
        record = json.loads(line)
        record['_base_dir'] = str(base_dir)
        record['_line_no'] = line_no
        records.append(record)
    return records

  if path.suffix.lower() == '.csv':
    with path.open('r', encoding='utf-8', newline='') as handle:
      records = []
      for line_no, record in enumerate(csv.DictReader(handle), 2):
        record['_base_dir'] = str(base_dir)
        record['_line_no'] = line_no
        records.append(record)
      return records

  raise ValueError('Manifest must be .jsonl or .csv.')


class PvadManifestDataset(Dataset):
  """Dataset backed by a JSONL or CSV manifest.

  Each record should point to frame features, frame labels, and optionally a
  speaker embedding:

  ``{"features": "utt001_feat.pt", "labels": "utt001_labels.pt",
  "embedding": "utt001_spk.pt"}``
  """

  def __init__(self, manifest_path, speaker_embedding_dim=64):
    self.records = read_manifest(manifest_path)
    self.speaker_embedding_dim = speaker_embedding_dim
    if not self.records:
      raise ValueError(f'No records found in {manifest_path}.')

  def __len__(self):
    return len(self.records)

  def __getitem__(self, index):
    record = self.records[index]
    base_dir = Path(record['_base_dir'])
    feature_path = _resolve_path(base_dir, _pick_record_value(
        record, ('features', 'feature', 'x', 'feat_path')))
    label_path = _resolve_path(base_dir, _pick_record_value(
        record, ('labels', 'label', 'y', 'label_path')))
    embedding_value = _pick_record_value(
        record, ('embedding', 'speaker_embedding', 'spk', 'embedding_path'))

    features = load_tensor(
        feature_path, _pick_record_value(record, ('feature_key', 'x_key')))
    labels = load_tensor(
        label_path, _pick_record_value(record, ('label_key', 'y_key')))
    features = features.float()
    labels = labels.long().view(-1)

    if features.dim() != 2:
      raise ValueError(f'features must have shape (T, D): {feature_path}')
    if labels.dim() != 1:
      raise ValueError(f'labels must have shape (T,): {label_path}')
    if features.size(0) != labels.size(0):
      raise ValueError(
          f'Feature/label length mismatch for manifest line '
          f'{record.get("_line_no")}: {features.size(0)} vs {labels.size(0)}')

    if embedding_value:
      embedding_path = _resolve_path(base_dir, embedding_value)
      embedding = load_tensor(embedding_path, _pick_record_value(
          record, ('embedding_key', 'speaker_embedding_key'))).float().view(-1)
    else:
      embedding = torch.zeros(self.speaker_embedding_dim, dtype=torch.float32)

    return {
        'id': _pick_record_value(record, ('id', 'utt_id', 'utterance_id'),
                                str(index)),
        'features': features,
        'labels': labels,
        'embedding': embedding,
        'length': features.size(0),
    }


class SyntheticPvadDataset(Dataset):
  """Small random dataset used only for smoke testing the training loop."""

  def __init__(self, size=8, min_frames=8, max_frames=16, input_dim=512,
               speaker_embedding_dim=64, num_classes=3, seed=0):
    self.size = size
    self.min_frames = min_frames
    self.max_frames = max_frames
    self.input_dim = input_dim
    self.speaker_embedding_dim = speaker_embedding_dim
    self.num_classes = num_classes
    self.seed = seed

  def __len__(self):
    return self.size

  def __getitem__(self, index):
    generator = torch.Generator().manual_seed(self.seed + index)
    frames = int(torch.randint(self.min_frames, self.max_frames + 1, (1,),
                               generator=generator))
    features = torch.randn(frames, self.input_dim, generator=generator)
    embedding = torch.randn(self.speaker_embedding_dim, generator=generator)
    labels = torch.randint(0, self.num_classes, (frames,),
                           generator=generator)
    return {
        'id': f'synthetic-{index}',
        'features': features,
        'labels': labels.long(),
        'embedding': embedding,
        'length': frames,
    }


def collate_pvad_batch(batch, label_pad=DEFAULT_IGNORE_INDEX):
  batch_size = len(batch)
  max_len = max(item['length'] for item in batch)
  feat_dim = batch[0]['features'].size(-1)
  emb_dim = batch[0]['embedding'].numel()

  features = batch[0]['features'].new_zeros(batch_size, max_len, feat_dim)
  labels = torch.full((batch_size, max_len), label_pad, dtype=torch.long)
  embeddings = batch[0]['embedding'].new_zeros(batch_size, emb_dim)
  lengths = torch.zeros(batch_size, dtype=torch.long)
  ids = []

  for row, item in enumerate(batch):
    length = item['length']
    features[row, :length] = item['features']
    labels[row, :length] = item['labels']
    embeddings[row] = item['embedding'].view(-1)
    lengths[row] = length
    ids.append(item['id'])

  return {
      'ids': ids,
      'features': features,
      'labels': labels,
      'embeddings': embeddings,
      'lengths': lengths,
  }
