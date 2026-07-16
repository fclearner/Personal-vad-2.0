import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import PvadManifestDataset
from evaluate import (contiguous_label_events, evaluate, sample_tags)


class FeatureLogitModel(torch.nn.Module):
  num_classes = 3

  def forward(self, inputs, embedding=None, input_lengths=None,
              return_lengths=False):
    logits = inputs[..., :3]
    return (logits, input_lengths) if return_lengths else logits


def test_sample_tags_are_nonexclusive_and_stable():
  record = {'scenario': 'overlap', 'tags': ['nearfield', 'overlap']}
  assert sample_tags(record) == ('overlap', 'nearfield')


def test_tts_events_do_not_relabel_target_overlap():
  labels = np.array([2, 1, 1, 0, 0, 2])
  events = contiguous_label_events(
      labels, ('tts_playback', 'overlap'), tts_playback=True, offset=10)
  assert [(event['start_frame'], event['end_frame'], event['event_type'])
          for event in events] == [
              (10, 11, 'non_speech'),
              (11, 13, 'tts_playback'),
              (13, 15, 'target'),
              (15, 16, 'non_speech'),
          ]


def _write_sample(tmp_path, sample_id, labels, scenario, tags):
  features = np.zeros((len(labels), 512), dtype=np.float32)
  features[np.arange(len(labels)), labels] = 10.0
  feature_path = tmp_path / f'{sample_id}_features.npy'
  label_path = tmp_path / f'{sample_id}_labels.npy'
  embedding_path = tmp_path / f'{sample_id}_embedding.npy'
  np.save(feature_path, features)
  np.save(label_path, np.asarray(labels, dtype=np.int64))
  np.save(embedding_path, np.ones(4, dtype=np.float32))
  return {
      'id': sample_id,
      'features': str(feature_path),
      'labels': str(label_path),
      'embedding': str(embedding_path),
      'recipe': str(tmp_path / 'recipes.jsonl'),
      'scenario': scenario,
      'tags': tags,
  }


def test_evaluate_reports_slices_events_and_errors(tmp_path):
  records = [
      _write_sample(
          tmp_path, 'near', [0, 0, 1, 2],
          'non_target_near', ['nearfield']),
      _write_sample(
          tmp_path, 'tts', [1, 1, 2, 2],
          'tts_echo', ['tts_playback']),
  ]
  manifest = tmp_path / 'dev.jsonl'
  with manifest.open('w', encoding='utf-8') as handle:
    for record in records:
      handle.write(json.dumps(record) + '\n')
  dataset = PvadManifestDataset(manifest, speaker_embedding_dim=4)
  feature_config = {'frame_shift_ms': 10.0, 'subsample_factor': 3}
  recipes = {
      record['id']: {'id': record['id'], 'feature_config': feature_config}
      for record in records}

  report = evaluate(
      FeatureLogitModel(), dataset, recipes, torch.device('cpu'),
      target_threshold=0.5, top_errors=2)
  assert report['classification']['overall']['accuracy'] == 1.0
  assert report['classification']['slices']['nearfield']['frames'] == 4
  assert report['classification']['slices']['tts_playback']['frames'] == 4
  assert report['events']['overall']['target_event_recall'] == 1.0
  assert report['events']['overall']['third_party_false_activation_rate'] == 0.0
  assert report['events']['overall']['false_interruption_rate'] == 0.0
  assert report['top_errors'][0]['frame_error_rate'] == 0.0
