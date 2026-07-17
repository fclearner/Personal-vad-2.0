from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit_speaker_embeddings import (
    collect_speaker_audio, embedding_metrics, summarize_scores)


def test_collect_speaker_audio_is_split_scoped_and_disjoint():
  recipes = [
      {
          'split': 'dev', 'target_speaker': 'a',
          'enrollment_paths': ['a-e1.wav', 'a-e2.wav'],
          'components': [
              {'role': 'target', 'speaker_id': 'a',
               'source_path': 'a-c1.wav'},
              {'role': 'non_target', 'speaker_id': 'b',
               'source_path': 'b-c1.wav'},
              {'role': 'non_target', 'speaker_id': 'qwen3_omni_tts',
               'source_path': 'tts.wav'},
          ],
      },
      {
          'split': 'dev', 'target_speaker': 'b',
          'enrollment_paths': ['b-e1.wav', 'b-e2.wav'],
          'components': [
              {'role': 'target', 'speaker_id': 'b',
               'source_path': 'b-c2.wav'},
              {'role': 'non_target', 'speaker_id': 'a',
               'source_path': 'a-c2.wav'},
          ],
      },
      {
          'split': 'train', 'target_speaker': 'x',
          'enrollment_paths': ['x-e1.wav', 'x-e2.wav'],
          'components': [
              {'role': 'target', 'speaker_id': 'x',
               'source_path': 'x-c1.wav'},
          ],
      },
  ]
  selected = collect_speaker_audio(recipes, 'dev', 2)
  assert sorted(selected) == ['a', 'b']
  assert selected['a']['current'] == ('a-c1.wav', 'a-c2.wav')
  assert selected['b']['current'] == ('b-c1.wav', 'b-c2.wav')


def test_embedding_metrics_reports_separable_speakers():
  enrollment = {
      'a': [np.array([1.0, 0.0]), np.array([0.9, 0.1])],
      'b': [np.array([0.0, 1.0]), np.array([0.1, 0.9])],
  }
  current = {
      'a': [np.array([1.0, 0.0])],
      'b': [np.array([0.0, 1.0])],
  }
  result = embedding_metrics(enrollment, current)
  assert result['closed_set_top1_accuracy'] == 1.0
  assert result['same_at_or_below_impostor_p99_rate'] == 0.0
  assert result['same_aggregate_to_current_cosine']['median'] > 0.9
  assert result['impostor_aggregate_to_current_cosine']['median'] < 0.2
  assert summarize_scores([]) == {'count': 0}
