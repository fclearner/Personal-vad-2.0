from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import PvadFeatureExtractor
from postprocessing import (
    NON_SPEECH_CLASS, NON_TARGET_CLASS, TARGET_CLASS, PvadPostprocessor,
    PvadPostprocessorConfig, TargetSpeechStateMachine,
    TargetSpeechStateMachineConfig)


def _config(**overrides):
  values = {
      'enter_thresholds': (0.6, 0.6, 0.6),
      'exit_thresholds': (0.4, 0.4, 0.4),
      'min_enter_frames': (2, 1, 1),
      'ema_alpha': 1.0,
  }
  values.update(overrides)
  return PvadPostprocessorConfig(**values)


def _frames(count, start=0):
  return PvadFeatureExtractor().frame_timing(count, start_index=start)


def test_target_confirmation_hysteresis_and_release():
  processor = PvadPostprocessor(_config())
  probabilities = np.array([
      [0.1, 0.1, 0.8],
      [0.7, 0.1, 0.2],
      [0.75, 0.1, 0.15],
      [0.55, 0.2, 0.25],
      [0.3, 0.1, 0.6],
  ])
  output = processor.process(probabilities, _frames(5))
  assert [item.state_class for item in output.decisions] == [2, 2, 0, 0, 2]
  assert [(item.previous_class, item.current_class)
          for item in output.transitions] == [(2, 0), (0, 2)]
  assert output.transitions[0].target_activated
  assert not output.transitions[0].target_released
  assert output.transitions[1].target_released
  assert output.transitions[0].decision_time_ms == 122.0
  record = output.decisions[2].to_dict()
  assert record['p_target'] == 0.75
  assert record['state_label'] == 'target'
  assert record['transition']['target_activated']
  assert record['decision_end_sample'] == 1952


def test_confirmation_state_survives_chunks_and_reset_is_explicit():
  processor = PvadPostprocessor(_config(min_enter_frames=(3, 1, 1)))
  first = processor.process(
      [[0.7, 0.1, 0.2], [0.8, 0.1, 0.1]], _frames(2))
  second = processor.process([[0.9, 0.05, 0.05]], _frames(1, start=2))
  assert not first.transitions
  assert second.transitions[0].target_activated
  try:
    processor.process([[0.1, 0.1, 0.8]], _frames(1, start=8))
  except ValueError as error:
    assert 'contiguous' in str(error)
  else:
    raise AssertionError('A discontinuous stream must be rejected.')
  processor.reset()
  output = processor.process([[0.1, 0.1, 0.8]], _frames(1, start=8))
  assert output.decisions[0].state_class == NON_SPEECH_CLASS


def test_ema_rejects_single_frame_target_spike():
  processor = PvadPostprocessor(_config(
      ema_alpha=0.5, min_enter_frames=(1, 1, 1)))
  output = processor.process(
      [[0.1, 0.1, 0.8], [0.9, 0.05, 0.05]], _frames(2))
  assert output.decisions[1].smoothed_probabilities[0] == 0.5
  assert output.decisions[1].state_class == NON_SPEECH_CLASS
  assert not output.transitions


def test_low_confidence_state_releases_to_configured_fallback():
  processor = PvadPostprocessor(_config(
      min_enter_frames=(1, 1, 2)))
  output = processor.process([
      [0.7, 0.1, 0.2],
      [0.35, 0.33, 0.32],
      [0.35, 0.33, 0.32],
  ], _frames(3))
  assert [item.state_class for item in output.decisions] == [0, 0, 2]
  assert output.transitions[-1].target_released


def test_stronger_non_target_candidate_wins():
  processor = PvadPostprocessor(_config(
      enter_thresholds=(0.35, 0.35, 0.6),
      exit_thresholds=(0.3, 0.3, 0.4),
      min_enter_frames=(1, 1, 1)))
  output = processor.process([[0.4, 0.5, 0.1]], _frames(1))
  assert output.decisions[0].raw_class == NON_TARGET_CLASS
  assert output.transitions[0].current_class == NON_TARGET_CLASS


def test_config_round_trip_and_input_validation():
  config = _config(min_posterior_margin=0.1)
  assert PvadPostprocessorConfig.from_dict(config.to_dict()) == config
  invalid_configs = [
      {'enter_thresholds': (0.5, 0.5),
       'exit_thresholds': (0.4, 0.4, 0.4),
       'min_enter_frames': (1, 1, 1), 'ema_alpha': 1.0},
      {'enter_thresholds': (0.5, 0.5, 0.5),
       'exit_thresholds': (0.6, 0.4, 0.4),
       'min_enter_frames': (1, 1, 1), 'ema_alpha': 1.0},
      {'enter_thresholds': (0.5, 0.5, 0.5),
       'exit_thresholds': (0.4, 0.4, 0.4),
       'min_enter_frames': (0, 1, 1), 'ema_alpha': 1.0},
  ]
  for values in invalid_configs:
    try:
      PvadPostprocessorConfig(**values)
    except ValueError:
      pass
    else:
      raise AssertionError('Invalid postprocessor config must be rejected.')

  processor = PvadPostprocessor(config)
  invalid_inputs = [
      ([[0.5, 0.5]], _frames(1)),
      ([[0.5, 0.5, 0.5]], _frames(1)),
      ([[np.nan, 0.5, 0.5]], _frames(1)),
      ([[0.1, 0.1, 0.8]], _frames(2)),
  ]
  for probabilities, frames in invalid_inputs:
    processor.reset()
    try:
      processor.process(probabilities, frames)
    except ValueError:
      pass
    else:
      raise AssertionError('Invalid postprocessor input must be rejected.')


def _target_speech_config(**overrides):
  values = {
      'activation_threshold': 0.55,
      'release_threshold': 0.25,
      'min_activation_frames': 2,
      'min_release_frames': 2,
  }
  values.update(overrides)
  return TargetSpeechStateMachineConfig(**values)


def test_target_speech_fsm_confirms_hangs_over_recovers_and_releases():
  processor = TargetSpeechStateMachine(_target_speech_config())
  frames = _frames(6)
  output = processor.process([
      [0.60, 0.20, 0.20],
      [0.70, 0.20, 0.10],
      [0.10, 0.80, 0.10],
      [0.30, 0.60, 0.10],
      [0.10, 0.20, 0.70],
      [0.05, 0.15, 0.80],
  ], frames)
  assert [item.state for item in output.decisions] == [
      'starting', 'active', 'hangover', 'active', 'hangover', 'idle']
  assert [item.transition.event for item in output.decisions
          if item.transition is not None] == [
              'target_speech_start', 'target_speech_end']
  assert output.decisions[1].identity_latched
  assert output.decisions[-1].identity_latched
  expected_segment = (
      frames[0].decision_start_sample,
      frames[3].decision_end_sample)
  assert output.completed_segments == (expected_segment,)
  assert processor.snapshot_segments() == (expected_segment,)


def test_target_speech_fsm_rejects_short_target_spike():
  processor = TargetSpeechStateMachine(_target_speech_config(
      min_activation_frames=3))
  output = processor.process([
      [0.90, 0.05, 0.05],
      [0.80, 0.10, 0.10],
      [0.10, 0.10, 0.80],
  ], _frames(3))
  assert not output.transitions
  assert not processor.target_active
  assert not processor.identity_latched
  assert processor.snapshot_segments() == ()


def test_target_speech_fsm_state_survives_chunks_and_reset_is_explicit():
  processor = TargetSpeechStateMachine(_target_speech_config())
  first = processor.process([[0.70, 0.20, 0.10]], _frames(1))
  second = processor.process(
      [[0.80, 0.10, 0.10]], _frames(1, start=1))
  assert first.decisions[-1].state == 'starting'
  assert second.transitions[0].target_activated
  assert processor.snapshot_segments(
      end_sample=_frames(1, start=2)[0].decision_end_sample)
  try:
    processor.process([[0.10, 0.10, 0.80]], _frames(1, start=8))
  except ValueError as error:
    assert 'contiguous' in str(error)
  else:
    raise AssertionError('A discontinuous stream must be rejected.')
  processor.reset()
  output = processor.process(
      [[0.10, 0.10, 0.80]], _frames(1, start=8))
  assert output.decisions[0].state == 'idle'
  assert not output.decisions[0].identity_latched


def test_target_speech_fsm_config_round_trip_and_input_validation():
  config = _target_speech_config()
  assert TargetSpeechStateMachineConfig.from_dict(
      config.to_dict()) == config
  invalid_configs = [
      {'activation_threshold': 1.1, 'release_threshold': 0.2,
       'min_activation_frames': 1, 'min_release_frames': 1},
      {'activation_threshold': 0.5, 'release_threshold': 0.6,
       'min_activation_frames': 1, 'min_release_frames': 1},
      {'activation_threshold': 0.5, 'release_threshold': 0.2,
       'min_activation_frames': 0, 'min_release_frames': 1},
  ]
  for values in invalid_configs:
    try:
      TargetSpeechStateMachineConfig(**values)
    except ValueError:
      pass
    else:
      raise AssertionError('Invalid target speech config must be rejected.')

  invalid_inputs = [
      ([[0.5, 0.5]], _frames(1)),
      ([[0.5, 0.5, 0.5]], _frames(1)),
      ([[np.nan, 0.5, 0.5]], _frames(1)),
      ([[0.1, 0.1, 0.8]], _frames(2)),
  ]
  for probabilities, frames in invalid_inputs:
    processor = TargetSpeechStateMachine(config)
    try:
      processor.process(probabilities, frames)
    except ValueError:
      pass
    else:
      raise AssertionError('Invalid target speech input must be rejected.')
