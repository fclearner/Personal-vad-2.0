"""Configurable causal postprocessing for Personal VAD probabilities."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from features import FeatureFrame


CLASS_NAMES = ('target', 'non_target', 'non_speech')
TARGET_CLASS = 0
NON_TARGET_CLASS = 1
NON_SPEECH_CLASS = 2


def _validate_class_values(name, values, predicate, message):
  values = tuple(values)
  if len(values) != len(CLASS_NAMES):
    raise ValueError(f'{name} must contain one value per class.')
  if not all(predicate(value) for value in values):
    raise ValueError(f'{name} {message}')
  return values


@dataclass(frozen=True)
class PvadPostprocessorConfig:
  """Calibrated thresholds and temporal constraints for three PVAD classes.

  Thresholds and confirmation lengths are deliberately constructor arguments:
  production values must come from a versioned dev-set calibration artifact.
  """

  enter_thresholds: tuple[float, float, float]
  exit_thresholds: tuple[float, float, float]
  min_enter_frames: tuple[int, int, int]
  ema_alpha: float
  min_posterior_margin: float = 0.0
  initial_class: int = NON_SPEECH_CLASS
  fallback_class: int = NON_SPEECH_CLASS
  sample_rate: int = 16000

  def __post_init__(self):
    enter = _validate_class_values(
        'enter_thresholds', self.enter_thresholds,
        lambda value: 0.0 <= value <= 1.0, 'must be within [0, 1].')
    exit_ = _validate_class_values(
        'exit_thresholds', self.exit_thresholds,
        lambda value: 0.0 <= value <= 1.0, 'must be within [0, 1].')
    minimum = _validate_class_values(
        'min_enter_frames', self.min_enter_frames,
        lambda value: isinstance(value, int) and value > 0,
        'must contain positive integers.')
    if any(off > on for on, off in zip(enter, exit_)):
      raise ValueError(
          'Each exit threshold must be no greater than its enter threshold.')
    if not 0.0 < self.ema_alpha <= 1.0:
      raise ValueError('ema_alpha must be within (0, 1].')
    if not 0.0 <= self.min_posterior_margin <= 1.0:
      raise ValueError('min_posterior_margin must be within [0, 1].')
    if not 0 <= self.initial_class < len(CLASS_NAMES):
      raise ValueError('initial_class is outside the class range.')
    if not 0 <= self.fallback_class < len(CLASS_NAMES):
      raise ValueError('fallback_class is outside the class range.')
    if self.sample_rate <= 0:
      raise ValueError('sample_rate must be positive.')
    object.__setattr__(self, 'enter_thresholds', enter)
    object.__setattr__(self, 'exit_thresholds', exit_)
    object.__setattr__(self, 'min_enter_frames', minimum)

  def to_dict(self):
    return {
        'enter_thresholds': list(self.enter_thresholds),
        'exit_thresholds': list(self.exit_thresholds),
        'min_enter_frames': list(self.min_enter_frames),
        'ema_alpha': self.ema_alpha,
        'min_posterior_margin': self.min_posterior_margin,
        'initial_class': self.initial_class,
        'fallback_class': self.fallback_class,
        'sample_rate': self.sample_rate,
    }

  @classmethod
  def from_dict(cls, payload):
    return cls(
        enter_thresholds=tuple(payload['enter_thresholds']),
        exit_thresholds=tuple(payload['exit_thresholds']),
        min_enter_frames=tuple(payload['min_enter_frames']),
        ema_alpha=payload['ema_alpha'],
        min_posterior_margin=payload.get('min_posterior_margin', 0.0),
        initial_class=payload.get('initial_class', NON_SPEECH_CLASS),
        fallback_class=payload.get('fallback_class', NON_SPEECH_CLASS),
        sample_rate=payload.get('sample_rate', 16000))


@dataclass(frozen=True)
class PvadStateTransition:
  frame_index: int
  decision_time_ms: float
  previous_class: int
  current_class: int

  @property
  def target_activated(self):
    return self.current_class == TARGET_CLASS

  @property
  def target_released(self):
    return (self.previous_class == TARGET_CLASS
            and self.current_class != TARGET_CLASS)

  def to_dict(self):
    return {
        'frame_index': self.frame_index,
        'decision_time_ms': self.decision_time_ms,
        'previous_class': self.previous_class,
        'previous_label': CLASS_NAMES[self.previous_class],
        'current_class': self.current_class,
        'current_label': CLASS_NAMES[self.current_class],
        'target_activated': self.target_activated,
        'target_released': self.target_released,
    }


@dataclass(frozen=True)
class PvadFrameDecision:
  frame: FeatureFrame
  probabilities: tuple[float, float, float]
  smoothed_probabilities: tuple[float, float, float]
  raw_class: int
  state_class: int
  pending_class: int | None
  pending_frames: int
  transition: PvadStateTransition | None

  def to_dict(self):
    return {
        'frame_index': self.frame.index,
        'stack_start_sample': self.frame.stack_start_sample,
        'decision_start_sample': self.frame.decision_start_sample,
        'decision_end_sample': self.frame.decision_end_sample,
        'p_target': self.probabilities[TARGET_CLASS],
        'p_non_target': self.probabilities[NON_TARGET_CLASS],
        'p_non_speech': self.probabilities[NON_SPEECH_CLASS],
        'smoothed_p_target': self.smoothed_probabilities[TARGET_CLASS],
        'smoothed_p_non_target': (
            self.smoothed_probabilities[NON_TARGET_CLASS]),
        'smoothed_p_non_speech': (
            self.smoothed_probabilities[NON_SPEECH_CLASS]),
        'raw_class': self.raw_class,
        'raw_label': CLASS_NAMES[self.raw_class],
        'state_class': self.state_class,
        'state_label': CLASS_NAMES[self.state_class],
        'pending_class': self.pending_class,
        'pending_label': (
            None if self.pending_class is None
            else CLASS_NAMES[self.pending_class]),
        'pending_frames': self.pending_frames,
        'transition': (
            None if self.transition is None else self.transition.to_dict()),
    }


@dataclass(frozen=True)
class PvadPostprocessOutput:
  decisions: tuple[PvadFrameDecision, ...]
  transitions: tuple[PvadStateTransition, ...]


TARGET_SPEECH_IDLE = 'idle'
TARGET_SPEECH_STARTING = 'starting'
TARGET_SPEECH_ACTIVE = 'active'
TARGET_SPEECH_HANGOVER = 'hangover'


@dataclass(frozen=True)
class TargetSpeechStateMachineConfig:
  """Calibrated target-speaker speech onset and release policy."""

  activation_threshold: float
  release_threshold: float
  min_activation_frames: int
  min_release_frames: int
  sample_rate: int = 16000

  def __post_init__(self):
    if not 0.0 <= self.activation_threshold <= 1.0:
      raise ValueError('activation_threshold must be within [0, 1].')
    if not 0.0 <= self.release_threshold <= self.activation_threshold:
      raise ValueError(
          'release_threshold must be within [0, activation_threshold].')
    if self.min_activation_frames <= 0:
      raise ValueError('min_activation_frames must be positive.')
    if self.min_release_frames <= 0:
      raise ValueError('min_release_frames must be positive.')
    if self.sample_rate <= 0:
      raise ValueError('sample_rate must be positive.')

  def to_dict(self):
    return {
        'activation_threshold': self.activation_threshold,
        'release_threshold': self.release_threshold,
        'min_activation_frames': self.min_activation_frames,
        'min_release_frames': self.min_release_frames,
        'sample_rate': self.sample_rate,
    }

  @classmethod
  def from_dict(cls, payload):
    return cls(
        activation_threshold=payload['activation_threshold'],
        release_threshold=payload['release_threshold'],
        min_activation_frames=payload['min_activation_frames'],
        min_release_frames=payload['min_release_frames'],
        sample_rate=payload.get('sample_rate', 16000))


@dataclass(frozen=True)
class TargetSpeechTransition:
  frame_index: int
  decision_time_ms: float
  event: str
  segment_start_sample: int
  segment_end_sample: int | None

  @property
  def target_activated(self):
    return self.event == 'target_speech_start'

  @property
  def target_released(self):
    return self.event == 'target_speech_end'

  def to_dict(self):
    return {
        'frame_index': self.frame_index,
        'decision_time_ms': self.decision_time_ms,
        'event': self.event,
        'segment_start_sample': self.segment_start_sample,
        'segment_end_sample': self.segment_end_sample,
        'target_activated': self.target_activated,
        'target_released': self.target_released,
    }


@dataclass(frozen=True)
class TargetSpeechFrameDecision:
  frame: FeatureFrame
  p_target: float
  state: str
  target_active: bool
  identity_latched: bool
  activation_run_frames: int
  release_run_frames: int
  transition: TargetSpeechTransition | None

  def to_dict(self):
    return {
        'frame_index': self.frame.index,
        'decision_start_sample': self.frame.decision_start_sample,
        'decision_end_sample': self.frame.decision_end_sample,
        'p_target': self.p_target,
        'state': self.state,
        'target_active': self.target_active,
        'identity_latched': self.identity_latched,
        'activation_run_frames': self.activation_run_frames,
        'release_run_frames': self.release_run_frames,
        'transition': (
            None if self.transition is None else self.transition.to_dict()),
    }


@dataclass(frozen=True)
class TargetSpeechStateMachineOutput:
  decisions: tuple[TargetSpeechFrameDecision, ...]
  transitions: tuple[TargetSpeechTransition, ...]
  completed_segments: tuple[tuple[int, int], ...]


class TargetSpeechStateMachine:
  """FSMN-style causal onset, hangover and release for target speech.

  Identity remains latched for the whole stream epoch after a confirmed target
  onset.  Callers must explicitly reset at an independent buffer boundary.
  """

  def __init__(self, config: TargetSpeechStateMachineConfig):
    self.config = config
    self.reset()

  def reset(self):
    self.target_active = False
    self.identity_latched = False
    self.activation_run_frames = 0
    self.release_run_frames = 0
    self.candidate_start_sample = None
    self.segment_start_sample = None
    self.last_target_end_sample = None
    self.completed_segments = []
    self.next_frame_index = None

  @property
  def state(self):
    if self.target_active:
      return (
          TARGET_SPEECH_HANGOVER
          if self.release_run_frames else TARGET_SPEECH_ACTIVE)
    if self.activation_run_frames:
      return TARGET_SPEECH_STARTING
    return TARGET_SPEECH_IDLE

  def snapshot_segments(self, end_sample=None):
    segments = list(self.completed_segments)
    if self.target_active and self.segment_start_sample is not None:
      segment_end = (
          self.last_target_end_sample if end_sample is None else end_sample)
      if segment_end is not None:
        segment_end = max(self.segment_start_sample, int(segment_end))
        segments.append((self.segment_start_sample, segment_end))
    return tuple(segments)

  def _start_transition(self, frame):
    self.target_active = True
    self.identity_latched = True
    self.segment_start_sample = int(self.candidate_start_sample)
    self.last_target_end_sample = int(frame.decision_end_sample)
    self.activation_run_frames = 0
    self.candidate_start_sample = None
    return TargetSpeechTransition(
        frame_index=frame.index,
        decision_time_ms=(
            frame.decision_end_sample * 1000.0 / self.config.sample_rate),
        event='target_speech_start',
        segment_start_sample=self.segment_start_sample,
        segment_end_sample=None)

  def _end_transition(self, frame):
    segment_start = int(self.segment_start_sample)
    segment_end_sample = (
        self.last_target_end_sample
        if self.last_target_end_sample is not None
        else frame.decision_start_sample)
    segment_end = max(
        segment_start, int(segment_end_sample))
    self.completed_segments.append((segment_start, segment_end))
    self.target_active = False
    self.release_run_frames = 0
    self.segment_start_sample = None
    self.last_target_end_sample = None
    return TargetSpeechTransition(
        frame_index=frame.index,
        decision_time_ms=(
            frame.decision_end_sample * 1000.0 / self.config.sample_rate),
        event='target_speech_end',
        segment_start_sample=segment_start,
        segment_end_sample=segment_end)

  def process(self, probabilities, frames: Sequence[FeatureFrame]):
    if hasattr(probabilities, 'detach'):
      probabilities = probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    frames = tuple(frames)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
      raise ValueError('probabilities must have shape (frames, 3).')
    if probabilities.shape[0] != len(frames):
      raise ValueError('probabilities and frames must have equal lengths.')
    if (not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0) or np.any(probabilities > 1.0)):
      raise ValueError('probabilities must be finite and within [0, 1].')
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
      raise ValueError(
          'Each probability row must sum to 1 within tolerance.')

    decisions = []
    transitions = []
    completed_before = len(self.completed_segments)
    for row, frame in zip(probabilities, frames):
      if (self.next_frame_index is not None
          and frame.index != self.next_frame_index):
        raise ValueError(
            'Feature frame indices must be contiguous; call reset() for a '
            'new stream.')
      self.next_frame_index = frame.index + 1
      p_target = float(row[TARGET_CLASS])
      transition = None
      if not self.target_active:
        if p_target >= self.config.activation_threshold:
          if not self.activation_run_frames:
            self.candidate_start_sample = int(frame.decision_start_sample)
          self.activation_run_frames += 1
          if self.activation_run_frames >= self.config.min_activation_frames:
            transition = self._start_transition(frame)
        else:
          self.activation_run_frames = 0
          self.candidate_start_sample = None
      elif p_target >= self.config.release_threshold:
        self.release_run_frames = 0
        self.last_target_end_sample = int(frame.decision_end_sample)
      else:
        self.release_run_frames += 1
        if self.release_run_frames >= self.config.min_release_frames:
          transition = self._end_transition(frame)

      if transition is not None:
        transitions.append(transition)
      decisions.append(TargetSpeechFrameDecision(
          frame=frame,
          p_target=p_target,
          state=self.state,
          target_active=self.target_active,
          identity_latched=self.identity_latched,
          activation_run_frames=self.activation_run_frames,
          release_run_frames=self.release_run_frames,
          transition=transition))
    return TargetSpeechStateMachineOutput(
        decisions=tuple(decisions),
        transitions=tuple(transitions),
        completed_segments=tuple(
            self.completed_segments[completed_before:]))


class PvadPostprocessor:
  """Causal hysteresis/confirmation state machine over three-class posteriors."""

  def __init__(self, config: PvadPostprocessorConfig):
    self.config = config
    self.reset()

  def reset(self):
    self.state_class = self.config.initial_class
    self.pending_class = None
    self.pending_frames = 0
    self.smoothed_probabilities = None
    self.next_frame_index = None

  def _candidate_class(self, probabilities):
    current_probability = probabilities[self.state_class]
    eligible = [
        class_index for class_index in range(len(CLASS_NAMES))
        if class_index != self.state_class
        and probabilities[class_index]
        >= self.config.enter_thresholds[class_index]
        and probabilities[class_index] - current_probability
        >= self.config.min_posterior_margin]
    if eligible:
      return max(eligible, key=lambda index: probabilities[index])
    if current_probability >= self.config.exit_thresholds[self.state_class]:
      return self.state_class
    return self.config.fallback_class

  def _advance_state(self, candidate, frame):
    if candidate == self.state_class:
      self.pending_class = None
      self.pending_frames = 0
      return None
    if candidate == self.pending_class:
      self.pending_frames += 1
    else:
      self.pending_class = candidate
      self.pending_frames = 1
    if self.pending_frames < self.config.min_enter_frames[candidate]:
      return None

    previous = self.state_class
    self.state_class = candidate
    self.pending_class = None
    self.pending_frames = 0
    return PvadStateTransition(
        frame_index=frame.index,
        decision_time_ms=(
            frame.decision_end_sample * 1000.0 / self.config.sample_rate),
        previous_class=previous,
        current_class=candidate)

  def process(self, probabilities, frames: Sequence[FeatureFrame]):
    if hasattr(probabilities, 'detach'):
      probabilities = probabilities.detach().cpu().numpy()
    probabilities = np.asarray(probabilities, dtype=np.float64)
    frames = tuple(frames)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
      raise ValueError('probabilities must have shape (frames, 3).')
    if probabilities.shape[0] != len(frames):
      raise ValueError('probabilities and frames must have equal lengths.')
    if (not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0) or np.any(probabilities > 1.0)):
      raise ValueError('probabilities must be finite and within [0, 1].')
    if probabilities.size and not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-4):
      raise ValueError('Each probability row must sum to one.')

    decisions = []
    transitions = []
    for row, frame in zip(probabilities, frames):
      if self.next_frame_index is not None and frame.index != self.next_frame_index:
        raise ValueError(
            'Feature frame indices must be contiguous; call reset() for a '
            'new stream.')
      self.next_frame_index = frame.index + 1
      if self.smoothed_probabilities is None:
        self.smoothed_probabilities = row.copy()
      else:
        alpha = self.config.ema_alpha
        self.smoothed_probabilities = (
            alpha * row + (1.0 - alpha) * self.smoothed_probabilities)
      candidate = self._candidate_class(self.smoothed_probabilities)
      transition = self._advance_state(candidate, frame)
      if transition is not None:
        transitions.append(transition)
      decisions.append(PvadFrameDecision(
          frame=frame,
          probabilities=tuple(float(value) for value in row),
          smoothed_probabilities=tuple(
              float(value) for value in self.smoothed_probabilities),
          raw_class=int(np.argmax(row)),
          state_class=self.state_class,
          pending_class=self.pending_class,
          pending_frames=self.pending_frames,
          transition=transition))
    return PvadPostprocessOutput(
        decisions=tuple(decisions), transitions=tuple(transitions))
