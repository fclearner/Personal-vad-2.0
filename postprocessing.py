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


@dataclass(frozen=True)
class PvadPostprocessOutput:
  decisions: tuple[PvadFrameDecision, ...]
  transitions: tuple[PvadStateTransition, ...]


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
