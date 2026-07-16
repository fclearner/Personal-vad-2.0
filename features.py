"""Deterministic streaming frontend for Personal VAD 2.0."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class PvadFeatureConfig:
  """Published feature dimensions plus explicit implementation choices."""

  sample_rate: int = 16000
  num_mel_bins: int = 128
  frame_length_ms: float = 32.0
  frame_shift_ms: float = 10.0
  stack_frames: int = 4
  subsample_factor: int = 3
  fft_size: int = 1024
  f_min: float = 0.0
  f_max: float | None = None
  log_floor: float = 1e-10
  mel_scale: str = 'htk'

  def __post_init__(self):
    positive = (self.sample_rate, self.num_mel_bins, self.frame_length_ms,
                self.frame_shift_ms, self.stack_frames,
                self.subsample_factor, self.fft_size, self.log_floor)
    if any(value <= 0 for value in positive):
      raise ValueError('Feature configuration values must be positive.')
    if self.fft_size < self.frame_length_samples:
      raise ValueError('fft_size cannot be shorter than the analysis window.')
    if self.fft_size & (self.fft_size - 1):
      raise ValueError('fft_size must be a power of two.')
    if self.mel_scale not in {'htk', 'slaney'}:
      raise ValueError("mel_scale must be 'htk' or 'slaney'.")

  @property
  def frame_length_samples(self) -> int:
    return round(self.sample_rate * self.frame_length_ms / 1000.0)

  @property
  def frame_shift_samples(self) -> int:
    return round(self.sample_rate * self.frame_shift_ms / 1000.0)

  @property
  def n_fft(self) -> int:
    return self.fft_size

  @property
  def output_dim(self) -> int:
    return self.num_mel_bins * self.stack_frames

  @property
  def output_shift_ms(self) -> float:
    return self.frame_shift_ms * self.subsample_factor

  @property
  def receptive_field_ms(self) -> float:
    return (self.frame_length_ms
            + self.frame_shift_ms * (self.stack_frames - 1))

  def to_dict(self):
    return asdict(self)


@dataclass(frozen=True)
class FeatureFrame:
  index: int
  stack_start_sample: int
  decision_start_sample: int
  decision_end_sample: int

  @property
  def available_at_sample(self) -> int:
    return self.decision_end_sample


def _as_mono_float(waveform: torch.Tensor) -> torch.Tensor:
  waveform = torch.as_tensor(waveform)
  if waveform.dim() == 2:
    waveform = waveform.float().mean(dim=0)
  elif waveform.dim() != 1:
    raise ValueError('waveform must be one-dimensional or channel-first.')
  if not waveform.is_floating_point():
    info = torch.iinfo(waveform.dtype)
    waveform = waveform.float() / float(max(abs(info.min), info.max))
  return waveform.float().contiguous()


def load_audio(path: str | Path, sample_rate: int = 16000) -> torch.Tensor:
  import soundfile as sf
  import torchaudio

  array, source_rate = sf.read(
      str(path), dtype='float32', always_2d=True)
  waveform = _as_mono_float(torch.from_numpy(array).transpose(0, 1))
  if source_rate != sample_rate:
    waveform = torchaudio.functional.resample(
        waveform, source_rate, sample_rate)
  return waveform


class PvadFeatureExtractor:
  """128-bin log-Mel, causal stack-four, factor-three frontend.

  The paper publishes the dimensions and timing but not CMVN, window, Mel scale,
  pre-emphasis, or dither. CMVN, pre-emphasis, and dither are therefore absent;
  all remaining implementation choices are serialized in PvadFeatureConfig.
  """

  def __init__(self, config: PvadFeatureConfig | None = None,
               device: torch.device | str = 'cpu'):
    self.config = config or PvadFeatureConfig()
    self.device = torch.device(device)
    self.window = torch.hann_window(
        self.config.frame_length_samples, periodic=True, device=self.device)
    self.mel_filter = self._build_mel_filter()
    self.reset()

  def _build_mel_filter(self):
    from torchaudio.functional import melscale_fbanks

    f_max = self.config.f_max or self.config.sample_rate / 2.0
    return melscale_fbanks(
        n_freqs=self.config.n_fft // 2 + 1,
        f_min=self.config.f_min,
        f_max=f_max,
        n_mels=self.config.num_mel_bins,
        sample_rate=self.config.sample_rate,
        norm=None,
        mel_scale=self.config.mel_scale).to(self.device)

  def reset(self):
    self.sample_buffer = torch.empty(0, device=self.device)
    self.base_history: list[torch.Tensor] = []
    self.base_frames_seen = 0

  def _log_mel(self, frames):
    spectrum = torch.fft.rfft(
        frames * self.window, n=self.config.n_fft, dim=-1)
    mel_energy = spectrum.abs().square() @ self.mel_filter
    return mel_energy.clamp_min(self.config.log_floor).log()

  def feed(self, waveform_chunk: torch.Tensor) -> torch.Tensor:
    chunk = _as_mono_float(waveform_chunk).to(self.device)
    if chunk.numel():
      self.sample_buffer = torch.cat((self.sample_buffer, chunk))
    frame_length = self.config.frame_length_samples
    frame_shift = self.config.frame_shift_samples
    if self.sample_buffer.numel() < frame_length:
      return torch.empty((0, self.config.output_dim), device=self.device)

    count = 1 + (self.sample_buffer.numel() - frame_length) // frame_shift
    frames = self.sample_buffer.unfold(0, frame_length, frame_shift)[:count]
    base_features = self._log_mel(frames)
    self.sample_buffer = self.sample_buffer[count * frame_shift:]

    outputs = []
    first_anchor = self.config.stack_frames - 1
    for feature in base_features:
      base_index = self.base_frames_seen
      self.base_history.append(feature)
      self.base_history = self.base_history[-self.config.stack_frames:]
      if (len(self.base_history) == self.config.stack_frames
          and (base_index - first_anchor) % self.config.subsample_factor == 0):
        outputs.append(torch.cat(self.base_history))
      self.base_frames_seen += 1
    if outputs:
      return torch.stack(outputs)
    return torch.empty((0, self.config.output_dim), device=self.device)

  def extract(self, waveform: torch.Tensor) -> torch.Tensor:
    return PvadFeatureExtractor(self.config, self.device).feed(waveform)

  def frame_timing(self, count: int,
                   start_index: int = 0) -> list[FeatureFrame]:
    if count < 0 or start_index < 0:
      raise ValueError('count and start_index must be non-negative.')
    timings = []
    for index in range(start_index, start_index + count):
      stack_base = index * self.config.subsample_factor
      decision_base = stack_base + self.config.stack_frames - 1
      decision_start = decision_base * self.config.frame_shift_samples
      timings.append(FeatureFrame(
          index=index,
          stack_start_sample=stack_base * self.config.frame_shift_samples,
          decision_start_sample=decision_start,
          decision_end_sample=(decision_start
                               + self.config.frame_length_samples)))
    return timings


def _covered_samples(start: int, end: int,
                     intervals: Sequence[tuple[int, int]]) -> int:
  clipped = sorted((max(start, interval_start), min(end, interval_end))
                   for interval_start, interval_end in intervals
                   if interval_end > start and interval_start < end)
  covered = 0
  cursor = start
  for interval_start, interval_end in clipped:
    interval_start = max(interval_start, cursor)
    if interval_end > interval_start:
      covered += interval_end - interval_start
      cursor = interval_end
  return covered


def labels_from_intervals(
    frames: Iterable[FeatureFrame],
    target_intervals: Sequence[tuple[int, int]],
    non_target_intervals: Sequence[tuple[int, int]],
    min_active_fraction: float = 0.5,
) -> torch.Tensor:
  """Map speaker activity to 0/1/2; target wins on overlap."""

  if not 0.0 < min_active_fraction <= 1.0:
    raise ValueError('min_active_fraction must be in (0, 1].')
  labels = []
  for frame in frames:
    length = frame.decision_end_sample - frame.decision_start_sample
    required = length * min_active_fraction
    target = _covered_samples(
        frame.decision_start_sample, frame.decision_end_sample,
        target_intervals) >= required
    non_target = _covered_samples(
        frame.decision_start_sample, frame.decision_end_sample,
        non_target_intervals) >= required
    labels.append(0 if target else 1 if non_target else 2)
  return torch.tensor(labels, dtype=torch.long)
