"""Generate deterministic, human-reviewable Personal VAD mixtures."""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from features import (PvadFeatureConfig, PvadFeatureExtractor, load_audio,
                      labels_from_intervals)
from prepare_speaker_manifest import (
    discover_speaker_wavs, partition_speakers, select_speakers)
from speaker_backends.modelscope_export import (
    aggregate_embeddings, load_pipeline as load_speaker_pipeline)


SAMPLE_RATE = 16000
MIX_SECONDS = 10.0
MAX_SOURCE_SECONDS = 7.0
TRAIN_SCENARIOS = {
    'target_only': 15,
    'non_target_near': 15,
    'non_target_far': 10,
    'overlap': 15,
    'wham_noise': 10,
    'high_noise_target': 5,
    'tts_echo': 5,
    'target_tts_overlap': 5,
}
DEV_SCENARIOS = {
    'target_only': 4,
    'non_target_near': 4,
    'non_target_far': 2,
    'overlap': 4,
    'wham_noise': 2,
    'high_noise_target': 1,
    'tts_echo': 2,
    'target_tts_overlap': 1,
}


def scaled_scenario_counts(weights, sample_count):
  """Scale scenario weights to an exact sample count deterministically."""

  if sample_count <= 0:
    raise ValueError('sample_count must be positive.')
  total_weight = sum(weights.values())
  counts = {}
  remainders = []
  for order, (scenario, weight) in enumerate(weights.items()):
    count, remainder = divmod(sample_count * weight, total_weight)
    counts[scenario] = count
    remainders.append((remainder, -order, scenario))
  missing = sample_count - sum(counts.values())
  for _, _, scenario in sorted(remainders, reverse=True)[:missing]:
    counts[scenario] += 1
  return counts


def scenario_plan(seed, train_samples=80, dev_samples=20):
  rng = random.Random(seed)
  result = {}
  for split, counts in (
      ('train', scaled_scenario_counts(TRAIN_SCENARIOS, train_samples)),
      ('dev', scaled_scenario_counts(DEV_SCENARIOS, dev_samples))):
    scenarios = [
        scenario for scenario, count in counts.items() for _ in range(count)]
    rng.shuffle(scenarios)
    result[split] = scenarios
  return result


def speaker_pair_plan(speakers, sample_count, seed):
  """Build deterministic, diverse target/non-target pairs by full cycles."""

  speakers = sorted(speakers)
  if len(speakers) < 2:
    raise ValueError('At least two speakers are required for pairing.')
  if sample_count <= 0:
    raise ValueError('sample_count must be positive.')
  pairs = []
  cycle = 0
  while len(pairs) < sample_count:
    order = list(speakers)
    random.Random(seed + cycle * 2).shuffle(order)
    offset = 1 + random.Random(seed + cycle * 2 + 1).randrange(
        len(order) - 1)
    pairs.extend(
        (target, order[(index + offset) % len(order)])
        for index, target in enumerate(order))
    cycle += 1
  return pairs[:sample_count]


def split_speakers(speakers, train_count=8):
  speakers = sorted(speakers)
  if len(speakers) <= train_count:
    raise ValueError('Need at least one held-out dev speaker.')
  train = speakers[:train_count]
  dev = speakers[train_count:]
  if set(train) & set(dev):
    raise AssertionError('Speaker split leakage detected.')
  return train, dev


def load_source_partitions(source_manifest, speakers):
  """Load and validate exact train/dev/test speakers from a source manifest."""

  path = Path(source_manifest).resolve()
  if path.is_dir():
    path = path / 'speaker_split.json'
  payload = json.loads(path.read_text(encoding='utf-8'))
  partitions = payload.get('partitions')
  if set(partitions or {}) != {'train', 'dev', 'test'}:
    raise ValueError('Source manifest must contain train/dev/test partitions.')
  flattened = [
      speaker for split in ('train', 'dev', 'test')
      for speaker in partitions[split]]
  if len(flattened) != len(set(flattened)):
    raise ValueError('Source manifest contains speaker leakage.')
  selected = select_speakers(speakers, flattened)
  return selected, {
      split: list(partitions[split]) for split in ('train', 'dev', 'test')}


def discover_tts_wavs(tts_root):
  """Require disjoint persisted Qwen response WAVs for train and dev."""

  root = Path(tts_root).resolve()
  result = {
      split: sorted((root / split).glob('response-*.wav'))
      for split in ('train', 'dev')}
  for split, paths in result.items():
    if not paths:
      raise ValueError(f'TTS {split} response WAVs are required in {root / split}.')
  resolved = {
      split: {str(path.resolve()) for path in paths}
      for split, paths in result.items()}
  if resolved['train'] & resolved['dev']:
    raise ValueError('TTS train/dev audio leakage detected.')
  return result


def transform_intervals(intervals, crop_start, length, offset=0,
                        tail_samples=0):
  transformed = []
  crop_end = crop_start + length
  for start, end in intervals:
    start = max(start, crop_start)
    end = min(end, crop_end)
    if end > start:
      transformed.append((
          offset + start - crop_start,
          offset + end - crop_start + tail_samples))
  return transformed


def simulated_farfield_rir(sample_rate=SAMPLE_RATE):
  taps = ((0.0, 0.35), (0.035, 0.52), (0.082, 0.34), (0.145, 0.20))
  impulse = torch.zeros(round(0.18 * sample_rate))
  for delay, amplitude in taps:
    impulse[round(delay * sample_rate)] = amplitude
  return impulse / impulse.square().sum().sqrt()


def _active_rms(waveform, intervals):
  active = [
      waveform[max(0, start):min(waveform.numel(), end)]
      for start, end in intervals if end > start]
  active = [piece for piece in active if piece.numel()]
  values = torch.cat(active) if active else waveform
  return values.square().mean().clamp_min(1e-12).sqrt()


def _load_vad_pipeline(model, device='cpu'):
  from modelscope.pipelines import pipeline
  from modelscope.utils.constant import Tasks
  return pipeline(
      task=Tasks.voice_activity_detection, model=str(model), device=device)


def _vad_intervals(vad_pipeline, audio_path, cache):
  key = str(Path(audio_path).resolve())
  if key not in cache:
    result = vad_pipeline(key)
    if not result or 'value' not in result[0]:
      raise RuntimeError(f'No VAD result for {key}.')
    cache[key] = [
        [round(start_ms * SAMPLE_RATE / 1000),
         round(end_ms * SAMPLE_RATE / 1000)]
        for start_ms, end_ms in result[0]['value']]
  intervals = [tuple(item) for item in cache[key]]
  if not intervals:
    raise RuntimeError(f'No speech interval found in {key}.')
  return intervals


def _prepare_source(audio_path, intervals):
  waveform = load_audio(audio_path, SAMPLE_RATE)
  max_samples = round(MAX_SOURCE_SECONDS * SAMPLE_RATE)
  if waveform.numel() <= max_samples:
    crop_start = 0
    length = waveform.numel()
  else:
    first_speech = min(start for start, _ in intervals)
    crop_start = max(
        0, min(first_speech - round(0.25 * SAMPLE_RATE),
               waveform.numel() - max_samples))
    length = max_samples
  waveform = waveform[crop_start:crop_start + length]
  intervals = transform_intervals(intervals, crop_start, length)
  if not intervals:
    raise RuntimeError(f'Cropping removed all speech from {audio_path}.')
  return waveform, intervals, crop_start


def _place_speech(audio_path, intervals, offset_samples, level_dbfs,
                  farfield=False):
  waveform, intervals, crop_start = _prepare_source(audio_path, intervals)
  rir_tail = 0
  if farfield:
    rir = simulated_farfield_rir()
    waveform = torchaudio.functional.fftconvolve(
        waveform, rir, mode='full')
    rir_tail = rir.numel() - 1
  active_rms = _active_rms(waveform, intervals)
  desired = 10.0 ** (level_dbfs / 20.0)
  scale = desired / float(active_rms)
  waveform = waveform * scale

  mix_samples = round(MIX_SECONDS * SAMPLE_RATE)
  available = max(0, mix_samples - offset_samples)
  waveform = waveform[:available]
  placed = torch.zeros(mix_samples)
  placed[offset_samples:offset_samples + waveform.numel()] = waveform
  placed_intervals = transform_intervals(
      intervals, 0, waveform.numel(), offset_samples, rir_tail)
  placed_intervals = [
      (start, min(end, mix_samples)) for start, end in placed_intervals
      if start < mix_samples]
  metadata = {
      'source_path': str(Path(audio_path).resolve()),
      'source_crop_start_sample': crop_start,
      'offset_sample': offset_samples,
      'level_dbfs': level_dbfs,
      'linear_gain': scale,
      'farfield_rir': farfield,
      'rir_tail_samples': rir_tail,
      'speech_intervals_samples': [list(item) for item in placed_intervals],
  }
  return placed, placed_intervals, metadata


def _noise_segment(noise_path, sample_count, rng):
  noise = load_audio(noise_path, SAMPLE_RATE)
  if noise.numel() < sample_count:
    repeats = math.ceil(sample_count / max(noise.numel(), 1))
    noise = noise.repeat(repeats)
  start = rng.randrange(noise.numel() - sample_count + 1)
  return noise[start:start + sample_count], start


def _add_noise(mixture, noise_path, rng, snr_db=None, level_dbfs=-25.0,
               signal_intervals=()):
  noise, source_start = _noise_segment(
      noise_path, mixture.numel(), rng)
  noise_rms = noise.square().mean().clamp_min(1e-12).sqrt()
  if snr_db is None:
    desired = 10.0 ** (level_dbfs / 20.0)
  else:
    signal_rms = _active_rms(mixture, signal_intervals)
    desired = float(signal_rms) / (10.0 ** (snr_db / 20.0))
  scale = desired / float(noise_rms)
  return mixture + noise * scale, {
      'source_path': str(Path(noise_path).resolve()),
      'source_start_sample': source_start,
      'linear_gain': scale,
      'snr_db': snr_db,
      'level_dbfs': level_dbfs if snr_db is None else None,
  }


def _limit_peak(mixture):
  peak = float(mixture.abs().max())
  scale = min(1.0, 0.98 / peak) if peak else 1.0
  return mixture * scale, scale


def _write_pcm16_and_reload(path, waveform):
  """Persist the review audio and return the exact decoded training signal."""
  sf.write(str(path), waveform.numpy(), SAMPLE_RATE, subtype='PCM_16')
  return load_audio(path, SAMPLE_RATE)


def _plot_review(path, waveform, labels, scenario, speakers, frame_shift_ms):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap

  seconds = torch.arange(waveform.numel()).numpy() / SAMPLE_RATE
  fig, axes = plt.subplots(
      2, 1, figsize=(12, 3.8), sharex=True,
      gridspec_kw={'height_ratios': [3, 1]})
  axes[0].plot(seconds, waveform.numpy(), linewidth=0.35)
  axes[0].set_ylabel('amplitude')
  axes[0].set_title(
      f'{scenario} | target={speakers["target"]} '
      f'non_target={speakers["non_target"]}')
  axes[1].imshow(
      labels.numpy()[None, :], aspect='auto', interpolation='nearest',
      cmap=ListedColormap(['#2ca02c', '#d62728', '#bdbdbd']),
      vmin=0, vmax=2,
      extent=[0, labels.numel() * frame_shift_ms / 1000.0, 0, 1])
  axes[1].set_yticks([])
  axes[1].set_ylabel('0/1/2')
  axes[1].set_xlabel('seconds')
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  plt.close(fig)


def _speaker_embedding(sv_pipeline, enrollment_paths):
  embeddings = []
  for audio_path in enrollment_paths:
    result = sv_pipeline([str(audio_path)], output_emb=True)
    matrix = np.asarray(result['embs'], dtype=np.float32)
    if matrix.shape[0] != 1:
      raise RuntimeError(f'Unexpected CAM++ embedding shape: {matrix.shape}')
    embeddings.append(matrix[0])
  return aggregate_embeddings(embeddings)


def _scenario_tags(scenario):
  mapping = {
      'target_only': ['target_only'],
      'non_target_near': ['nearfield'],
      'non_target_far': ['farfield'],
      'overlap': ['overlap', 'nearfield'],
      'wham_noise': ['wham'],
      'high_noise_target': ['wham', 'high_noise'],
      'tts_echo': ['tts_playback'],
      'target_tts_overlap': ['tts_playback', 'overlap'],
  }
  return mapping[scenario]


class ReviewSetGenerator:
  def __init__(self, args):
    self.args = args
    self.rng = random.Random(args.seed)
    self.output_dir = Path(args.output_dir).resolve()
    self.audio_dir = self.output_dir / 'audio'
    self.feature_dir = self.output_dir / 'features'
    self.label_dir = self.output_dir / 'labels'
    self.embedding_dir = self.output_dir / 'embeddings'
    self.plot_dir = self.output_dir / 'plots'
    for directory in (
        self.audio_dir, self.feature_dir, self.label_dir,
        self.embedding_dir, self.plot_dir):
      directory.mkdir(parents=True, exist_ok=True)

    discovered_speakers = discover_speaker_wavs(
        args.aishell_wav_root, min_utterances=args.min_utterances)
    if args.source_manifest:
      self.speaker_wavs, partitions = load_source_partitions(
          args.source_manifest, discovered_speakers)
    else:
      self.speaker_wavs = discovered_speakers
      partitions = partition_speakers(
          self.speaker_wavs, args.train_speakers, args.dev_speakers,
          args.test_speakers, args.seed)
    self.train_speakers = partitions['train']
    self.dev_speakers = partitions['dev']
    self.test_speakers = partitions['test']
    self.enrollment = {
        speaker: wavs[:2] for speaker, wavs in self.speaker_wavs.items()}
    self.current = {
        speaker: wavs[2:] for speaker, wavs in self.speaker_wavs.items()}
    enrollment_paths = {
        str(path.resolve()) for paths in self.enrollment.values()
        for path in paths}
    current_paths = {
        str(path.resolve()) for paths in self.current.values()
        for path in paths}
    if enrollment_paths & current_paths:
      raise AssertionError('Enrollment/current audio leakage detected.')

    self.wham = {
        'train': sorted((Path(args.wham_root) / 'tr').glob('*.wav')),
        'dev': sorted((Path(args.wham_root) / 'cv').glob('*.wav')),
    }
    self.tts = discover_tts_wavs(args.tts_root)
    if not self.wham['train'] or not self.wham['dev']:
      raise ValueError('WHAM train/cv noise files are required.')

    self.vad_cache_path = self.output_dir / 'vad_cache.json'
    self.vad_cache = {}
    if self.vad_cache_path.exists():
      self.vad_cache = json.loads(
          self.vad_cache_path.read_text(encoding='utf-8'))
    self.vad_pipeline = _load_vad_pipeline(
        args.vad_model, args.preprocess_device)
    self.sv_pipeline = load_speaker_pipeline(
        args.campplus_model, args.preprocess_device)
    self.feature_extractor = PvadFeatureExtractor(PvadFeatureConfig())
    self.embedding_paths = self._export_embeddings()

  def _export_embeddings(self):
    paths = {}
    active_speakers = sorted(self.train_speakers + self.dev_speakers)
    for speaker in active_speakers:
      vector = _speaker_embedding(
          self.sv_pipeline, self.enrollment[speaker])
      output = self.embedding_dir / f'{speaker}.npy'
      np.save(output, vector)
      paths[speaker] = output
    return paths

  def _source(self, speaker, sample_index, offset=0):
    candidates = self.current[speaker]
    for attempt in range(len(candidates)):
      path = candidates[(sample_index + offset + attempt) % len(candidates)]
      try:
        intervals = _vad_intervals(
            self.vad_pipeline, path, self.vad_cache)
        return path, intervals
      except RuntimeError:
        continue
    raise RuntimeError(f'No VAD-positive current audio for {speaker}.')

  def _tts_source(self, split, sample_index):
    paths = self.tts[split]
    path = paths[sample_index % len(paths)]
    return path, _vad_intervals(self.vad_pipeline, path, self.vad_cache)

  def _build_one(self, split, scenario, sample_index, speaker_pair):
    target_speaker, non_target_speaker = speaker_pair
    target_path, target_vad = self._source(
        target_speaker, sample_index)
    non_target_path, non_target_vad = self._source(
        non_target_speaker, sample_index, offset=11)
    target_intervals = []
    non_target_intervals = []
    components = []
    mixture = torch.zeros(round(MIX_SECONDS * SAMPLE_RATE))

    def add_speech(role, path, source_intervals, offset_seconds,
                   level_dbfs, farfield=False, speaker_id=None):
      nonlocal mixture
      audio, intervals, metadata = _place_speech(
          path, source_intervals, round(offset_seconds * SAMPLE_RATE),
          level_dbfs, farfield)
      mixture = mixture + audio
      metadata.update({'role': role, 'speaker_id': speaker_id})
      components.append(metadata)
      if role == 'target':
        target_intervals.extend(intervals)
      else:
        non_target_intervals.extend(intervals)

    if scenario in {
        'target_only', 'overlap', 'high_noise_target',
        'target_tts_overlap'}:
      add_speech(
          'target', target_path, target_vad, 0.8, -20.0,
          speaker_id=target_speaker)
    if scenario == 'non_target_near':
      add_speech(
          'non_target', non_target_path, non_target_vad, 0.8, -16.0,
          speaker_id=non_target_speaker)
    elif scenario == 'non_target_far':
      add_speech(
          'non_target', non_target_path, non_target_vad, 1.0, -25.0,
          farfield=True, speaker_id=non_target_speaker)
    elif scenario == 'overlap':
      sir_db = self.rng.choice((-5.0, 0.0, 5.0))
      add_speech(
          'non_target', non_target_path, non_target_vad, 1.4,
          -20.0 - sir_db, speaker_id=non_target_speaker)
    elif scenario in {'tts_echo', 'target_tts_overlap'}:
      tts_path, tts_vad = self._tts_source(split, sample_index)
      add_speech(
          'non_target', tts_path, tts_vad,
          0.8 if scenario == 'tts_echo' else 1.5,
          -17.0, farfield=True, speaker_id='qwen3_omni_tts')

    noise_metadata = None
    if scenario in {'wham_noise', 'high_noise_target'}:
      noises = self.wham[split]
      noise_path = noises[self.rng.randrange(len(noises))]
      mixture, noise_metadata = _add_noise(
          mixture, noise_path, self.rng,
          snr_db=None if scenario == 'wham_noise'
          else self.rng.choice((-5.0, 0.0, 5.0)),
          signal_intervals=target_intervals + non_target_intervals)
      noise_metadata['role'] = 'background_noise'
      components.append(noise_metadata)

    mixture, limiter_scale = _limit_peak(mixture)
    sample_id = f'{split}-{sample_index:04d}-{scenario}'
    audio_path = self.audio_dir / f'{sample_id}.wav'
    feature_path = self.feature_dir / f'{sample_id}.npy'
    label_path = self.label_dir / f'{sample_id}.npy'
    plot_path = self.plot_dir / f'{sample_id}.png'
    stored_mixture = _write_pcm16_and_reload(audio_path, mixture)
    features = self.feature_extractor.extract(stored_mixture)
    timings = self.feature_extractor.frame_timing(features.size(0))
    labels = labels_from_intervals(
        timings, target_intervals, non_target_intervals)

    expected = {
        'target_only': 0,
        'non_target_near': 1,
        'non_target_far': 1,
        'overlap': 0,
        'wham_noise': 2,
        'high_noise_target': 0,
        'tts_echo': 1,
        'target_tts_overlap': 0,
    }[scenario]
    if not labels.eq(expected).any():
      raise RuntimeError(
          f'{scenario} sample has no expected class {expected}.')

    np.save(feature_path, features.numpy().astype(np.float32))
    np.save(label_path, labels.numpy().astype(np.int64))
    has_non_target = any(
        component.get('role') == 'non_target'
        for component in components)
    speaker_payload = {
        'target': target_speaker,
        'non_target': (
            'qwen3_omni_tts' if 'tts' in scenario
            else non_target_speaker if has_non_target else None),
    }
    _plot_review(
        plot_path, stored_mixture, labels, scenario, speaker_payload,
        self.feature_extractor.config.output_shift_ms)

    class_counts = {
        str(class_id): int(labels.eq(class_id).sum())
        for class_id in range(3)}
    recipe = {
        'id': sample_id,
        'seed': self.args.seed,
        'split': split,
        'scenario': scenario,
        'tags': _scenario_tags(scenario),
        'sample_rate': SAMPLE_RATE,
        'duration_samples': mixture.numel(),
        'feature_config': self.feature_extractor.config.to_dict(),
        'target_speaker': target_speaker,
        'non_target_speaker': speaker_payload['non_target'],
        'enrollment_paths': [
            str(path.resolve()) for path in self.enrollment[target_speaker]],
        'components': components,
        'target_intervals_samples': [
            list(item) for item in target_intervals],
        'non_target_intervals_samples': [
            list(item) for item in non_target_intervals],
        'limiter_scale': limiter_scale,
        'class_counts': class_counts,
        'artifacts': {
            'audio': str(audio_path),
            'features': str(feature_path),
            'labels': str(label_path),
            'embedding': str(self.embedding_paths[target_speaker]),
            'plot': str(plot_path),
        },
    }
    manifest = {
        'id': sample_id,
        'features': str(feature_path),
        'labels': str(label_path),
        'embedding': str(self.embedding_paths[target_speaker]),
        'recipe': str(self.output_dir / 'recipes.jsonl'),
        'split': split,
        'scenario': scenario,
        'tags': recipe['tags'],
        'target_speaker': target_speaker,
        'non_target_speaker': speaker_payload['non_target'],
    }
    return recipe, manifest

  def run(self):
    plans = scenario_plan(
        self.args.seed, self.args.train_samples, self.args.dev_samples)
    recipes = []
    manifests = {'train': [], 'dev': []}
    total_counts = Counter()
    scenario_counts = Counter()

    for split, speakers in (
        ('train', self.train_speakers), ('dev', self.dev_speakers)):
      pair_seed = self.args.seed + (0 if split == 'train' else 1_000_000)
      pairs = speaker_pair_plan(speakers, len(plans[split]), pair_seed)
      for index, (scenario, pair) in enumerate(zip(plans[split], pairs)):
        recipe, manifest = self._build_one(
            split, scenario, index, pair)
        recipes.append(recipe)
        manifests[split].append(manifest)
        total_counts.update({
            class_id: count
            for class_id, count in recipe['class_counts'].items()})
        scenario_counts[(split, scenario)] += 1
        print(
            recipe['id'], recipe['target_speaker'],
            recipe['non_target_speaker'], recipe['class_counts'])

    with (self.output_dir / 'recipes.jsonl').open(
        'w', encoding='utf-8') as handle:
      for recipe in recipes:
        handle.write(json.dumps(recipe, ensure_ascii=False) + '\n')
    for split, rows in manifests.items():
      with (self.output_dir / f'{split}.jsonl').open(
          'w', encoding='utf-8') as handle:
        for row in rows:
          handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    self.vad_cache_path.write_text(
        json.dumps(self.vad_cache, ensure_ascii=False, indent=2),
        encoding='utf-8')
    inventory = {
        'source_manifest': (
            str(Path(self.args.source_manifest).resolve())
            if self.args.source_manifest else None),
        'aishell_wav_roots': [
            str(Path(root).resolve()) for root in self.args.aishell_wav_root],
        'wham_root': str(Path(self.args.wham_root).resolve()),
        'tts_root': str(Path(self.args.tts_root).resolve()),
        'tts_sources': {
            split: [str(path.resolve()) for path in paths]
            for split, paths in self.tts.items()},
        'train_speakers': self.train_speakers,
        'dev_speakers': self.dev_speakers,
        'test_speakers': self.test_speakers,
        'speaker_overlap': {
            'train_dev': sorted(
                set(self.train_speakers) & set(self.dev_speakers)),
            'train_test': sorted(
                set(self.train_speakers) & set(self.test_speakers)),
            'dev_test': sorted(
                set(self.dev_speakers) & set(self.test_speakers)),
        },
        'enrollment': {
            speaker: [str(path.resolve()) for path in paths]
            for speaker, paths in self.enrollment.items()},
        'current_candidate_counts': {
            speaker: len(paths) for speaker, paths in self.current.items()},
        'test_policy': (
            'Held-out test speakers are inventoried but never mixed or embedded.'),
    }
    (self.output_dir / 'source_inventory.json').write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2),
        encoding='utf-8')
    summary = {
        'samples': len(recipes),
        'split_counts': {
            split: len(rows) for split, rows in manifests.items()},
        'scenario_counts': {
            f'{split}/{scenario}': count
            for (split, scenario), count in sorted(scenario_counts.items())},
        'class_frame_counts': dict(sorted(total_counts.items())),
        'speaker_overlap': inventory['speaker_overlap'],
        'heldout_test_speakers': len(self.test_speakers),
        'feature_dim': self.feature_extractor.config.output_dim,
        'embedding_dim': int(np.load(
            next(iter(self.embedding_paths.values()))).size),
    }
    (self.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
  parser = argparse.ArgumentParser(
      description='Generate deterministic Personal VAD mixtures.')
  parser.add_argument(
      '--aishell-wav-root', action='append', required=True,
      help='Speaker wav root; repeat for additional non-overlapping roots.')
  parser.add_argument('--wham-root', required=True)
  parser.add_argument('--tts-root', required=True)
  parser.add_argument('--vad-model', required=True)
  parser.add_argument('--campplus-model', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument(
      '--source-manifest',
      help='Directory or speaker_split.json with exact source partitions.')
  parser.add_argument('--seed', type=int, default=20260716)
  parser.add_argument('--train-speakers', type=int, default=8)
  parser.add_argument('--dev-speakers', type=int, default=4)
  parser.add_argument('--test-speakers', type=int, default=0)
  parser.add_argument('--train-samples', type=int, default=80)
  parser.add_argument('--dev-samples', type=int, default=20)
  parser.add_argument('--min-utterances', type=int, default=6)
  parser.add_argument(
      '--preprocess-device', choices=('cpu', 'cuda'), default='cpu')
  return parser.parse_args()


def main():
  ReviewSetGenerator(parse_args()).run()


if __name__ == '__main__':
  main()
