"""Frame, slice, and control-event metrics for Personal VAD."""

from collections import defaultdict

import numpy as np


DEFAULT_CLASS_NAMES = ('target', 'non_target', 'non_speech')


def _ratio(numerator, denominator):
  return float(numerator / denominator) if denominator else None


class ClassificationAccumulator:
  """Bounded-memory confusion and approximate target PR/ROC accumulator."""

  def __init__(self, num_classes=3, target_class=0, num_score_bins=1000):
    if num_classes <= 1:
      raise ValueError('num_classes must be greater than one.')
    if not 0 <= target_class < num_classes:
      raise ValueError('target_class is outside the class range.')
    if num_score_bins < 2:
      raise ValueError('num_score_bins must be at least two.')
    self.num_classes = num_classes
    self.target_class = target_class
    self.num_score_bins = num_score_bins
    self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    self.positive_hist = np.zeros(num_score_bins, dtype=np.int64)
    self.negative_hist = np.zeros(num_score_bins, dtype=np.int64)

  def update(self, labels, predictions, target_scores):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    target_scores = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    if not (labels.size == predictions.size == target_scores.size):
      raise ValueError('labels, predictions, and target_scores must align.')

    valid = ((labels >= 0) & (labels < self.num_classes)
             & (predictions >= 0) & (predictions < self.num_classes)
             & np.isfinite(target_scores))
    labels = labels[valid].astype(np.int64)
    predictions = predictions[valid].astype(np.int64)
    scores = np.clip(target_scores[valid], 0.0, 1.0)
    if labels.size == 0:
      return

    flat_indices = labels * self.num_classes + predictions
    self.confusion += np.bincount(
        flat_indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes)
    bins = np.minimum(
        (scores * self.num_score_bins).astype(np.int64),
        self.num_score_bins - 1)
    positive = labels == self.target_class
    self.positive_hist += np.bincount(
        bins[positive], minlength=self.num_score_bins)
    self.negative_hist += np.bincount(
        bins[~positive], minlength=self.num_score_bins)

  def _target_curves(self, include_curves):
    true_positive = np.cumsum(self.positive_hist[::-1])
    false_positive = np.cumsum(self.negative_hist[::-1])
    positives = int(self.positive_hist.sum())
    negatives = int(self.negative_hist.sum())
    thresholds = np.arange(
        self.num_score_bins - 1, -1, -1, dtype=np.float64
    ) / self.num_score_bins

    recall = (true_positive / positives if positives
              else np.zeros_like(true_positive, dtype=np.float64))
    false_positive_rate = (
        false_positive / negatives if negatives
        else np.zeros_like(false_positive, dtype=np.float64))
    precision = np.divide(
        true_positive, true_positive + false_positive,
        out=np.ones_like(true_positive, dtype=np.float64),
        where=(true_positive + false_positive) > 0)

    roc_auc = None
    if positives and negatives:
      roc_auc = float(np.trapezoid(
          np.r_[0.0, recall], np.r_[0.0, false_positive_rate]))
    average_precision = None
    if positives:
      average_precision = float(np.sum(
          np.diff(np.r_[0.0, recall]) * precision))

    result = {
        'positive_support': positives,
        'negative_support': negatives,
        'average_precision': average_precision,
        'roc_auc': roc_auc,
    }
    if include_curves:
      result.update({
          'thresholds': thresholds.tolist(),
          'precision': precision.tolist(),
          'recall': recall.tolist(),
          'false_positive_rate': false_positive_rate.tolist(),
      })
    return result

  def compute(self, include_curves=False):
    per_class = {}
    f1_values = []
    names = (DEFAULT_CLASS_NAMES if self.num_classes == 3 else
             tuple(f'class_{index}' for index in range(self.num_classes)))
    for class_index, class_name in enumerate(names):
      true_positive = int(self.confusion[class_index, class_index])
      support = int(self.confusion[class_index].sum())
      predicted = int(self.confusion[:, class_index].sum())
      precision = _ratio(true_positive, predicted)
      recall = _ratio(true_positive, support)
      f1 = None
      if precision is not None and recall is not None:
        f1 = _ratio(2 * precision * recall, precision + recall)
      if f1 is not None:
        f1_values.append(f1)
      per_class[class_name] = {
          'precision': precision,
          'recall': recall,
          'f1': f1,
          'support': support,
      }

    total = int(self.confusion.sum())
    correct = int(np.trace(self.confusion))
    return {
        'accuracy': _ratio(correct, total),
        'frames': total,
        'confusion_matrix': self.confusion.tolist(),
        'per_class': per_class,
        'macro_f1': (float(np.mean(f1_values)) if f1_values else None),
        'target_vs_rest': self._target_curves(include_curves),
    }


class SlicedClassificationAccumulator:
  """Overall metrics plus arbitrary non-exclusive evaluation slices."""

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.overall = ClassificationAccumulator(**kwargs)
    self.slices = {}

  def update(self, labels, predictions, target_scores, slice_tags):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    target_scores = np.asarray(target_scores).reshape(-1)
    if len(slice_tags) != labels.size:
      raise ValueError('slice_tags must contain one entry per frame.')
    self.overall.update(labels, predictions, target_scores)
    grouped = defaultdict(list)
    for index, tags in enumerate(slice_tags):
      tags = (tags,) if isinstance(tags, str) else tags
      for tag in tags:
        grouped[str(tag)].append(index)
    for tag, indices in grouped.items():
      accumulator = self.slices.setdefault(
          tag, ClassificationAccumulator(**self.kwargs))
      accumulator.update(labels[indices], predictions[indices],
                         target_scores[indices])

  def compute(self, include_curves=False):
    return {
        'overall': self.overall.compute(include_curves),
        'slices': {
            name: accumulator.compute(include_curves)
            for name, accumulator in sorted(self.slices.items())
        },
    }


def _event_tags(event):
  tags = event.get('tags', ())
  return (tags,) if isinstance(tags, str) else tuple(tags)


def _contiguous_regions(mask):
  mask = np.asarray(mask, dtype=bool)
  if mask.size == 0:
    return 0
  padded = np.r_[False, mask, False].astype(np.int8)
  return int(np.count_nonzero(np.diff(padded) == 1))


def _compute_event_subset(target_scores, events, frame_shift_ms,
                          duration_seconds, target_threshold):
  active = np.asarray(target_scores) >= target_threshold
  target_events = 0
  target_detected = 0
  third_party_events = 0
  third_party_false = 0
  tts_events = 0
  tts_false = 0
  false_activations = 0
  latencies = []

  for event in events:
    start = max(0, int(event['start_frame']))
    end = min(active.size, int(event['end_frame']))
    if end <= start:
      continue
    event_active = active[start:end]
    event_type = event['event_type']
    if event_type == 'target':
      target_events += 1
      detected = np.flatnonzero(event_active)
      if detected.size:
        target_detected += 1
        latencies.append(float(detected[0] * frame_shift_ms))
    elif event_type == 'non_target':
      third_party_events += 1
      if event_active.any():
        third_party_false += 1
      false_activations += _contiguous_regions(event_active)
    elif event_type == 'tts_playback':
      tts_events += 1
      if event_active.any():
        tts_false += 1
      false_activations += _contiguous_regions(event_active)
    elif event_type == 'non_speech':
      false_activations += _contiguous_regions(event_active)
    else:
      raise ValueError(f'Unknown event_type: {event_type}')

  latency_p50 = float(np.percentile(latencies, 50)) if latencies else None
  latency_p95 = float(np.percentile(latencies, 95)) if latencies else None
  hours = duration_seconds / 3600.0
  return {
      'target_events': target_events,
      'target_event_recall': _ratio(target_detected, target_events),
      'third_party_events': third_party_events,
      'third_party_false_activation_rate': _ratio(
          third_party_false, third_party_events),
      'tts_playback_events': tts_events,
      'false_interruption_rate': _ratio(tts_false, tts_events),
      'detection_latency_p50_ms': latency_p50,
      'detection_latency_p95_ms': latency_p95,
      'false_activations': false_activations,
      'false_activations_per_hour': _ratio(false_activations, hours),
  }


def event_metrics(target_scores, events, frame_shift_ms, duration_seconds=None,
                  target_threshold=0.5):
  """Compute control metrics overall and for non-exclusive event tags."""

  target_scores = np.asarray(target_scores, dtype=np.float64).reshape(-1)
  if duration_seconds is None:
    duration_seconds = target_scores.size * frame_shift_ms / 1000.0
  overall = _compute_event_subset(
      target_scores, events, frame_shift_ms, duration_seconds,
      target_threshold)
  tags = sorted({tag for event in events for tag in _event_tags(event)})
  sliced = {}
  for tag in tags:
    selected = [event for event in events if tag in _event_tags(event)]
    slice_frames = sum(max(0, int(event['end_frame'])
                           - int(event['start_frame'])) for event in selected)
    slice_duration = slice_frames * frame_shift_ms / 1000.0
    sliced[tag] = _compute_event_subset(
        target_scores, selected, frame_shift_ms, slice_duration,
        target_threshold)
  return {'overall': overall, 'slices': sliced}
