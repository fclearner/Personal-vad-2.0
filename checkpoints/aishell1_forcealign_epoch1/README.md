# AISHELL1 Force-Align Baseline

This directory contains the first full AISHELL1 baseline checkpoint for the
Personal VAD 2.0 implementation.

## Artifact

- `best.pt`
  - Size: 4,935,335 bytes
  - SHA256: `8d099ac94f5e4ea606a56e07799ba9243f6cb935aae44375b4badefbbb2bd381`

## Training Data

- Source split: AISHELL1 train
- Records: 120,098
- Frames: 54,068,199
- Labels:
  - Class 0: speech, 39,853,387 frames
  - Class 2: non-speech, 14,214,812 frames
- Feature format used during training: compressed `npz` with float16 512-dim
  stacked log-mel features
- Speaker embedding: not used
- Non-target speech class: not constructed in this run

## Train/Validation Split

- Train records: 118,098
- Validation records: 2,000

Validation after one epoch:

```text
train_loss=0.0755
train_acc=0.9725
valid_loss=0.0716
valid_acc=0.9743
```

## AISHELL1 Test Evaluation

- Source split: AISHELL1 test
- Records: 7,176
- Frames: 3,596,689
- Labels:
  - Class 0: speech, 2,572,920 frames
  - Class 2: non-speech, 1,023,769 frames

Test metrics for `best.pt`:

```text
loss=0.0900
strict_3class_accuracy=0.9670
binary_vad_accuracy=0.9670
speech_f1=0.9773
non_speech_f1=0.9395
```

The strict 3-class and binary VAD accuracies are identical in this run because
the generated labels only contain class 0 and class 2, and the model did not
predict class 1 on the test set.

Full machine-readable reports are stored next to the checkpoint:

- `train_metrics.jsonl`
- `train_generation_report.json`
- `split_summary.json`
- `test_generation_report.json`
- `test_eval_best.json`
