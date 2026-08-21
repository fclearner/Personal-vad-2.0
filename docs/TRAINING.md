# Training and release guide

This guide describes the speaker-aware training path used for the published
epoch-38 pilot checkpoint. It covers source partitioning, deterministic
mixture generation, frozen CAM++ enrollment embeddings, training, evaluation,
dev-only FSM calibration, and inference-only export.

The published checkpoint is a control candidate, not a production-quality
model. It was never evaluated on the reserved test speakers and misses the
near-field non-target false-activation target. See the
[checkpoint model card](../checkpoints/pilot4k_speaker_aware_epoch38/README.md)
before using it in a control path.

## Reproducibility boundary

The repository contains all Personal VAD source code and the aggregate recipe
metadata, but it intentionally does not contain raw speech/noise/TTS audio,
speaker IDs, enrollment audio or embeddings, upstream model weights, or
manifests with absolute source paths. Consequently:

- the commands below reproduce the method with licensed local inputs;
- the published seed, counts, feature configuration, optimizer settings, and
  calibration grid reproduce the reference experiment design;
- byte-identical retraining of the published checkpoint is not possible from
  this repository alone because the exact source list and generated TTS WAVs
  are not published;
- CUDA kernels and dependency changes can also prevent bitwise-identical
  results even with identical data.

Keep generated manifests private. They contain local absolute paths and
speaker/source identifiers. Enrollment audio and embeddings are biometric
data and must not be committed.

## Reference experiment

The reference recipe used 200 speaker-disjoint AISHELL-1 speakers:

| Split | Speakers | Enrollment clips | Current utterances | Mixtures |
|---|---:|---:|---:|---:|
| train | 160 (80 F / 80 M) | 320 | 31,517 | 4,000 |
| dev | 20 (10 F / 10 M) | 40 | 3,883 | 200 |
| test | 20 (10 F / 10 M) | 40 | 3,915 | 0 |

Each mixture is 10 seconds. Two utterances per target speaker are reserved for
enrollment and never used as current audio. Train, dev, and test speakers do
not overlap. The test speakers are inventoried but are not mixed, embedded, or
used for model/FSM selection.

Frame labels are exclusive:

- `0`: target speaker speech; target wins during overlap;
- `1`: non-target speech, including synthetic TTS playback;
- `2`: no speech, including WHAM! background noise by itself.

ASR text, semantic-end labels, valid/invalid labels, and keywords never enter
the Personal VAD labels.

## 1. Create an environment

Python 3.10 or 3.11 is recommended. Install the PyTorch/torchaudio pair that
matches the host CUDA runtime first, then install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

The code and test suite were re-verified on Python 3.11 with PyTorch 2.x,
torchaudio 2.x, ModelScope 1.37, FunASR 1.3, NumPy 2.x, and SoundFile 0.14.
This is a compatibility reference, not a byte-exact lock for the July 2026
training run.

Run the small synthetic checks before preparing data:

```bash
python tests/smoke_test.py
python train.py \
  --smoke-test \
  --epochs 1 \
  --batch-size 2 \
  --speaker-embedding-dim 192 \
  --device cpu
```

## 2. Prepare licensed source assets

Provide these local assets and follow their upstream terms documented in
[`THIRD_PARTY.md`](../THIRD_PARTY.md):

- speaker-separated, mono 16 kHz AISHELL-1 WAVs under `Sxxxx/*.wav`;
- official AISHELL-1 `speaker.info` metadata;
- WHAM! noise with `tr/*.wav` and `cv/*.wav`;
- persisted Qwen3-Omni TTS WAVs in separate
  `train/response-*.wav` and `dev/response-*.wav` directories;
- a local ModelScope FSMN VAD model used only to obtain speech intervals;
- `iic/speech_campplus_sv_zh-cn_16k-common` or its local cache.

The optional `extract_aishell_webdataset.py` tool safely extracts supported
flat `BAC009SxxxxWxxxx.wav` gzip-tar shards. It is not required when normal
speaker directories already exist:

```bash
python extract_aishell_webdataset.py \
  --archive /data/shards/aishell-000.tar.gz \
  --archive /data/shards/aishell-001.tar.gz \
  --speaker-info /data/aishell/resource_aishell/speaker.info \
  --gender F \
  --gender M \
  --output-root /data/pvad/aishell_wav \
  --report /data/pvad/extract_report.json
```

## 3. Freeze the speaker split

Create a text file with one selected AISHELL speaker ID per line. To mirror the
reference counts, select exactly 100 female and 100 male speakers with at least
six utterances each. Then generate the stable split:

```bash
python prepare_speaker_manifest.py \
  --wav-root /data/pvad/aishell_wav \
  --speaker-info /data/aishell/resource_aishell/speaker.info \
  --speaker-list /data/pvad/manifests/balanced200.txt \
  --output-dir /data/pvad/manifests/balanced200_seed20260716 \
  --train-speakers 160 \
  --dev-speakers 20 \
  --test-speakers 20 \
  --enrollment-utterances 2 \
  --min-utterances 6 \
  --sample-rate 16000 \
  --seed 20260716
```

Inspect `summary.json`. All three speaker-overlap arrays must be empty. Do not
edit `speaker_split.json` after mixture generation begins; it is the stable
identity partition for all later steps.

## 4. Generate deterministic mixtures and features

The generator creates the WAV, 512-dimensional acoustic features, frame
labels, frozen 192-dimensional CAM++ embedding, review plot, recipe, and
training manifest for every sample:

```bash
python generate_review_set.py \
  --aishell-wav-root /data/pvad/aishell_wav \
  --source-manifest /data/pvad/manifests/balanced200_seed20260716 \
  --wham-root /data/wham_noise \
  --tts-root /data/pvad/qwen3_omni_tts \
  --vad-model /models/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  --campplus-model /models/speech_campplus_sv_zh-cn_16k-common \
  --output-dir /data/pvad/pilot4k/v1_split_stable_identity_seed20260716 \
  --seed 20260716 \
  --train-samples 4000 \
  --dev-samples 200 \
  --preprocess-device cpu
```

CPU preprocessing was used for the reference set. Each sample receives a
stable seed based only on the global seed, split, and sample index. Regenerating
one sample therefore does not renumber the random streams of later samples.

The scenarios cover target-only, near/far non-target, target/non-target
overlap, WHAM! noise, high-noise target speech, TTS echo, and target/TTS
overlap. Far-field speech uses a recorded deterministic parametric RIR recipe.
`train/` and `dev/` use disjoint WHAM! and TTS sources.

The frontend is fixed across generation, training, evaluation, and inference:

- 16 kHz mono input;
- 128 log-Mel bins, 32 ms periodic-Hann window, 10 ms shift;
- 1024-point FFT, HTK Mel scale, no CMVN/pre-emphasis/dither;
- current plus three past frames stacked to 512 dimensions;
- factor-three temporal sampling, producing one decision every 30 ms.

Review the generated plots and `summary.json` before training. The reference
pilot counts are recorded in `metadata/DATASET_SUMMARY.json`.

## 5. Audit enrollment separation

Re-extract CAM++ embeddings from enrollment and current speech to check
same-speaker stability, impostor separation, and closed-set retrieval:

```bash
python audit_speaker_embeddings.py \
  --recipes /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/recipes.jsonl \
  --campplus-model /models/speech_campplus_sv_zh-cn_16k-common \
  --split dev \
  --current-per-speaker 2 \
  --device cpu \
  --output /data/pvad/audits/pilot4k_dev_embeddings.json
```

Any enrollment/current path overlap or missing current speaker blocks the run.
The generator L2-normalizes each of the two enrollment embeddings, averages
them, and L2-normalizes the mean.

## 6. Run bounded training checks

First check the actual manifests for 20 steps without starting a long run:

```bash
python train.py \
  --train-manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/train.jsonl \
  --valid-manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/dev.jsonl \
  --output-dir /data/pvad/runs/pilot4k_batch_debug \
  --speaker-embedding-dim 192 \
  --batch-size 16 \
  --epochs 1 \
  --max-steps-per-epoch 20 \
  --enrollment-drop-prob 0.1 \
  --selection-metric target_f1 \
  --device cuda \
  --amp
```

Repeat with candidate batch sizes and select a stable value from recorded
examples/s, frames/s, peak allocated/reserved memory, NaN/OOM behavior, and
dataloader stability. A small fixed-data overfit run should also show a clear
loss decrease before the long pilot run.

## 7. Train the reference configuration

The published checkpoint used this model and optimizer configuration:

```bash
python train.py \
  --train-manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/train.jsonl \
  --valid-manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/dev.jsonl \
  --output-dir /data/pvad/runs/pilot4k_reference \
  --device cuda \
  --epochs 40 \
  --batch-size 64 \
  --num-workers 0 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --grad-clip 5 \
  --seed 20260717 \
  --amp \
  --save-every-epoch \
  --selection-metric target_f1 \
  --input-dim 512 \
  --encoder-dim 64 \
  --speaker-embedding-dim 192 \
  --num-classes 3 \
  --num-encoder-layers 4 \
  --num-speaker-layers 2 \
  --num-attention-heads 8 \
  --linear-units 64 \
  --dropout-rate 0.1 \
  --attention-dropout-rate 0.0 \
  --conv-kernel-size 7 \
  --left-context 31 \
  --subsampling linear \
  --enrollment-drop-prob 0.1
```

Enrollment dropout applies only during training. For a selected sample it
zeros the embedding and rewrites class `1` frames to class `0`, teaching the
enrollment-free generic-speech mode. It does not change validation labels.

`last.pt` is the latest epoch. `best.pt` is selected by validation target-class
F1. With `--save-every-epoch`, `epoch-NNNN.pt` files are also kept. The
reference `best.pt` was selected at epoch 38 with target F1 `0.863301`.

Resume with the same model/data arguments plus `--resume /path/to/last.pt`.
`--epochs` remains the final epoch number, not an additional epoch count.

## 8. Evaluate frame and event metrics

Evaluate the frozen checkpoint on the recipe-backed dev manifest:

```bash
python evaluate.py \
  --checkpoint /data/pvad/runs/pilot4k_reference/best.pt \
  --manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/dev.jsonl \
  --output /data/pvad/runs/pilot4k_reference/dev_evaluation.json \
  --device cuda \
  --target-threshold 0.5 \
  --min-active-frames 1 \
  --include-curves
```

Report all three per-class precision/recall/F1 values, the confusion matrix,
macro F1, target-vs-rest PR/ROC, event metrics, near/far/overlap/noise/TTS
slices, and top error samples. Do not select a model from an aggregate score
while hiding a failing safety slice.

## 9. Calibrate the target-speech FSM on dev only

The deployed-style policy uses a strong onset threshold, a lower continuation
threshold, and release hangover. The reference 280-candidate grid was:

```bash
python calibrate_target_fsm.py \
  --checkpoint /data/pvad/runs/pilot4k_reference/best.pt \
  --manifest /data/pvad/pilot4k/v1_split_stable_identity_seed20260716/dev.jsonl \
  --output /data/pvad/runs/pilot4k_reference/target_fsm_dev.json \
  --device cuda \
  --activation-threshold 0.55 \
  --activation-continue-thresholds 0.25 0.30 0.35 0.40 0.45 0.50 0.55 \
  --min-activation-frames 3 \
  --release-thresholds 0.10 0.15 0.20 0.25 0.30 0.40 0.50 \
  --min-release-frames 3 5 8 12 16 24 32 \
  --max-release-p95-ms 900 \
  --target-frame-recall-tolerance 0.001
```

The selected configuration is published in
`checkpoints/pilot4k_speaker_aware_epoch38/target_fsm.json`:

- activation / continuation / release: `0.55 / 0.35 / 0.20`;
- minimum activation / release: `3 / 12` frames (`90 / 360` ms);
- target event recall: `1.0` on dev;
- detection P95: `393` ms; release P95: `825` ms.

Calibration consumes dev only. Freeze the model and FSM before evaluating a
held-out test set. The published reference did not perform that final test and
must not be represented as test-validated.

## 10. Export a public inference checkpoint

Training checkpoints contain optimizer state, metrics, training arguments, and
local paths. Never publish `best.pt` directly. Export the whitelisted package:

```bash
python export_inference_checkpoint.py \
  --checkpoint /data/pvad/runs/pilot4k_reference/best.pt \
  --output /data/pvad/releases/best_inference.pt
```

The export contains only `state_dict`, model configuration, epoch, selection
metric/score, class names, the source checkpoint SHA256, and an export format
version. Inspect the printed hashes and update the model card/transfer manifest
for a release. PyTorch serialization bytes can vary by version; validate field
values and inference output in addition to the file checksum.

## 11. Verify release inference and streaming

Export two or more enrollment clips, then run the bundled checkpoint with the
same target FSM used by the duplex Personal VAD control candidate:

```bash
python -m speaker_backends.modelscope_export \
  enrollment_1.wav enrollment_2.wav \
  --output-dir data/enrollment \
  --aggregate-output target.spk.npy

python examples/pilot4k_speaker_aware_inference.py \
  --audio current.wav \
  --embedding data/enrollment/enrollment_1.spk.npy \
  --embedding data/enrollment/enrollment_2.spk.npy \
  --include-frames \
  --output pvad_result.json
```

The output includes raw three-class segments and calibrated target-speech FSM
segments/transitions. For live audio, use `PvadStreamingAdapter` from
`streaming.py`, feed its probabilities and frames to
`TargetSpeechStateMachine`, and call both objects' `reset()` at every
independent utterance or session boundary. Never reuse attention/convolution
caches or identity latch state across users.

Run the complete regression suite before publishing:

```bash
pytest -q
sha256sum checkpoints/pilot4k_speaker_aware_epoch38/best_inference.pt
```

The expected bundled checkpoint SHA256 is
`142419ebd37fb0a160acc3c75c0571cf8d0c420f2535fb450a5cbb3f7add3753`.
