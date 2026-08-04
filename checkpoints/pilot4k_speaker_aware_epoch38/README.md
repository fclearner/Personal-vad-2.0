# Pilot4k speaker-aware checkpoint (epoch 38)

This directory contains a sanitized inference-only export of a trained,
speaker-conditioned Personal VAD model. It predicts three frame classes:

- `0`: target-speaker speech
- `1`: non-target-speaker speech
- `2`: non-speech

The checkpoint is a trained control candidate, not a production-accepted
release. In particular, the reserved test speakers were not evaluated and the
near-field non-target false-activation requirement was not met.

## Artifacts and provenance

| Artifact | Size | SHA256 |
|---|---:|---|
| `best_inference.pt` | 1,843,557 bytes | `142419ebd37fb0a160acc3c75c0571cf8d0c420f2535fb450a5cbb3f7add3753` |
| Original DGX `best.pt` | 5,087,103 bytes | `4ccafe36a985bbbc6d461c367cf8a259bac25341407b276eeba591f587d2e8d8` |

The original checkpoint is not published. The inference export contains only
`state_dict`, `model_config`, epoch/selection fields, class names, the source
checksum, and an export format version. It contains no optimizer, training
arguments, absolute paths, raw manifests, audio, or speaker/utterance IDs.
`SOURCE_CHECKPOINT.json` and the repository-level `TRANSFER_MANIFEST.json`
record the remaining provenance and file checksums.

The source checkpoint was selected at epoch 38 by dev target-class F1
(`0.8633011912`). Its strongest recorded pre-training code evidence is commit
`6ccf66ca38d88c977e11098b9ae48cda3148d8b0`; the checkpoint itself did not
embed a Git commit.

## Model configuration

- 512-dimensional acoustic input; 64-dimensional encoder
- four causal Conformer encoder blocks and two speaker pre-net blocks
- eight attention heads, 64 linear units, kernel size 7, left context 31
- 192-dimensional external speaker embedding projected into cosine-to-FiLM
- three output classes, linear/no additional subsampling
- 435,843 trainable Personal VAD parameters
- frozen, offline speaker embedding backend during data preparation/training

Training used AdamW, cross-entropy, CUDA AMP, batch size 64, learning rate
`1e-3`, weight decay `1e-4`, gradient clipping at 5, dropout `0.1`, enrollment
drop probability `0.1`, 40 planned epochs, and seed `20260717`.

## CAM++ speaker conditioning

CAM++ is open source. This work used the implementation in the
[ModelScope 3D-Speaker project](https://github.com/modelscope/3D-Speaker),
which is released under Apache-2.0, with the exact ModelScope model
[`iic/speech_campplus_sv_zh-cn_16k-common`](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common).
CAM++ produced frozen 192-dimensional enrollment embeddings offline; it was not
fine-tuned as part of Personal VAD.

This repository does **not** bundle or relicense CAM++ weights. Obtain the
upstream model from ModelScope/3D-Speaker and follow its license and model-card
terms. Each of two enrollment clips was L2-normalized, the two vectors were
averaged, and the mean was L2-normalized again.

## Data recipe

The speaker-disjoint recipe is
`pilot4k/v1_split_stable_identity_seed20260716`. It was built from these source
categories:

- AISHELL-1 `balanced200`: Mandarin speech and speaker identities
- WHAM `tr`/`cv`: background noise
- Qwen3-Omni TTS: synthetic playback/echo speech
- FSMN speech intervals: speech activity boundaries used for mixing labels
- CAM++ enrollment embeddings: frozen speaker identity conditioning

Only aggregate metadata is published. The exact counts are also machine
readable in `metadata/DATASET_SUMMARY.json`.

| Split | Speakers (F/M) | Current / enrollment utterances | Source hours | Generated recipes | Generated duration |
|---|---:|---:|---:|---:|---:|
| train | 160 (80/80) | 31,517 / 320 | 39.1962 h | 4,000 | 40,000 s |
| dev | 20 (10/10) | 3,883 / 40 | 4.8470 h | 200 | 2,000 s |
| test | 20 (10/10) | 3,915 / 40 | 5.1100 h | 0 | not generated |

The generated train set contains 1,328,000 decision frames with class counts
275,644 target / 249,826 non-target / 802,530 non-speech. The generated dev set
contains 66,400 frames with counts 14,120 / 12,369 / 39,911. The 20 test
speakers were reserved and never mixed or embedded into generated examples.

The first two utterances of each speaker were enrollment audio and were
excluded from current audio. Target/non-target pairs were created only within a
split by cyclic non-zero speaker offsets, so the two identities differ. A
32 ms decision window was labeled target when target coverage was at least
50%; otherwise non-target when non-target coverage was at least 50%; otherwise
non-speech. Target wins overlap. Pure WHAM is non-speech; TTS playback is
non-target. During training only, enrollment dropout zeroed the embedding and
rewrote non-target labels to target for the selected sample.

## Validation results and limits

At epoch 38, dev accuracy was `0.936943` and macro F1 was `0.900994`:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| target | 0.864282 | 0.862323 | 0.863301 |
| non-target | 0.852702 | 0.847118 | 0.849901 |
| non-speech | 0.988382 | 0.991180 | 0.989779 |

The dev-calibrated target FSM used activation/continue/release thresholds
`0.55 / 0.35 / 0.20`, three activation frames, and twelve release frames. It
reached 100/100 target-event recall, 90 ms median detection latency, 393 ms P95
detection latency, and 99% release success. These are dev results, not held-out
test results.

Known limitations are material: no generated test evaluation was performed;
near-field non-target active rate was 33.42% against a <=2% product goal;
overlap release P95 was 1,624.5 ms; high-noise non-speech active rate was
15.45%; and the recipe is mostly AISHELL-1 plus synthetic mixing/noise/TTS. It
does not establish robustness to real mobile acoustics, enrollment drift, or a
broad range of unseen speakers.

Full aggregate metrics, confusion matrices, raw-threshold results, calibrated
FSM slices, and limitations are in `metadata/DEV_METRICS.json`.

## Run the example

Install the base and speaker-export dependencies, then export two enrollment
clips with the open-source CAM++ backend:

```bash
pip install -r requirements.txt -r requirements-speaker.txt
python -m speaker_backends.modelscope_export \
  enrollment_1.wav enrollment_2.wav \
  --output-dir data/enrollment
```

Run Personal VAD on current audio:

```bash
python examples/pilot4k_speaker_aware_inference.py \
  --audio current.wav \
  --embedding data/enrollment/enrollment_1.spk.npy \
  --embedding data/enrollment/enrollment_2.spk.npy \
  --output pvad_result.json
```

The example verifies the checkpoint checksum, normalizes/aggregates the CAM++
embeddings exactly as in training, extracts the 512-dimensional acoustic
features, and emits merged target/non-target/non-speech segments. The emitted
segments use raw three-class argmax predictions; they are intentionally not
presented as the calibrated FSM evaluation.
