# Personal-vad-2.0

PyTorch implementation of "Personal VAD 2.0: Optimizing Personal Voice
Activity Detection for On-Device Speech Recognition".

The default model follows the paper setup: 512-dim stacked acoustic features,
64-dim Conformer blocks, 4 encoder layers, a 2-layer speaker pre-net, FiLM
conditioning from speaker cosine similarity, and 3 frame classes:

- `0`: target speaker speech
- `1`: non-target speaker speech
- `2`: non-speech

![Personal VAD 2.0](./personal_vad2.0.png)

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test

```bash
python tests/smoke_test.py
python train.py --smoke-test --epochs 1 --batch-size 2 --device cpu
```

## Data Manifest

Training uses a JSONL or CSV manifest. Paths are resolved relative to the
manifest file. Features and labels can be `.pt`, `.pth`, `.npy`, or `.npz`.

Example JSONL:

```json
{"id": "utt001", "features": "utt001_feat.pt", "labels": "utt001_labels.pt", "embedding": "utt001_spk.pt"}
{"id": "utt002", "features": "utt002_feat.npy", "labels": "utt002_labels.npy"}
```

Each feature tensor must have shape `(frames, 512)`, each label tensor must have
shape `(frames,)`, and each speaker embedding must have shape `(64,)` by
default. If `embedding` is omitted, the sample is treated as enrollment-less and
uses a zero speaker embedding.

The in-repo `PvadFeatureExtractor` implements the paper-published dimensions:
128 log-Mel energies from a 32 ms window at a 10 ms shift, four contiguous
frames stacked to 512 dimensions, followed by factor-three time subsampling.
The product frontend makes that stack causal (current plus three past frames).
The paper does not publish a CMVN, window, Mel scale, or FFT-size recipe, so
those choices are explicit rather than presented as paper facts: no CMVN,
Hann window, HTK Mel scale, and a 1024-point zero-padded FFT. The complete
`PvadFeatureConfig` must match between data generation and inference.

## Train

```bash
python train.py \
  --train-manifest data/train.jsonl \
  --valid-manifest data/valid.jsonl \
  --output-dir runs/pvad2 \
  --epochs 20 \
  --batch-size 16
```

The training loop implements the enrollment-less paradigm from the paper:
with `--enrollment-drop-prob 0.2`, a sampled utterance has its speaker embedding
replaced with zeros and its non-target speech labels (`1`) rewritten to target
speech (`0`). Checkpoints are written to `last.pt` and `best.pt`.
By default, `best.pt` is selected by validation target-class F1 rather than
loss. `--selection-metric macro_f1` and `--selection-metric loss` are
available for controlled comparisons. Metrics include all three per-class
precision/recall/F1/support values, a confusion matrix, macro-F1, and
target-vs-rest approximate PR/ROC summaries. `--max-steps-per-epoch` bounds
20-step batch-size and stability checks without starting a long run.

## Streaming Inference

`PvadStreamingAdapter` accepts arbitrary mono-audio chunks and returns aligned
three-class logits/probabilities. The audio frontend already applies factor-three
time subsampling, so the corresponding model must use `subsampling='linear'`.
Each Conformer layer keeps bounded attention K/V and causal-convolution caches.

```python
import torch

from model import Pvad2
from streaming import PvadStreamingAdapter

model = Pvad2.load_model('best.pt').eval()
embedding = torch.from_numpy(campplus_embedding)
adapter = PvadStreamingAdapter(
    model, embedding, device='cuda', measure_rtf=False)
output = adapter.feed_audio(mono_pcm_chunk)
target_probability = output.probabilities[:, 0]
```

Set `measure_rtf=True` only for benchmarking: CUDA synchronization is then
included in the reported cumulative real-time factor. Call `reset()` at an
utterance/session boundary; changing batch size or reusing caches across
independent audio is rejected.

## Speaker Embeddings

The model can consume external speaker embeddings by changing
`--speaker-embedding-dim`. A local smoke test with ModelScope CAM++ works with
192-dim embeddings:

```bash
pip install -r requirements-speaker.txt
python -m speaker_backends.modelscope_export \
  path/to/enrollment.wav \
  --output-dir data/embeddings
```

Then reference the generated `.spk.npy` path from the manifest and train with:

```bash
python train.py \
  --train-manifest data/train.jsonl \
  --speaker-embedding-dim 192
```

For offline use after the first download, pass the local cached model directory
instead of the model id, for example:

```bash
python -m speaker_backends.modelscope_export \
  path/to/enrollment.wav \
  --model C:\Users\alanf\.cache\modelscope\hub\models\iic\speech_campplus_sv_zh-cn_16k-common \
  --output-dir data/embeddings
```
