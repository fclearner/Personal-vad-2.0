# Third-party software, models, and data

The repository source code is released under Apache-2.0. That license does not
replace the terms of external software, pretrained models, or datasets used to
prepare a training run.

| Component | How it is used | Terms / source |
|---|---|---|
| WeNet / ESPnet-derived model layers | Conformer and attention implementation | [WeNet](https://github.com/wenet-e2e/wenet), Apache-2.0; retained source-file notices apply |
| ModelScope 3D-Speaker CAM++ | Frozen 192-dimensional enrollment embeddings | [3D-Speaker](https://github.com/modelscope/3D-Speaker), Apache-2.0; weights are not bundled |
| AISHELL-1 | Speaker-separated Mandarin source speech | [OpenSLR SLR33](https://www.openslr.org/33/), Apache-2.0; audio is not bundled |
| WHAM! noise | Background-noise augmentation | [WHAM!](https://wham.whisper.ai/), CC BY-NC 4.0; audio is not bundled |
| Qwen3-Omni-30B-A3B-Instruct | Persisted synthetic TTS playback examples | [Qwen3-Omni model card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct), Apache-2.0; model and generated audio are not bundled |
| ModelScope FSMN VAD | Offline speech intervals during mixture generation | Obtain from ModelScope under the selected model's terms; weights are not bundled |

`checkpoints/pilot4k_speaker_aware_epoch38/best_inference.pt` was trained from
the source categories above. In particular, WHAM! is non-commercial. The
checkpoint is provided for research and evaluation; this repository does not
grant rights beyond the applicable upstream data/model terms. Review those
terms before redistribution or commercial use.

Enrollment audio and embeddings are biometric data. Do not commit them to this
repository, and collect or retain them only with an appropriate legal basis and
user consent.
