import argparse
from pathlib import Path

import numpy as np


DEFAULT_MODEL_ID = 'iic/speech_campplus_sv_zh-cn_16k-common'


def parse_args():
  parser = argparse.ArgumentParser(
      description='Export speaker embeddings with a ModelScope SV model.')
  parser.add_argument('audio', nargs='+',
                      help='Input 16 kHz mono wav files or other audio files '
                           'supported by soundfile.')
  parser.add_argument('--model', default=DEFAULT_MODEL_ID,
                      help='ModelScope model id or a local cached model dir.')
  parser.add_argument('--output-dir', required=True,
                      help='Directory where .npy embeddings will be written.')
  parser.add_argument('--suffix', default='.spk.npy',
                      help='Suffix appended to each audio stem.')
  return parser.parse_args()


def load_pipeline(model):
  try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
  except ImportError as exc:
    raise RuntimeError(
        'ModelScope speaker export requires modelscope and soundfile. '
        'Install them with: pip install -r requirements-speaker.txt') from exc

  return pipeline(task=Tasks.speaker_verification, model=model)


def export_embeddings(model, audio_paths, output_dir, suffix):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  sv_pipeline = load_pipeline(model)
  written = []

  for audio_path in audio_paths:
    audio_path = Path(audio_path)
    result = sv_pipeline([str(audio_path)], output_emb=True)
    embedding = np.asarray(result['embs'], dtype=np.float32)
    if embedding.shape[0] != 1:
      raise RuntimeError(
          f'Expected one embedding for {audio_path}, got {embedding.shape}.')

    output_path = output_dir / f'{audio_path.stem}{suffix}'
    np.save(output_path, embedding[0])
    written.append((audio_path, output_path, embedding.shape[1]))

  return written


def main():
  args = parse_args()
  written = export_embeddings(args.model, args.audio, args.output_dir,
                              args.suffix)
  for audio_path, output_path, dim in written:
    print(f'{audio_path} -> {output_path} ({dim} dims)')


if __name__ == '__main__':
  main()
