import argparse
from pathlib import Path

import numpy as np


DEFAULT_MODEL_ID = 'iic/speech_campplus_sv_zh-cn_16k-common'


def l2_normalize(embedding, eps=1e-12):
  """Return a float32 unit vector and reject degenerate embeddings."""
  embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
  if embedding.size == 0:
    raise ValueError('Cannot normalize an empty embedding.')
  norm = float(np.linalg.norm(embedding))
  if not np.isfinite(norm) or norm <= eps:
    raise ValueError('Cannot normalize a zero or non-finite embedding.')
  return embedding / norm


def aggregate_embeddings(embeddings):
  """L2-normalize each enrollment, average, then normalize the result."""
  matrix = np.asarray(embeddings, dtype=np.float32)
  if matrix.ndim != 2 or matrix.shape[0] == 0:
    raise ValueError('embeddings must have shape (utterances, dimensions).')
  normalized = np.stack(
      [l2_normalize(embedding) for embedding in matrix], axis=0)
  return l2_normalize(normalized.mean(axis=0))


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
  parser.add_argument('--normalize', action=argparse.BooleanOptionalAction,
                      default=True, help='L2-normalize exported embeddings.')
  parser.add_argument('--aggregate-output', default='',
                      help='Optional filename for the normalized group mean.')
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


def export_embeddings(model, audio_paths, output_dir, suffix, normalize=True):
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
    vector = l2_normalize(embedding[0]) if normalize else embedding[0]
    np.save(output_path, vector)
    written.append((audio_path, output_path, vector.size))

  return written


def main():
  args = parse_args()
  written = export_embeddings(
      args.model, args.audio, args.output_dir, args.suffix,
      normalize=args.normalize)
  for audio_path, output_path, dim in written:
    vector = np.load(output_path)
    print(f'{audio_path} -> {output_path} ({dim} dims, '
          f'norm={np.linalg.norm(vector):.6f})')
  if args.aggregate_output:
    aggregate = aggregate_embeddings([np.load(item[1]) for item in written])
    aggregate_path = Path(args.output_dir) / args.aggregate_output
    np.save(aggregate_path, aggregate)
    print(f'aggregate -> {aggregate_path} (norm={np.linalg.norm(aggregate):.6f})')


if __name__ == '__main__':
  main()
