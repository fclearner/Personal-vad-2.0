from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.pvad2 import Pvad2
from train import apply_enrollment_dropout


def main():
  torch.manual_seed(0)
  model = Pvad2(dropout_rate=0.0)
  features = torch.randn(2, 10, 512)
  lengths = torch.tensor([10, 7])
  embeddings = torch.randn(2, 64)
  labels = torch.randint(0, 3, (2, 10))
  labels[1, 7:] = -100

  logits, output_lengths = model(
      features, embeddings, lengths, return_lengths=True)
  assert logits.shape == (2, 10, 3), logits.shape
  assert output_lengths.tolist() == [10, 7], output_lengths
  assert torch.isfinite(logits).all()

  dropped_embeddings, dropped_labels = apply_enrollment_dropout(
      embeddings, labels, prob=1.0, target_class=0, non_target_class=1,
      ignore_index=-100)
  assert torch.equal(dropped_embeddings, torch.zeros_like(embeddings))
  assert not dropped_labels.eq(1).any()
  assert dropped_labels[1, 7:].eq(-100).all()

  criterion = nn.CrossEntropyLoss(ignore_index=-100)
  loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
  loss.backward()
  assert torch.isfinite(loss)
  print('smoke ok')


if __name__ == '__main__':
  main()
