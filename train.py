import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (DEFAULT_IGNORE_INDEX, PvadManifestDataset,
                     SyntheticPvadDataset, collate_pvad_batch)
from model.pvad2 import Pvad2


def parse_args():
  parser = argparse.ArgumentParser(description='Train Personal VAD 2.0.')
  parser.add_argument('--train-manifest', type=str, default=None,
                      help='JSONL/CSV manifest for training data.')
  parser.add_argument('--valid-manifest', type=str, default=None,
                      help='Optional JSONL/CSV manifest for validation data.')
  parser.add_argument('--output-dir', type=str, default='runs/pvad2',
                      help='Directory for checkpoints and metrics.')
  parser.add_argument('--resume', type=str, default=None,
                      help='Checkpoint to resume from.')
  parser.add_argument('--device', type=str, default=None,
                      help='cpu, cuda, or cuda:N. Defaults to CUDA if present.')
  parser.add_argument('--epochs', type=int, default=20)
  parser.add_argument('--batch-size', type=int, default=16)
  parser.add_argument('--num-workers', type=int, default=0)
  parser.add_argument('--lr', type=float, default=1e-3)
  parser.add_argument('--weight-decay', type=float, default=1e-4)
  parser.add_argument('--grad-clip', type=float, default=5.0)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--amp', action='store_true',
                      help='Use CUDA automatic mixed precision.')

  parser.add_argument('--input-dim', type=int, default=512)
  parser.add_argument('--encoder-dim', type=int, default=64)
  parser.add_argument('--speaker-embedding-dim', type=int, default=64)
  parser.add_argument('--num-classes', type=int, default=3)
  parser.add_argument('--num-encoder-layers', type=int, default=4)
  parser.add_argument('--num-speaker-layers', type=int, default=2)
  parser.add_argument('--num-attention-heads', type=int, default=8)
  parser.add_argument('--linear-units', type=int, default=64)
  parser.add_argument('--dropout-rate', type=float, default=0.1)
  parser.add_argument('--attention-dropout-rate', type=float, default=0.0)
  parser.add_argument('--conv-kernel-size', type=int, default=7)
  parser.add_argument('--left-context', type=int, default=31)
  parser.add_argument('--subsampling', choices=('linear', 'conv2d3'),
                      default='linear')

  parser.add_argument('--enrollment-drop-prob', type=float, default=0.2,
                      help='Probability p0 for enrollment-less augmentation.')
  parser.add_argument('--target-class', type=int, default=0)
  parser.add_argument('--non-target-class', type=int, default=1)
  parser.add_argument('--ignore-index', type=int, default=DEFAULT_IGNORE_INDEX)
  parser.add_argument('--smoke-test', action='store_true',
                      help='Train on a tiny synthetic dataset.')
  return parser.parse_args()


def resolve_device(device_arg):
  if device_arg:
    return torch.device(device_arg)
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(args):
  return Pvad2(input_dim=args.input_dim,
               encoder_dim=args.encoder_dim,
               speaker_embedding_dim=args.speaker_embedding_dim,
               num_classes=args.num_classes,
               num_encoder_layers=args.num_encoder_layers,
               num_speaker_layers=args.num_speaker_layers,
               num_attention_heads=args.num_attention_heads,
               linear_units=args.linear_units,
               dropout_rate=args.dropout_rate,
               attention_dropout_rate=args.attention_dropout_rate,
               conv_kernel_size=args.conv_kernel_size,
               left_context=args.left_context,
               subsampling=args.subsampling)


def build_datasets(args):
  if args.smoke_test:
    train_set = SyntheticPvadDataset(
        size=8, input_dim=args.input_dim,
        speaker_embedding_dim=args.speaker_embedding_dim,
        num_classes=args.num_classes, seed=args.seed)
    valid_set = SyntheticPvadDataset(
        size=4, input_dim=args.input_dim,
        speaker_embedding_dim=args.speaker_embedding_dim,
        num_classes=args.num_classes, seed=args.seed + 1000)
    return train_set, valid_set

  if args.train_manifest is None:
    raise ValueError('--train-manifest is required unless --smoke-test is set.')

  train_set = PvadManifestDataset(
      args.train_manifest, speaker_embedding_dim=args.speaker_embedding_dim)
  valid_set = None
  if args.valid_manifest:
    valid_set = PvadManifestDataset(
        args.valid_manifest, speaker_embedding_dim=args.speaker_embedding_dim)
  return train_set, valid_set


def build_loader(dataset, args, shuffle):
  return DataLoader(dataset,
                    batch_size=args.batch_size,
                    shuffle=shuffle,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=lambda batch: collate_pvad_batch(
                        batch, label_pad=args.ignore_index))


def apply_enrollment_dropout(embeddings, labels, prob, target_class,
                             non_target_class, ignore_index):
  if prob <= 0:
    return embeddings, labels
  sample_mask = torch.rand(embeddings.size(0), device=embeddings.device) < prob
  if not sample_mask.any():
    return embeddings, labels

  embeddings = embeddings.clone()
  labels = labels.clone()
  embeddings[sample_mask] = 0.0
  rewrite_mask = sample_mask.unsqueeze(1) & labels.eq(non_target_class)
  labels = torch.where(rewrite_mask, labels.new_full((), target_class), labels)
  labels = torch.where(labels.eq(ignore_index), labels.new_full((), ignore_index),
                       labels)
  return embeddings, labels


def align_targets_to_logits(labels, logits, ignore_index):
  if labels.size(1) == logits.size(1):
    return labels
  if labels.size(1) > logits.size(1):
    return labels[:, :logits.size(1)]
  pad = labels.new_full((labels.size(0), logits.size(1) - labels.size(1)),
                        ignore_index)
  return torch.cat([labels, pad], dim=1)


def frame_accuracy(logits, labels, ignore_index):
  valid = labels.ne(ignore_index)
  total = valid.sum().item()
  if total == 0:
    return 0, 0
  predictions = logits.argmax(dim=-1)
  correct = (predictions.eq(labels) & valid).sum().item()
  return correct, total


def run_epoch(model, loader, criterion, optimizer, scaler, device, args,
              training):
  model.train(training)
  total_loss = 0.0
  total_frames = 0
  total_correct = 0

  for batch in loader:
    features = batch['features'].to(device)
    labels = batch['labels'].to(device)
    embeddings = batch['embeddings'].to(device)
    lengths = batch['lengths'].to(device)

    if training:
      embeddings, labels = apply_enrollment_dropout(
          embeddings, labels, args.enrollment_drop_prob, args.target_class,
          args.non_target_class, args.ignore_index)
      optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(training):
      amp_enabled = args.amp and device.type == 'cuda'
      with torch.cuda.amp.autocast(enabled=amp_enabled):
        logits, _ = model(features, embeddings, lengths, return_lengths=True)
        labels = align_targets_to_logits(labels, logits, args.ignore_index)
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         labels.reshape(-1))

      if training:
        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
          scaler.unscale_(optimizer)
          nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    frames = labels.ne(args.ignore_index).sum().item()
    correct, total = frame_accuracy(logits.detach(), labels, args.ignore_index)
    total_loss += loss.item() * max(frames, 1)
    total_frames += frames
    total_correct += correct

  avg_loss = total_loss / max(total_frames, 1)
  accuracy = total_correct / max(total_frames, 1)
  return {'loss': avg_loss, 'accuracy': accuracy, 'frames': total_frames}


def save_checkpoint(path, model, optimizer, epoch, train_metrics, valid_metrics,
                    args):
  package = Pvad2.serialize(model, optimizer=optimizer, epoch=epoch,
                            tr_loss=train_metrics['loss'],
                            cv_loss=(valid_metrics or {}).get('loss'))
  package['args'] = vars(args)
  package['train_metrics'] = train_metrics
  package['valid_metrics'] = valid_metrics
  torch.save(package, path)


def append_metrics(path, row):
  with path.open('a', encoding='utf-8') as handle:
    handle.write(json.dumps(row, sort_keys=True) + '\n')


def main():
  args = parse_args()
  torch.manual_seed(args.seed)
  device = resolve_device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  train_set, valid_set = build_datasets(args)
  train_loader = build_loader(train_set, args, shuffle=True)
  valid_loader = build_loader(valid_set, args, shuffle=False) if valid_set else None

  model = build_model(args).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
  criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
  scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == 'cuda')

  start_epoch = 1
  best_valid = math.inf
  if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if 'optim_dict' in checkpoint:
      optimizer.load_state_dict(checkpoint['optim_dict'])
    start_epoch = int(checkpoint.get('epoch', 0)) + 1
    best_valid = float(checkpoint.get('best_valid', best_valid))

  metrics_path = output_dir / 'metrics.jsonl'
  for epoch in range(start_epoch, args.epochs + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer,
                              scaler, device, args, training=True)
    valid_metrics = None
    if valid_loader is not None:
      valid_metrics = run_epoch(model, valid_loader, criterion, optimizer,
                                scaler, device, args, training=False)

    row = {'epoch': epoch, 'train': train_metrics, 'valid': valid_metrics}
    append_metrics(metrics_path, row)
    save_checkpoint(output_dir / 'last.pt', model, optimizer, epoch,
                    train_metrics, valid_metrics, args)

    valid_loss = valid_metrics['loss'] if valid_metrics else train_metrics['loss']
    if valid_loss < best_valid:
      best_valid = valid_loss
      save_checkpoint(output_dir / 'best.pt', model, optimizer, epoch,
                      train_metrics, valid_metrics, args)

    valid_text = ''
    if valid_metrics:
      valid_text = (f" valid_loss={valid_metrics['loss']:.4f}"
                    f" valid_acc={valid_metrics['accuracy']:.4f}")
    print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f}"
          f" train_acc={train_metrics['accuracy']:.4f}{valid_text}")


if __name__ == '__main__':
  main()
