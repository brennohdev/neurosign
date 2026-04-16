"""Entrypoint de treinamento do NeuroSign com augmentation e cosine annealing.

Uso:
    uv run python -m neurosign_ml.training.train \
        --manifest data/processed/landmarks_holistic/manifest.json \
        --labels   data/processed/landmarks_holistic/labels.json \
        --output   checkpoints_holistic/ \
        --epochs   100
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neurosign_ml.data.dataset import SignDataset
from neurosign_ml.models.bilstm_attention import BiLSTMAttention
from neurosign_ml.training.trainer import Trainer

logger = logging.getLogger(__name__)


def collate_fn(batch: list[tuple[torch.Tensor, int]]):
    sequences, labels = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels, dtype=torch.long)


def load_manifest(manifest_path: Path, labels_path: Path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(labels_path) as f:
        label_map: list[str] = json.load(f)

    base_dir = manifest_path.parent.parent
    samples = []
    for entry in manifest:
        npy_path = base_dir / entry["landmarks_path"]
        if not npy_path.exists():
            continue
        landmarks = np.load(str(npy_path))
        samples.append({
            "gloss": entry["gloss"],
            "subset": entry["subset"],
            "landmarks": landmarks,
        })
    return samples, label_map


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    logger.info("Carregando dados...")
    samples, label_map = load_manifest(args.manifest, args.labels)
    num_classes = len(label_map)

    # Detecta número de features do primeiro sample
    input_size = samples[0]["landmarks"].shape[1]
    logger.info("Samples: %d | Classes: %d | Features/frame: %d", len(samples), num_classes, input_size)

    train_samples = [s for s in samples if s["subset"] == "train"]
    val_samples   = [s for s in samples if s["subset"] == "val"]
    test_samples  = [s for s in samples if s["subset"] == "test"]
    logger.info("Train: %d | Val: %d | Test: %d", len(train_samples), len(val_samples), len(test_samples))

    use_augment = not args.no_augment
    train_ds = SignDataset(train_samples, label_map, augment=use_augment)
    val_ds   = SignDataset(val_samples, label_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = BiLSTMAttention(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Modelo: %d parâmetros | hidden=%d | dropout=%.1f | augment=%s",
                total_params, args.hidden_size, args.dropout, use_augment)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing com warmup de 5 épocas
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device="auto",
        checkpoint_dir=args.output,
        scheduler=scheduler,
    )

    trainer.fit(train_loader, val_loader, epochs=args.epochs, resume=args.resume)
    logger.info("Treinamento concluído. Checkpoints em: %s", args.output)


if __name__ == "__main__":
    main()
