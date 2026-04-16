"""Training loop, checkpoint management and metric logging for NeuroSign.

Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import csv
import heapq
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute Top-k accuracy over a batch.

    Args:
        outputs: Logits tensor of shape (B, num_classes).
        targets: Ground-truth class indices of shape (B,).
        k:       Number of top predictions to consider.

    Returns:
        Accuracy as a float in [0.0, 1.0].
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(k, dim=1, largest=True, sorted=True)  # (B, k)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().sum().item() / batch_size


def _detect_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the training loop, checkpointing and metric logging.

    Args:
        model:           PyTorch model to train.
        optimizer:       Optimiser instance.
        criterion:       Loss function (e.g. ``nn.CrossEntropyLoss()``).
        device:          Device string ``"cuda" | "mps" | "cpu"``.
                         Pass ``"auto"`` (or an empty string) to auto-detect.
        checkpoint_dir:  Directory where checkpoints and metrics are saved.
        max_checkpoints: Maximum number of checkpoints to keep (default 3).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        checkpoint_dir: Path,
        max_checkpoints: int = 3,
        scheduler=None,
    ) -> None:
        if not device or device == "auto":
            device = _detect_device()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.scheduler = scheduler

        self.model.to(self.device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "tb_logs"))
        self._metrics_path = self.checkpoint_dir / "metrics.csv"

        # Min-heap of (val_top1, checkpoint_path) — keeps track of saved ckpts
        # We use a min-heap so the worst checkpoint is always at index 0.
        self._saved_checkpoints: list[tuple[float, str]] = []

        self._start_epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Run one training epoch.

        Returns:
            Dict with keys ``"loss"``, ``"top1"``, ``"top5"`` (floats).
        """
        self.model.train()
        return self._run_epoch(dataloader, training=True)

    def eval_epoch(self, dataloader: DataLoader) -> dict:
        """Run one evaluation epoch (no gradient updates).

        Returns:
            Dict with keys ``"loss"``, ``"top1"``, ``"top5"`` (floats).
        """
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(dataloader, training=False)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume: Optional[Path] = None,
    ) -> None:
        """Full training loop.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.
            epochs:       Total number of epochs to train.
            resume:       Optional path to a checkpoint to resume from.
        """
        if resume is not None:
            self._load_checkpoint(resume)

        self._init_csv()

        for epoch in range(self._start_epoch, epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)

            self._log_metrics(epoch, train_metrics, val_metrics)
            self._save_checkpoint(epoch, val_metrics["top1"], val_metrics["top5"])

            import logging as _logging
            _logging.getLogger(__name__).info(
                "Época %d/%d | loss=%.4f top1=%.3f top5=%.3f | val_loss=%.4f val_top1=%.3f val_top5=%.3f",
                epoch + 1, epochs,
                train_metrics["loss"], train_metrics["top1"], train_metrics["top5"],
                val_metrics["loss"], val_metrics["top1"], val_metrics["top5"],
            )

        self._writer.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, dataloader: DataLoader, *, training: bool) -> dict:
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        n_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if training:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if training:
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            num_classes = outputs.size(1)
            k5 = min(5, num_classes)

            total_loss += loss.item()
            total_top1 += top_k_accuracy(outputs, targets, k=1)
            total_top5 += top_k_accuracy(outputs, targets, k=k5)
            n_batches += 1

        if n_batches == 0:
            return {"loss": 0.0, "top1": 0.0, "top5": 0.0}

        return {
            "loss": total_loss / n_batches,
            "top1": total_top1 / n_batches,
            "top5": total_top5 / n_batches,
        }

    def _log_metrics(
        self,
        epoch: int,
        train: dict,
        val: dict,
    ) -> None:
        # TensorBoard
        for key in ("loss", "top1", "top5"):
            self._writer.add_scalar(f"train/{key}", train[key], epoch)
            self._writer.add_scalar(f"val/{key}", val[key], epoch)

        # CSV
        row = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_top1": train["top1"],
            "train_top5": train["top5"],
            "val_loss": val["loss"],
            "val_top1": val["top1"],
            "val_top5": val["top5"],
        }
        with open(self._metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _init_csv(self) -> None:
        """Write CSV header if the file does not exist yet."""
        if not self._metrics_path.exists():
            with open(self._metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss", "train_top1", "train_top5",
                        "val_loss", "val_top1", "val_top5",
                    ],
                )
                writer.writeheader()

    def _save_checkpoint(self, epoch: int, val_top1: float, val_top5: float) -> None:
        ckpt_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "config": {},
            },
            ckpt_path,
        )

        # Push onto the min-heap (worst val_top1 at root)
        heapq.heappush(self._saved_checkpoints, (val_top1, str(ckpt_path)))

        # Evict the worst checkpoint when we exceed max_checkpoints
        while len(self._saved_checkpoints) > self.max_checkpoints:
            worst_score, worst_path = heapq.heappop(self._saved_checkpoints)
            try:
                Path(worst_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _load_checkpoint(self, path: Path) -> None:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._start_epoch = ckpt.get("epoch", 0) + 1
