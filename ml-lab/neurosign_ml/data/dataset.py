"""PyTorch Dataset para sequências de landmarks com augmentation opcional."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from neurosign_ml.data.augment import augment_sequence


class SignDataset(Dataset):
    """Dataset de sequências de landmarks com suporte a data augmentation.

    Args:
        samples: Lista de dicts com {"gloss": str, "landmarks": np.ndarray (T, F)}.
        label_map: Lista ordenada de glosses que define o índice de classe.
        augment: Se True, aplica augmentation aleatória (usar apenas no treino).
        hand_features: Número de features das mãos (84 para só mãos, 84 para Holistic também).
    """

    def __init__(
        self,
        samples: list[dict],
        label_map: list[str],
        augment: bool = False,
        hand_features: int = 84,
    ) -> None:
        self.samples = samples
        self.label_map = label_map
        self.augment = augment
        self.hand_features = hand_features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        landmarks = np.asarray(sample["landmarks"], dtype=np.float32)
        tensor = torch.from_numpy(landmarks)  # (T, F)

        if self.augment:
            tensor = augment_sequence(tensor, hand_features=self.hand_features)

        label_idx = self.label_map.index(sample["gloss"])
        return tensor, label_idx
