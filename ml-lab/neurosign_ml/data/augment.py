"""Data augmentation para sequências de landmarks.

Técnicas implementadas:
  - Ruído gaussiano nos landmarks
  - Flip horizontal (espelha mão esquerda/direita)
  - Time warping (estica/comprime a sequência temporalmente)
  - Frame dropout (remove frames aleatórios e interpola)
"""

from __future__ import annotations

import numpy as np
import torch


def gaussian_noise(seq: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Adiciona ruído gaussiano à sequência."""
    return seq + torch.randn_like(seq) * std


def horizontal_flip(seq: torch.Tensor, hand_features: int = 84) -> torch.Tensor:
    """Espelha coordenadas x (1 - x) para simular mão oposta.

    Funciona para qualquer número de features — só inverte as coords x
    dos primeiros hand_features (mãos) e das features de pose se presentes.
    """
    result = seq.clone()
    # Inverte coords x (índices pares) dos landmarks das mãos
    result[:, 0:hand_features:2] = 1.0 - seq[:, 0:hand_features:2]
    # Se há pose (features além das mãos), inverte x da pose também
    if seq.shape[1] > hand_features:
        result[:, hand_features::2] = 1.0 - seq[:, hand_features::2]
    return result


def time_warp(seq: torch.Tensor, factor_range: tuple = (0.8, 1.2)) -> torch.Tensor:
    """Estica ou comprime a sequência temporalmente via interpolação."""
    T = seq.shape[0]
    factor = np.random.uniform(*factor_range)
    new_T = max(2, int(T * factor))

    # Interpola: (1, C, T) → (1, C, new_T) → (new_T, C)
    x = seq.T.unsqueeze(0)  # (1, C, T)
    warped = torch.nn.functional.interpolate(x, size=new_T, mode="linear", align_corners=False)
    return warped.squeeze(0).T  # (new_T, C)


def frame_dropout(seq: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
    """Remove frames aleatórios e interpola para manter o comprimento."""
    T = seq.shape[0]
    if T <= 2:
        return seq
    mask = torch.rand(T) > drop_prob
    if mask.sum() < 2:
        return seq
    kept = seq[mask]
    # Interpola de volta para T frames
    x = kept.T.unsqueeze(0)
    restored = torch.nn.functional.interpolate(x, size=T, mode="linear", align_corners=False)
    return restored.squeeze(0).T


def augment_sequence(
    seq: torch.Tensor,
    noise_std: float = 0.02,
    flip_prob: float = 0.5,
    warp_prob: float = 0.5,
    dropout_prob: float = 0.3,
    hand_features: int = 84,
) -> torch.Tensor:
    """Aplica pipeline de augmentation com probabilidades configuráveis."""
    if torch.rand(1).item() < flip_prob:
        seq = horizontal_flip(seq, hand_features=hand_features)
    if torch.rand(1).item() < warp_prob:
        seq = time_warp(seq)
    if torch.rand(1).item() < dropout_prob:
        seq = frame_dropout(seq)
    seq = gaussian_noise(seq, std=noise_std)
    return seq
