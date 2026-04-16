"""Landmark normalization — mirrors the logic in normalizer.ts.

Layout: [hand0_lm0_x, hand0_lm0_y, ..., hand1_lm20_x, hand1_lm20_y]
        84 floats total (2 hands × 21 landmarks × 2 coords).

Requisitos: 6.3
"""

from __future__ import annotations

import numpy as np


def normalize_landmarks(raw: np.ndarray) -> np.ndarray:
    """Normalize a single frame of hand landmarks.

    Steps (mirrors normalizer.ts):
    1. Translation — subtract wrist (landmark[0], indices [0, 1]) from all points.
    2. Scale — divide by the Euclidean distance between landmark[0] and
       landmark[9] (indices [18, 19]).
    3. If distance == 0 (hand absent), return zeros.

    Args:
        raw: Shape ``(84,)``, dtype float32.

    Returns:
        Normalized array, shape ``(84,)``, dtype float32.
    """
    raw = np.asarray(raw, dtype=np.float32)

    wrist_x = raw[0]
    wrist_y = raw[1]
    lm9_x = raw[18]
    lm9_y = raw[19]

    dx = lm9_x - wrist_x
    dy = lm9_y - wrist_y
    distance = float(np.sqrt(dx * dx + dy * dy))

    if distance == 0.0:
        return np.zeros(84, dtype=np.float32)

    result = raw.copy()
    result[0::2] = (raw[0::2] - wrist_x) / distance  # x coords
    result[1::2] = (raw[1::2] - wrist_y) / distance  # y coords

    return result


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """Normalize every frame in a temporal sequence.

    Args:
        sequence: Shape ``(T, 84)``, dtype float32.

    Returns:
        Normalized array, shape ``(T, 84)``, dtype float32.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    return np.stack([normalize_landmarks(frame) for frame in sequence], axis=0)
