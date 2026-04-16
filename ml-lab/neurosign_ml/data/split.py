"""Stratified dataset split without gloss leakage between sets.

Each gloss is assigned entirely to one split (train / val / test).

Requisitos: 6.4
"""

from __future__ import annotations

import random
from collections import defaultdict


def split_dataset(
    samples: list[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split samples into train / val / test without gloss leakage.

    All samples belonging to the same gloss are kept in the same split.
    Glosses are shuffled with the given seed for reproducibility, then
    assigned greedily to each split until the target ratio is reached.

    Args:
        samples: List of dicts, each with at least ``{"gloss": str, ...}``.
        train_ratio: Fraction of samples for training.
        val_ratio: Fraction of samples for validation.
        test_ratio: Fraction of samples for testing.
        seed: Random seed for reproducibility.

    Returns:
        ``(train, val, test)`` — three lists of sample dicts.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    # Group samples by gloss
    by_gloss: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_gloss[sample["gloss"]].append(sample)

    glosses = list(by_gloss.keys())
    rng = random.Random(seed)
    rng.shuffle(glosses)

    total = len(samples)
    train_target = int(round(total * train_ratio))
    val_target = int(round(total * val_ratio))

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for gloss in glosses:
        gloss_samples = by_gloss[gloss]
        if len(train) < train_target:
            train.extend(gloss_samples)
        elif len(val) < val_target:
            val.extend(gloss_samples)
        else:
            test.extend(gloss_samples)

    return train, val, test
