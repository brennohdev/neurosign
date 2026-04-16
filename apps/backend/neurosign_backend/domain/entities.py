"""Entidades de domínio do NeuroSign — independentes de frameworks externos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float  # [0.0, 1.0]
    rank: int          # 1-5


@dataclass(frozen=True)
class InferenceResult:
    predictions: tuple[Prediction, ...]  # sempre 5 elementos
    latency_ms: float
