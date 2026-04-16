"""EnvConfig — lê e valida variáveis de ambiente do NeuroSign Backend."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EnvConfig:
    window_size: int
    stride: int
    model_path: Path
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Lê variáveis de ambiente e encerra o processo listando todas as ausentes."""
        missing: list[str] = []

        raw_window_size = os.environ.get("WINDOW_SIZE")
        raw_stride = os.environ.get("STRIDE")
        raw_model_path = os.environ.get("MODEL_PATH")

        if raw_window_size is None:
            missing.append("WINDOW_SIZE")
        if raw_stride is None:
            missing.append("STRIDE")
        if raw_model_path is None:
            missing.append("MODEL_PATH")

        if missing:
            sys.exit(
                f"Erro: variáveis de ambiente obrigatórias ausentes: {', '.join(missing)}"
            )

        return cls(
            window_size=int(raw_window_size),  # type: ignore[arg-type]
            stride=int(raw_stride),  # type: ignore[arg-type]
            model_path=Path(raw_model_path),  # type: ignore[arg-type]
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8000")),
        )
