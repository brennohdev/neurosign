"""Ports (interfaces) de domínio do NeuroSign — sem dependências de I/O."""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from neurosign_backend.domain.entities import Prediction


class InferencePort(Protocol):
    """Interface para o motor de inferência."""

    def predict(self, window: np.ndarray) -> list[Prediction]:
        """Executa inferência sobre uma janela temporal.

        Args:
            window: array de shape (window_size, 84), dtype float32.

        Returns:
            Lista de Prediction ordenada por score descendente.
        """
        ...


class SessionPort(Protocol):
    """Interface para gerenciamento de buffer de sessão (sliding window)."""

    def add_frame(
        self, session_id: str, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Adiciona um frame ao buffer da sessão.

        Args:
            session_id: identificador único da sessão WebSocket.
            frame: array de shape (84,), dtype float32.

        Returns:
            Janela completa de shape (window_size, 84) quando o buffer
            atingir window_size frames; None caso contrário.
        """
        ...
