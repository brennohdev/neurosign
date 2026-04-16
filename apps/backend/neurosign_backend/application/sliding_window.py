"""Implementação do buffer de sliding window para sessões WebSocket."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class SlidingWindowBuffer:
    """Buffer de sliding window por sessão, implementa SessionPort.

    Mantém um deque por session_id sem maxlen — o controle de tamanho é
    manual para que o stride seja aplicado corretamente após cada emissão.
    """

    def __init__(self, window_size: int, stride: int) -> None:
        self._window_size = window_size
        self._stride = stride
        self._buffers: dict[str, deque] = {}

    def add_frame(
        self, session_id: str, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """Adiciona um frame ao buffer da sessão.

        Args:
            session_id: identificador único da sessão WebSocket.
            frame: array de shape (84,), dtype float32.

        Returns:
            Janela de shape (window_size, 84) quando o buffer atingir
            window_size frames; None caso contrário.
        """
        if session_id not in self._buffers:
            self._buffers[session_id] = deque()

        buf = self._buffers[session_id]
        buf.append(frame)

        if len(buf) == self._window_size:
            window = np.array(list(buf), dtype=np.float32)
            # Descarta os `stride` frames mais antigos (início do deque)
            for _ in range(self._stride):
                buf.popleft()
            return window

        return None

    def clear_session(self, session_id: str) -> None:
        """Remove o buffer da sessão (chamado na desconexão).

        Args:
            session_id: identificador único da sessão a remover.
        """
        self._buffers.pop(session_id, None)
