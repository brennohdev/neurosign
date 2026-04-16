"""Caso de uso de inferência — orquestra SessionPort e InferencePort."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from neurosign_backend.domain.entities import InferenceResult
from neurosign_backend.domain.ports import InferencePort, SessionPort


class RunInferenceUseCase:
    """Orquestra o fluxo: adicionar frame → (se janela pronta) inferir.

    Depende apenas de ports (interfaces), sem acoplamento a adaptadores
    concretos ou frameworks externos.
    """

    def __init__(
        self,
        session_port: SessionPort,
        inference_port: InferencePort,
    ) -> None:
        self._session_port = session_port
        self._inference_port = inference_port

    def process(
        self, session_id: str, frame: np.ndarray
    ) -> Optional[InferenceResult]:
        """Adiciona um frame à sessão e executa inferência se a janela estiver pronta.

        Args:
            session_id: Identificador único da sessão WebSocket.
            frame: Array de shape (84,), dtype float32.

        Returns:
            InferenceResult com predições e latência quando a janela estiver
            completa; None enquanto o buffer ainda estiver acumulando frames.
        """
        window = self._session_port.add_frame(session_id, frame)

        if window is None:
            return None

        start = time.perf_counter()
        predictions = self._inference_port.predict(window)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return InferenceResult(
            predictions=tuple(predictions),
            latency_ms=latency_ms,
        )
