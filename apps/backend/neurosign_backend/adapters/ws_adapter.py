"""WebSocketSessionAdapter — handler FastAPI para o endpoint /ws/{session_id}."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

from neurosign_backend.domain.entities import InferenceResult

if TYPE_CHECKING:
    from neurosign_backend.application.run_inference import RunInferenceUseCase

logger = logging.getLogger(__name__)

FRAME_SIZE = 150  # mãos (84) + pose corporal (66)


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    use_case: "RunInferenceUseCase",
) -> None:
    """Handler assíncrono para o endpoint WebSocket /ws/{session_id}.

    Fluxo por mensagem recebida:
      1. Deserializa JSON — fecha com código 1003 se inválido.
      2. Valida que ``frame`` tem exatamente 84 elementos — descarta silenciosamente se não.
      3. Converte para ``np.ndarray`` shape (84,) dtype float32.
      4. Chama ``use_case.process(session_id, frame)``.
      5. Se retornar ``InferenceResult``, serializa e envia ``PredictionMessage``.

    Na desconexão (``WebSocketDisconnect``), limpa o buffer da sessão.

    Args:
        websocket: Instância do WebSocket FastAPI.
        session_id: Identificador único da sessão.
        use_case: Caso de uso de inferência que orquestra sliding window + ONNX.
    """
    await websocket.accept()
    logger.info("Sessão %s conectada.", session_id)

    try:
        while True:
            raw = await websocket.receive_text()

            # 1. Deserializa JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(
                    "Sessão %s: JSON inválido recebido — fechando com código 1003.",
                    session_id,
                )
                await websocket.close(code=1003)
                return

            # 2. Valida dimensão do frame
            frame_list = data.get("frame")
            if not isinstance(frame_list, list) or len(frame_list) != FRAME_SIZE:
                logger.warning(
                    "Sessão %s: frame com %s elementos (esperado %d) — descartado.",
                    session_id,
                    len(frame_list) if isinstance(frame_list, list) else "?",
                    FRAME_SIZE,
                )
                continue

            # 3. Converte para ndarray float32
            frame: np.ndarray = np.array(frame_list, dtype=np.float32)

            # 4. Processa via caso de uso
            result = use_case.process(session_id, frame)

            # 5. Envia resposta se janela completa
            if isinstance(result, InferenceResult):
                message = {
                    "predictions": [
                        {
                            "label": p.label,
                            "confidence": p.confidence,
                            "rank": p.rank,
                        }
                        for p in result.predictions
                    ],
                    "session_id": session_id,
                    "timestamp_ms": int(time.time() * 1000),
                }
                await websocket.send_text(json.dumps(message))

    except WebSocketDisconnect:
        logger.info("Sessão %s desconectada — limpando buffer.", session_id)
        use_case._session_port.clear_session(session_id)
