"""Entrypoint FastAPI do NeuroSign Backend."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

from neurosign_backend.adapters.config import EnvConfig
from neurosign_backend.adapters.onnx_adapter import OnnxInferenceAdapter
from neurosign_backend.adapters.ws_adapter import websocket_endpoint
from neurosign_backend.application.run_inference import RunInferenceUseCase
from neurosign_backend.application.sliding_window import SlidingWindowBuffer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Inicializa e finaliza recursos da aplicação."""
    config = EnvConfig.from_env()

    # Carrega label_map do arquivo JSON ao lado do modelo, com fallback numérico
    labels_path = config.model_path.parent / "labels.json"
    if labels_path.exists():
        with labels_path.open() as f:
            label_map: list[str] = json.load(f)
        logger.info("label_map carregado de %s (%d classes).", labels_path, len(label_map))
    else:
        # Fallback: strings numéricas "0", "1", ... até 50 classes
        label_map = [str(i) for i in range(50)]
        logger.warning(
            "%s não encontrado — usando label_map numérico de fallback (%d classes).",
            labels_path,
            len(label_map),
        )

    onnx_adapter = OnnxInferenceAdapter(config.model_path, label_map)
    sliding_window = SlidingWindowBuffer(config.window_size, config.stride)
    use_case = RunInferenceUseCase(sliding_window, onnx_adapter)

    app.state.use_case = use_case
    app.state.config = config

    logger.info(
        "NeuroSign Backend iniciado — window_size=%d, stride=%d, model=%s",
        config.window_size,
        config.stride,
        config.model_path,
    )

    yield

    logger.info("NeuroSign Backend encerrando.")


app = FastAPI(title="NeuroSign Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURAÇÃO DE CORS ---
# Permite que o React (porta 5173 ou 3000) converse com o FastAPI (porta 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, trocaríamos "*" pelo domínio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Endpoint de health check."""
    return {"status": "ok"}


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
    """Endpoint WebSocket — delega para websocket_endpoint."""
    await websocket_endpoint(websocket, session_id, app.state.use_case)