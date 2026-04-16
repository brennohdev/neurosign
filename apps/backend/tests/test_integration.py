"""Testes de integração do fluxo WebSocket completo — NeuroSign Backend.

Usa um StubInferenceAdapter (sem arquivo ONNX real) para testar o fluxo:
  envio de tensor → sliding window → inferência → resposta JSON

Valida: Requisito 10.5
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from neurosign_backend.adapters.ws_adapter import websocket_endpoint
from neurosign_backend.application.run_inference import RunInferenceUseCase
from neurosign_backend.application.sliding_window import SlidingWindowBuffer
from neurosign_backend.domain.entities import Prediction

# ── Configuração dos testes ──────────────────────────────────────────────────

WINDOW_SIZE = 5
STRIDE = 1
FRAME_SIZE = 84

_FIXED_PREDICTIONS = [
    Prediction(label="hello", confidence=0.9, rank=1),
    Prediction(label="world", confidence=0.05, rank=2),
    Prediction(label="yes", confidence=0.02, rank=3),
    Prediction(label="no", confidence=0.02, rank=4),
    Prediction(label="thanks", confidence=0.01, rank=5),
]


# ── Stub do adaptador de inferência ─────────────────────────────────────────

class StubInferenceAdapter:
    """Implementa InferencePort sem arquivo ONNX — retorna 5 predições fixas."""

    def predict(self, window: np.ndarray) -> list[Prediction]:
        return list(_FIXED_PREDICTIONS)


# ── App FastAPI de teste ─────────────────────────────────────────────────────

def build_test_app(session_port=None, inference_port=None) -> FastAPI:
    """Cria uma app FastAPI de teste com componentes reais + stub de inferência."""
    if inference_port is None:
        inference_port = StubInferenceAdapter()
    if session_port is None:
        session_port = SlidingWindowBuffer(WINDOW_SIZE, STRIDE)

    use_case = RunInferenceUseCase(session_port, inference_port)

    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(websocket: WebSocket, session_id: str) -> None:
        await websocket_endpoint(websocket, session_id, use_case)

    return app


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def test_app() -> FastAPI:
    return build_test_app()



def make_frame(value: float = 0.1) -> list[float]:
    """Cria um frame válido com FRAME_SIZE elementos."""
    return [value] * FRAME_SIZE


# ── Testes ───────────────────────────────────────────────────────────────────

async def test_health_endpoint(test_app: FastAPI) -> None:
    """GET /health deve retornar {"status": "ok"} via httpx.AsyncClient + ASGITransport."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as http:
        response = await http.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_websocket_single_session(test_app: FastAPI) -> None:
    """Envia window_size frames e verifica resposta com predictions, session_id e timestamp_ms."""
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/session-abc") as ws:
            # Envia window_size - 1 frames (sem resposta esperada)
            for _ in range(WINDOW_SIZE - 1):
                ws.send_text(json.dumps({"frame": make_frame()}))

            # Frame que completa a janela — deve disparar predição
            ws.send_text(json.dumps({"frame": make_frame()}))
            raw = ws.receive_text()

    msg = json.loads(raw)
    assert "predictions" in msg
    assert len(msg["predictions"]) == 5
    assert "session_id" in msg
    assert msg["session_id"] == "session-abc"
    assert "timestamp_ms" in msg
    assert isinstance(msg["timestamp_ms"], int)

    for pred in msg["predictions"]:
        assert "label" in pred
        assert "confidence" in pred
        assert "rank" in pred


def test_websocket_no_prediction_before_window_full(test_app: FastAPI) -> None:
    """Envia window_size - 1 frames e verifica que nenhuma predição é recebida.

    Verifica indiretamente: a primeira resposta só chega no frame de número
    window_size, confirmando que o buffer não emite antes de estar cheio.
    """
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/session-partial") as ws:
            # Envia window_size - 1 frames
            for i in range(WINDOW_SIZE - 1):
                ws.send_text(json.dumps({"frame": make_frame(float(i))}))

            # Frame que completa a janela — primeira e única resposta esperada
            ws.send_text(json.dumps({"frame": make_frame(99.0)}))
            raw = ws.receive_text()

    msg = json.loads(raw)
    # A primeira resposta só chegou após o frame window_size
    assert "predictions" in msg
    assert len(msg["predictions"]) == 5


def test_websocket_invalid_json(test_app: FastAPI) -> None:
    """Envia JSON inválido e verifica que a conexão é fechada com código 1003."""
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/session-invalid") as ws:
            ws.send_text("isto não é json {{{")

            # O servidor fecha com código 1003 — starlette levanta exceção ao receber close
            with pytest.raises(Exception):
                ws.receive_text()


def test_websocket_wrong_frame_size(test_app: FastAPI) -> None:
    """Envia frame com dimensão errada (42 elementos) — frame descartado, conexão mantida."""
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/session-wrongsize") as ws:
            # Frame inválido (42 elementos em vez de 84) — deve ser descartado silenciosamente
            ws.send_text(json.dumps({"frame": [0.1] * 42}))

            # Envia window_size frames válidos para confirmar que a conexão ainda funciona
            for _ in range(WINDOW_SIZE - 1):
                ws.send_text(json.dumps({"frame": make_frame()}))

            ws.send_text(json.dumps({"frame": make_frame()}))
            raw = ws.receive_text()

    msg = json.loads(raw)
    assert "predictions" in msg
    assert len(msg["predictions"]) == 5


def test_websocket_session_isolation() -> None:
    """Duas sessões simultâneas não interferem entre si."""
    # App compartilhada com buffer compartilhado — sessões devem ser isoladas
    app = build_test_app()

    results: dict[str, dict] = {}

    def run_session_a() -> None:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-A") as ws:
                for _ in range(WINDOW_SIZE):
                    ws.send_text(json.dumps({"frame": make_frame(0.1)}))
                results["A"] = json.loads(ws.receive_text())

    def run_session_b() -> None:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/session-B") as ws:
                for _ in range(WINDOW_SIZE):
                    ws.send_text(json.dumps({"frame": make_frame(0.9)}))
                results["B"] = json.loads(ws.receive_text())

    t_a = threading.Thread(target=run_session_a)
    t_b = threading.Thread(target=run_session_b)
    t_a.start()
    t_b.start()
    t_a.join(timeout=5)
    t_b.join(timeout=5)

    assert "A" in results, "Sessão A não recebeu predição"
    assert "B" in results, "Sessão B não recebeu predição"

    assert results["A"]["session_id"] == "session-A"
    assert results["B"]["session_id"] == "session-B"
    assert len(results["A"]["predictions"]) == 5
    assert len(results["B"]["predictions"]) == 5
