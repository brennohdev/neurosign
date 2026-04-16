"""Adaptador ONNX Runtime para inferência — implementa InferencePort."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurosign_backend.domain.entities import Prediction


class OnnxInferenceAdapter:
    """Carrega um modelo ONNX e executa inferência, implementando InferencePort.

    A sessão ONNX é criada uma única vez no __init__ e reutilizada em todas
    as chamadas a predict().
    """

    def __init__(self, model_path: Path, label_map: list[str]) -> None:
        """Inicializa o adaptador carregando a sessão ONNX.

        Args:
            model_path: Caminho para o arquivo .onnx.
            label_map: Lista de rótulos de classe na mesma ordem das saídas
                       do modelo (índice i → label_map[i]).

        Raises:
            RuntimeError: Se o arquivo .onnx não existir no caminho informado.
        """
        if not model_path.exists():
            raise RuntimeError(
                f"Modelo ONNX não encontrado: '{model_path}'. "
                "Verifique se MODEL_PATH aponta para um arquivo .onnx válido."
            )

        import onnxruntime as ort  # importação local para facilitar testes sem o pacote

        self._session = ort.InferenceSession(str(model_path))
        self._label_map = label_map

    def predict(self, window: np.ndarray) -> list[Prediction]:
        """Executa inferência sobre uma janela temporal.

        Args:
            window: Array de shape (window_size, 84), dtype float32.

        Returns:
            Lista de até 5 Prediction ordenada por confidence descendente,
            com rank de 1 a 5.
        """
        # Adiciona dimensão batch: (window_size, 84) → (1, window_size, 84)
        batch = window[np.newaxis, ...].astype(np.float32)

        outputs = self._session.run(None, {"input": batch})
        logits = outputs[0][0]  # shape: (num_classes,)

        # Softmax manual com numpy (sem scipy)
        shifted = logits - logits.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()

        # Top-5 por índice de confiança descendente
        top_k = min(5, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = [
            Prediction(
                label=self._label_map[idx],
                confidence=float(probs[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]

        return predictions
