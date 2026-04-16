"""Validação da qualidade da quantização ONNX (Float32 vs Int8).

Requisitos: 8.5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def _run_inference(session: ort.InferenceSession, samples: list[np.ndarray]) -> list[int]:
    """Executa inferência em todas as amostras e retorna as predições Top-1."""
    input_name = session.get_inputs()[0].name
    predictions: list[int] = []
    for sample in samples:
        x = sample if sample.ndim == 3 else sample[np.newaxis, ...]
        output = session.run(None, {input_name: x.astype(np.float32)})[0]
        predictions.append(int(np.argmax(output, axis=-1).flat[0]))
    return predictions


def validate_quantization(
    fp32_path: Path,
    int8_path: Path,
    test_samples: list[np.ndarray],
    labels: list[int],
    max_delta_pp: float = 2.0,
) -> dict:
    """Compara a acurácia Top-1 entre o modelo Float32 e o modelo Int8.

    Args:
        fp32_path: Caminho para o modelo ONNX Float32.
        int8_path: Caminho para o modelo ONNX Int8.
        test_samples: Lista de arrays de entrada, cada um com shape (T, 84) ou (1, T, 84).
        labels: Lista de rótulos inteiros corretos (mesmo comprimento que ``test_samples``).
        max_delta_pp: Diferença máxima permitida em pontos percentuais (padrão 2.0).

    Returns:
        Dicionário com as chaves:
        - ``fp32_top1``: Acurácia Top-1 do modelo Float32 (0.0 a 1.0).
        - ``int8_top1``: Acurácia Top-1 do modelo Int8 (0.0 a 1.0).
        - ``delta_pp``: Diferença absoluta em pontos percentuais.

    Raises:
        AssertionError: Se ``abs(fp32_top1 - int8_top1) * 100 > max_delta_pp``.
    """
    fp32_session = ort.InferenceSession(str(fp32_path))
    int8_session = ort.InferenceSession(str(int8_path))

    fp32_preds = _run_inference(fp32_session, test_samples)
    int8_preds = _run_inference(int8_session, test_samples)

    n = len(labels)
    fp32_top1 = sum(p == g for p, g in zip(fp32_preds, labels)) / n
    int8_top1 = sum(p == g for p, g in zip(int8_preds, labels)) / n

    delta_pp = abs(fp32_top1 - int8_top1) * 100.0

    if delta_pp > max_delta_pp:
        raise AssertionError(
            f"Degradação de acurácia ({delta_pp:.2f}pp) excede o limite de "
            f"{max_delta_pp}pp. fp32_top1={fp32_top1:.4f}, int8_top1={int8_top1:.4f}"
        )

    return {
        "fp32_top1": fp32_top1,
        "int8_top1": int8_top1,
        "delta_pp": delta_pp,
    }
