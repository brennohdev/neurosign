"""Benchmark de latência de inferência ONNX.

Requisitos: 8.4, 4.3
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark_onnx(
    model_path: Path,
    window_size: int = 30,
    n_samples: int = 100,
    input_size: int = 84,
) -> dict:
    """Mede a latência de inferência de um modelo ONNX.

    Gera ``n_samples`` inputs aleatórios e mede o tempo de cada inferência
    individualmente, retornando percentis P50, P95 e média.

    Args:
        model_path: Caminho para o arquivo .onnx.
        window_size: Número de frames na janela temporal (padrão 30).
        n_samples: Número de amostras para o benchmark (padrão 100).

    Returns:
        Dicionário com as chaves:
        - ``p50_ms``: Latência no percentil 50 (mediana), em milissegundos.
        - ``p95_ms``: Latência no percentil 95, em milissegundos.
        - ``mean_ms``: Latência média, em milissegundos.
        - ``n_samples``: Número de amostras utilizadas.
    """
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    latencies_ms: list[float] = []

    for _ in range(n_samples):
        x = np.random.randn(1, window_size, input_size).astype(np.float32)
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "mean_ms": float(np.mean(latencies)),
        "n_samples": n_samples,
    }
