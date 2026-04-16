"""Quantização dinâmica de modelos ONNX Float32 → Int8.

Requisitos: 8.2

Nota: para modelos LSTM, ``quantize_dynamic`` é mais adequado que
``quantize_static``, pois não requer calibração com dados reais.
"""

from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_model(fp32_path: Path, int8_path: Path) -> Path:
    """Quantiza um modelo ONNX Float32 para Int8 usando quantização dinâmica.

    Args:
        fp32_path: Caminho para o modelo ONNX Float32 de entrada.
        int8_path: Caminho de saída para o modelo ONNX Int8.

    Returns:
        O caminho ``int8_path`` após a quantização.
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    return int8_path
