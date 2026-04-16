"""Exportação do modelo PyTorch para formato ONNX.

Requisitos: 8.1
"""

from __future__ import annotations

from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    window_size: int = 30,
    input_size: int = 84,
    opset: int = 17,
) -> Path:
    """Exporta um modelo PyTorch para ONNX.

    Args:
        model: Modelo PyTorch a ser exportado.
        output_path: Caminho de saída para o arquivo .onnx.
        window_size: Número de frames na janela temporal (padrão 30).
        input_size: Número de features por frame (padrão 84).
        opset: Versão do opset ONNX (padrão 17).

    Returns:
        O caminho ``output_path`` após a exportação.
    """
    model.eval()
    model.cpu()

    dummy_input = torch.zeros(1, window_size, input_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        dynamo=False,
    )

    return output_path
