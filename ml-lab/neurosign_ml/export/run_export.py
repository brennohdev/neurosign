"""Script de exportação completo: checkpoint → ONNX → Int8 → benchmark.

Uso:
    uv run python -m neurosign_ml.export.run_export \
        --checkpoint checkpoints_holistic/checkpoint_epoch0074.pt \
        --labels     data/processed/landmarks_holistic/labels.json \
        --output-dir models/ \
        --input-size 150 \
        --hidden-size 256
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from neurosign_ml.models.bilstm_attention import BiLSTMAttention
from neurosign_ml.export.export_onnx import export_to_onnx
from neurosign_ml.export.quantize import quantize_model
from neurosign_ml.export.benchmark import benchmark_onnx

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--input-size", type=int, default=150)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=30)
    args = parser.parse_args()

    with open(args.labels) as f:
        label_map: list[str] = json.load(f)
    num_classes = len(label_map)
    logger.info("Classes: %d | Input size: %d", num_classes, args.input_size)

    # Carrega checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    logger.info("Checkpoint: época %d | val_top1=%.3f",
                ckpt.get("epoch", "?"), ckpt.get("val_top1", 0))

    # Reconstrói modelo
    model = BiLSTMAttention(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Modelo carregado: %d parâmetros", sum(p.numel() for p in model.parameters()))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Exporta para ONNX Float32
    fp32_path = args.output_dir / "neurosign_fp32.onnx"
    export_to_onnx(model, fp32_path, window_size=args.window_size, input_size=args.input_size)
    fp32_size_mb = fp32_path.stat().st_size / 1024 / 1024
    logger.info("ONNX Float32: %s (%.1f MB)", fp32_path, fp32_size_mb)

    # Quantiza para Int8
    int8_path = args.output_dir / "neurosign_int8.onnx"
    quantize_model(fp32_path, int8_path)
    int8_size_mb = int8_path.stat().st_size / 1024 / 1024
    logger.info("ONNX Int8: %s (%.1f MB)", int8_path, int8_size_mb)

    # Benchmark
    logger.info("Executando benchmark (100 amostras)...")
    results = benchmark_onnx(int8_path, window_size=args.window_size, input_size=args.input_size)
    logger.info("Latência Int8 — P50: %.1fms | P95: %.1fms | Média: %.1fms",
                results["p50_ms"], results["p95_ms"], results["mean_ms"])

    # Salva labels.json junto ao modelo para o backend
    import shutil
    shutil.copy(args.labels, args.output_dir / "labels.json")
    logger.info("labels.json copiado para %s", args.output_dir)

    logger.info("Exportação concluída. Modelos em: %s", args.output_dir)


if __name__ == "__main__":
    main()
