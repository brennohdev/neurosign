"""Extração paralela de landmarks com MediaPipe Holistic (mãos + corpo + face).

Holistic extrai 543 landmarks por frame:
  - 21 landmarks por mão × 2 mãos = 42 pontos × 2 coords = 84 features
  - 33 landmarks de pose (corpo) × 2 coords = 66 features
  - 468 landmarks de face × 2 coords = 936 features (opcional, desativado por padrão)

Com pose=True e face=False: 84 + 66 = 150 features por frame.
Com pose=True e face=True:  84 + 66 + 936 = 1086 features por frame.

Uso:
    uv run python -m neurosign_ml.data.extract_landmarks \
        --annotations data/raw/nslt_300.json \
        --wlasl      data/raw/WLASL_v0.3.json \
        --videos-dir data/raw/videos \
        --output-dir data/processed/landmarks_holistic \
        --num-classes 300 \
        --workers 8 \
        --use-pose \
        --no-face
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp_proc
from pathlib import Path

import cv2
import numpy as np

from neurosign_ml.data.normalize import normalize_landmarks

logger = logging.getLogger(__name__)

NUM_HANDS = 2
NUM_HAND_LANDMARKS = 21
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468

HAND_FEATURES = NUM_HANDS * NUM_HAND_LANDMARKS * 2   # 84
POSE_FEATURES = NUM_POSE_LANDMARKS * 2                # 66
FACE_FEATURES = NUM_FACE_LANDMARKS * 2                # 936


def _extract_holistic_frame(results, use_pose: bool, use_face: bool) -> np.ndarray:
    """Extrai features de um frame Holistic em array flat float32."""
    parts = []

    # Mãos (84 features) — sempre incluídas
    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        hand = np.zeros(NUM_HAND_LANDMARKS * 2, dtype=np.float32)
        if hand_landmarks:
            for i, lm in enumerate(hand_landmarks.landmark):
                hand[i * 2] = lm.x
                hand[i * 2 + 1] = lm.y
        parts.append(hand)

    # Pose corporal (66 features)
    if use_pose:
        pose = np.zeros(POSE_FEATURES, dtype=np.float32)
        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                pose[i * 2] = lm.x
                pose[i * 2 + 1] = lm.y
        parts.append(pose)

    # Face (936 features — opcional)
    if use_face:
        face = np.zeros(FACE_FEATURES, dtype=np.float32)
        if results.face_landmarks:
            for i, lm in enumerate(results.face_landmarks.landmark):
                face[i * 2] = lm.x
                face[i * 2 + 1] = lm.y
        parts.append(face)

    return np.concatenate(parts)


def _process_one(args: tuple) -> tuple[str, bool]:
    """Processa um único vídeo num worker separado."""
    video_id, video_path, frame_start, frame_end, out_path, use_pose, use_face = args

    import mediapipe as mp

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return video_id, False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end = total_frames if frame_end == -1 else min(frame_end, total_frames)
        start = max(0, frame_start - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames: list[np.ndarray] = []

        with mp.solutions.holistic.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.5,
        ) as holistic:
            for _ in range(end - start):
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                raw_hands = _extract_holistic_frame(results, use_pose=False, use_face=False)
                normalized_hands = normalize_landmarks(raw_hands)

                if use_pose or use_face:
                    extras = _extract_holistic_frame(
                        results,
                        use_pose=use_pose,
                        use_face=use_face,
                    )[HAND_FEATURES:]  # descarta mãos (já normalizadas)
                    frame_vec = np.concatenate([normalized_hands, extras])
                else:
                    frame_vec = normalized_hands

                frames.append(frame_vec)

        cap.release()

        if not frames:
            feature_size = HAND_FEATURES
            if use_pose:
                feature_size += POSE_FEATURES
            if use_face:
                feature_size += FACE_FEATURES
            landmarks = np.zeros((1, feature_size), dtype=np.float32)
        else:
            landmarks = np.stack(frames, axis=0)

        np.save(str(out_path), landmarks)
        return video_id, True

    except Exception as e:
        logger.debug("Erro em %s: %s", video_id, e)
        return video_id, False


def build_gloss_map(wlasl_path: Path) -> dict[int, str]:
    with open(wlasl_path) as f:
        wlasl = json.load(f)
    return {i: entry["gloss"] for i, entry in enumerate(wlasl)}


def run(
    annotations_path: Path,
    wlasl_path: Path,
    videos_dir: Path,
    output_dir: Path,
    num_classes: int,
    workers: int,
    use_pose: bool,
    use_face: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    feature_size = HAND_FEATURES
    if use_pose:
        feature_size += POSE_FEATURES
    if use_face:
        feature_size += FACE_FEATURES
    logger.info("Features por frame: %d (mãos=84, pose=%s, face=%s)",
                feature_size, use_pose, use_face)

    with open(annotations_path) as f:
        annotations: dict = json.load(f)

    gloss_map = build_gloss_map(wlasl_path)

    missing_path = videos_dir.parent / "missing.txt"
    missing: set[str] = set()
    if missing_path.exists():
        missing = {line.strip() for line in missing_path.read_text().splitlines() if line.strip()}

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple] = []
    meta_map: dict[str, dict] = {}

    for video_id, meta in annotations.items():
        subset = meta["subset"]
        action_idx, frame_start, frame_end = meta["action"]

        if action_idx >= num_classes:
            continue

        gloss = gloss_map.get(action_idx)
        if gloss is None:
            continue

        video_filename = f"{int(video_id):05d}.mp4"
        video_path = videos_dir / video_filename

        if video_id in missing or not video_path.exists():
            continue

        gloss_dir = output_dir / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)
        out_path = gloss_dir / f"{video_id}.npy"

        meta_map[video_id] = {"gloss": gloss, "subset": subset}

        if out_path.exists():
            continue

        tasks.append((video_id, video_path, frame_start, frame_end, out_path, use_pose, use_face))

    already_done = len(meta_map) - len(tasks)
    logger.info("A processar: %d | Retomada: %d | Workers: %d", len(tasks), already_done, workers)

    processed = already_done
    failed = 0

    with mp_proc.Pool(processes=workers) as pool:
        for i, (video_id, success) in enumerate(
            pool.imap_unordered(_process_one, tasks, chunksize=4), 1
        ):
            if success:
                processed += 1
            else:
                failed += 1
            if i % 200 == 0:
                logger.info("Progresso: %d/%d | Falhas: %d", i, len(tasks), failed)

    logger.info("Concluído. Processados: %d | Falhas: %d", processed, failed)

    manifest: list[dict] = []
    for video_id, info in meta_map.items():
        out_path = output_dir / info["gloss"] / f"{video_id}.npy"
        if out_path.exists():
            manifest.append({
                "video_id": video_id,
                "gloss": info["gloss"],
                "subset": info["subset"],
                "landmarks_path": str(out_path.relative_to(output_dir.parent)),
            })

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    labels = sorted({m["gloss"] for m in manifest})
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    logger.info("manifest.json: %d samples | labels.json: %d classes | features/frame: %d",
                len(manifest), len(labels), feature_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--wlasl", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=300)
    parser.add_argument("--workers", type=int, default=mp_proc.cpu_count())
    parser.add_argument("--use-pose", action="store_true", default=True)
    parser.add_argument("--no-face", action="store_false", dest="use_face")
    parser.set_defaults(use_face=False)
    args = parser.parse_args()
    run(
        args.annotations, args.wlasl, args.videos_dir, args.output_dir,
        args.num_classes, args.workers, args.use_pose, args.use_face,
    )


if __name__ == "__main__":
    main()
