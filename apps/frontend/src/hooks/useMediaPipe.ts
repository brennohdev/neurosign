// src/hooks/useMediaPipe.ts
import { useEffect, useRef, useState, type RefObject } from "react";
import { Holistic, type Results } from "@mediapipe/holistic";
import { normalize } from "../lib/normalizer";

// Layout: [hand0 (84)] + [pose (66)] = 150 features
const NUM_HAND_LANDMARKS = 21;
const NUM_HANDS = 2;
const NUM_POSE_LANDMARKS = 33;
const HAND_FEATURES = NUM_HANDS * NUM_HAND_LANDMARKS * 2; // 84
const POSE_FEATURES = NUM_POSE_LANDMARKS * 2;              // 66
export const FRAME_SIZE = HAND_FEATURES + POSE_FEATURES;   // 150

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

// Conexões de pose relevantes (upper body)
const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
];

function extractFrame(results: Results): Float32Array {
  const raw = new Float32Array(FRAME_SIZE);

  // Mãos: left_hand_landmarks = mão 0, right_hand_landmarks = mão 1
  const hands = [results.leftHandLandmarks, results.rightHandLandmarks];
  for (let h = 0; h < NUM_HANDS; h++) {
    const landmarks = hands[h];
    if (!landmarks) continue;
    const offset = h * NUM_HAND_LANDMARKS * 2;
    for (let i = 0; i < NUM_HAND_LANDMARKS; i++) {
      raw[offset + i * 2]     = landmarks[i].x;
      raw[offset + i * 2 + 1] = landmarks[i].y;
    }
  }

  // Pose corporal
  if (results.poseLandmarks) {
    const offset = HAND_FEATURES;
    for (let i = 0; i < NUM_POSE_LANDMARKS; i++) {
      raw[offset + i * 2]     = results.poseLandmarks[i].x;
      raw[offset + i * 2 + 1] = results.poseLandmarks[i].y;
    }
  }

  return raw;
}

/**
 * Hook que inicializa MediaPipe Holistic (mãos + pose) e processa frames a ≥ 15 FPS.
 * Emite Float32Array[150] normalizado via `frame` state.
 * Desenha esqueleto de mãos e pose no canvas se fornecido.
 */
export function useMediaPipe(
  videoRef: RefObject<HTMLVideoElement>,
  canvasRef?: RefObject<HTMLCanvasElement>,
) {
  const [frame, setFrame] = useState<Float32Array | null>(null);
  const holisticRef = useRef<Holistic | null>(null);
  const rafRef = useRef<number | null>(null);
  const activeRef = useRef(true);

  useEffect(() => {
    activeRef.current = true;

    const holistic = new Holistic({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6,
    });

    holistic.onResults((results: Results) => {
      if (!activeRef.current) return;

      // Extrai e normaliza landmarks
      const raw = extractFrame(results);
      // Normaliza apenas a parte das mãos (primeiros 84 valores)
      const handsRaw = raw.slice(0, HAND_FEATURES) as Float32Array;
      const handsNorm = normalize(handsRaw);
      const combined = new Float32Array(FRAME_SIZE);
      combined.set(handsNorm, 0);
      combined.set(raw.slice(HAND_FEATURES), HAND_FEATURES);
      setFrame(combined);

      // Desenha esqueleto no canvas
      if (canvasRef?.current && videoRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        const video = videoRef.current;
        if (!ctx) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Desenha mãos
        const hands = [results.leftHandLandmarks, results.rightHandLandmarks];
        for (const landmarks of hands) {
          if (!landmarks) continue;
          ctx.lineWidth = 3;
          ctx.strokeStyle = "#FFFFFF";
          ctx.lineCap = "round";
          for (const [a, b] of HAND_CONNECTIONS) {
            if (!landmarks[a] || !landmarks[b]) continue;
            ctx.beginPath();
            ctx.moveTo(landmarks[a].x * canvas.width, landmarks[a].y * canvas.height);
            ctx.lineTo(landmarks[b].x * canvas.width, landmarks[b].y * canvas.height);
            ctx.stroke();
          }
          for (const lm of landmarks) {
            ctx.beginPath();
            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "#FF1493";
            ctx.fill();
          }
        }

        // Desenha pose (upper body)
        if (results.poseLandmarks) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#00BFFF";
          for (const [a, b] of POSE_CONNECTIONS) {
            const lmA = results.poseLandmarks[a];
            const lmB = results.poseLandmarks[b];
            if (!lmA || !lmB) continue;
            ctx.beginPath();
            ctx.moveTo(lmA.x * canvas.width, lmA.y * canvas.height);
            ctx.lineTo(lmB.x * canvas.width, lmB.y * canvas.height);
            ctx.stroke();
          }
          for (const [a, b] of POSE_CONNECTIONS) {
            for (const idx of [a, b]) {
              const lm = results.poseLandmarks[idx];
              if (!lm) continue;
              ctx.beginPath();
              ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
              ctx.fillStyle = "#00BFFF";
              ctx.fill();
            }
          }
        }
      }
    });

    holisticRef.current = holistic;

    async function processFrame() {
      if (!activeRef.current) return;
      const video = videoRef.current;
      if (video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        await holisticRef.current?.send({ image: video });
      }
      rafRef.current = requestAnimationFrame(processFrame);
    }

    rafRef.current = requestAnimationFrame(processFrame);

    return () => {
      activeRef.current = false;
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      holisticRef.current?.close();
      holisticRef.current = null;
    };
  }, [videoRef, canvasRef]);

  return { frame };
}
