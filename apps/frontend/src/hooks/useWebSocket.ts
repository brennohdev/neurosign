import { useEffect, useRef, useState } from "react";
import { WsClient } from "../lib/wsClient";
import type { PredictionMessage } from "../types";

/**
 * Hook that manages a persistent WebSocket connection via WsClient.
 * Exposes sendFrame and lastPrediction.
 *
 * Requisitos: 2.2, 2.3
 */
export function useWebSocket(url: string) {
  const clientRef = useRef<WsClient | null>(null);
  const [lastPrediction, setLastPrediction] = useState<PredictionMessage | null>(null);

  useEffect(() => {
    const client = new WsClient(url);
    clientRef.current = client;

    client.onMessage((msg: PredictionMessage) => {
      setLastPrediction(msg);
    });

    return () => {
      client.destroy();
      clientRef.current = null;
    };
  }, [url]);

  function sendFrame(frame: Float32Array): void {
    clientRef.current?.send(frame);
  }

  return { sendFrame, lastPrediction };
}
