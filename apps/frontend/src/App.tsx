import { useEffect, useRef, useState } from "react";
import { useMediaPipe } from "./hooks/useMediaPipe";
import { useWebSocket } from "./hooks/useWebSocket";
import { VideoFeed } from "./components/VideoFeed";
import { PredictionDisplay } from "./components/PredictionDisplay";
import { Controls } from "./components/Controls";
import type { PredictionItem } from "./types";
import "./App.css";

// Generate a stable session ID for the lifetime of this page load.
const sessionId = crypto.randomUUID();

// Lógica corrigida: Pega a URL base da env (ou usa localhost) e SEMPRE anexa o sessionId
const baseWsUrl = (import.meta.env.VITE_WS_URL as string | undefined) ?? "ws://localhost:8000/ws";
// Garante que não teremos barras duplas na montagem da URL
const wsUrl = baseWsUrl.endsWith('/') ? `${baseWsUrl}${sessionId}` : `${baseWsUrl}/${sessionId}`;

const RECONNECT_TIMEOUT_MS = 5000;

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null); 
  
  const { frame } = useMediaPipe(videoRef, canvasRef); 
  const { sendFrame, lastPrediction } = useWebSocket(wsUrl);

  const [isExpanded, setIsExpanded] = useState(false);
  const [isSpeechEnabled, setIsSpeechEnabled] = useState(false);
  const [isReconnecting, setIsReconnecting] = useState(false);

  // Forward each new frame to the WebSocket.
  useEffect(() => {
    if (frame) {
      sendFrame(frame);
    }
  }, [frame, sendFrame]);

  // Show reconnection indicator when no prediction arrives for > 5s.
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsReconnecting(true);
    }, RECONNECT_TIMEOUT_MS);

    if (lastPrediction !== null) {
      setIsReconnecting(false);
    }

    return () => clearTimeout(timer);
  }, [lastPrediction]);

  const predictions: PredictionItem[] | null =
    lastPrediction?.predictions ?? null;

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">NeuroSign</h1>
        <p className="app__subtitle">Tradutor de Língua de Sinais em tempo real</p>
      </header>

      <main className="app__main">
        <section className="app__video-section" aria-label="Feed da câmera">
          <VideoFeed videoRef={videoRef} canvasRef={canvasRef} />
        </section>

        <section className="app__prediction-section" aria-label="Predições">
          {isReconnecting && (
            <div className="app__reconnecting" role="status" aria-live="polite">
              Aguardando conexão com o servidor...
            </div>
          )}
          <PredictionDisplay
            predictions={predictions}
            isExpanded={isExpanded}
            onToggleExpanded={() => setIsExpanded((v) => !v)}
            isSpeechEnabled={isSpeechEnabled}
            onToggleSpeech={() => setIsSpeechEnabled((v) => !v)}
          />
        </section>
      </main>

      <footer className="app__footer">
        <Controls
          isExpanded={isExpanded}
          onToggleExpanded={() => setIsExpanded((v) => !v)}
          isSpeechEnabled={isSpeechEnabled}
          onToggleSpeech={() => setIsSpeechEnabled((v) => !v)}
          isConnected={!isReconnecting}
        />
      </footer>
    </div>
  );
}