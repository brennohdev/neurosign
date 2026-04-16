import { useEffect, useState, type RefObject } from "react";

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>; // Prop adicionada
}

/**
 * Renders the camera feed and requests getUserMedia permission.
 * Overlay a canvas to draw MediaPipe landmarks.
 */
export function VideoFeed({ videoRef, canvasRef }: VideoFeedProps) {
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [errorMessage, setErrorMessage] = useState<string>("");

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStatus("ready");
      } catch (err) {
        const error = err as Error;
        if (
          error.name === "NotAllowedError" ||
          error.name === "PermissionDeniedError"
        ) {
          setErrorMessage(
            "Acesso à câmera negado. Por favor, permita o acesso à câmera nas configurações do navegador e recarregue a página."
          );
        } else if (
          error.name === "NotFoundError" ||
          error.name === "DevicesNotFoundError"
        ) {
          setErrorMessage(
            "Nenhuma câmera encontrada. Conecte uma câmera e tente novamente."
          );
        } else {
          setErrorMessage(
            `Não foi possível acessar a câmera: ${error.message || "erro desconhecido"}.`
          );
        }
        setStatus("error");
      }
    }

    startCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [videoRef]);

  return (
    <div className="video-feed" style={{ position: "relative" }}>
      {status === "loading" && (
        <div className="video-feed__loading" aria-live="polite">
          <span className="video-feed__spinner" aria-hidden="true" />
          <p>Aguardando câmera...</p>
        </div>
      )}

      {status === "error" && (
        <div className="video-feed__error" role="alert">
          <p>{errorMessage}</p>
        </div>
      )}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="video-feed__video"
        style={{ 
          display: status === "ready" ? "block" : "none",
          transform: "scaleX(-1)", // Efeito espelho para UX
          width: "100%",
          height: "auto"
        }}
        aria-label="Feed da câmera para detecção de sinais"
      />

      {/* Canvas invisível sobreposto ao vídeo para desenhar os pontos */}
      <canvas
        ref={canvasRef}
        className="video-feed__canvas"
        style={{
          display: status === "ready" ? "block" : "none",
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          transform: "scaleX(-1)", // Deve acompanhar o espelhamento do vídeo
          pointerEvents: "none" // Permite clicar "através" do canvas
        }}
      />
    </div>
  );
}