interface ControlsProps {
  isExpanded: boolean;
  onToggleExpanded: () => void;
  isSpeechEnabled: boolean;
  onToggleSpeech: () => void;
  isConnected: boolean;
}

/**
 * Toolbar with toggle buttons for expanded mode and speech synthesis,
 * plus a WebSocket connection status indicator.
 *
 * Requisitos: 2.3, 2.4, 5.2
 */
export function Controls({
  isExpanded,
  onToggleExpanded,
  isSpeechEnabled,
  onToggleSpeech,
  isConnected,
}: ControlsProps) {
  return (
    <div className="controls">
      <div className="controls__status" aria-live="polite">
        <span
          className={`controls__status-dot ${isConnected ? "controls__status-dot--connected" : "controls__status-dot--disconnected"}`}
          aria-hidden="true"
        />
        <span className="controls__status-label">
          {isConnected ? "Conectado" : "Reconectando..."}
        </span>
      </div>

      <div className="controls__buttons">
        <button
          type="button"
          onClick={onToggleExpanded}
          aria-pressed={isExpanded}
          className="controls__btn"
        >
          {isExpanded ? "Recolher Top-5" : "Expandir Top-5"}
        </button>

        <button
          type="button"
          onClick={onToggleSpeech}
          aria-pressed={isSpeechEnabled}
          className="controls__btn"
        >
          {isSpeechEnabled ? "🔊 Voz ligada" : "🔇 Voz desligada"}
        </button>
      </div>
    </div>
  );
}
