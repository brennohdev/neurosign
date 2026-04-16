import { useEffect, useRef, useState } from "react";
import type { PredictionItem } from "../types";

interface PredictionDisplayProps {
  predictions: PredictionItem[] | null;
  isExpanded: boolean;
  onToggleExpanded: () => void;
  isSpeechEnabled: boolean;
  onToggleSpeech: () => void;
}

/**
 * Displays the top prediction label and optionally a Top-5 list with confidence bars.
 * Updates with a 100ms debounce and speaks the label via Web Speech API when enabled.
 *
 * Requisitos: 5.2, 5.3, 5.4, 5.5
 */
export function PredictionDisplay({
  predictions,
  isExpanded,
  onToggleExpanded,
  isSpeechEnabled,
  onToggleSpeech,
}: PredictionDisplayProps) {
  const [displayedPredictions, setDisplayedPredictions] = useState<
    PredictionItem[] | null
  >(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSpokenRef = useRef<string>("");

  useEffect(() => {
    if (debounceRef.current !== null) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      setDisplayedPredictions(predictions);

      if (
        isSpeechEnabled &&
        predictions &&
        predictions.length > 0 &&
        predictions[0].label !== lastSpokenRef.current
      ) {
        const label = predictions[0].label;
        lastSpokenRef.current = label;
        const utterance = new SpeechSynthesisUtterance(label);
        window.speechSynthesis.speak(utterance);
      }

      debounceRef.current = null;
    }, 100);

    return () => {
      if (debounceRef.current !== null) {
        clearTimeout(debounceRef.current);
        debounceRef.current = null;
      }
    };
  }, [predictions, isSpeechEnabled]);

  const topLabel =
    displayedPredictions && displayedPredictions.length > 0
      ? displayedPredictions[0].label
      : "Aguardando sinal...";

  return (
    <div className="prediction-display">
      <div className="prediction-display__main">
        <span className="prediction-display__label">{topLabel}</span>
      </div>

      {isExpanded && displayedPredictions && displayedPredictions.length > 0 && (
        <ul className="prediction-display__list" aria-label="Top 5 predições">
          {displayedPredictions.slice(0, 5).map((item) => (
            <li key={item.rank} className="prediction-display__item">
              <span className="prediction-display__item-label">{item.label}</span>
              <div
                className="prediction-display__bar-track"
                role="progressbar"
                aria-valuenow={Math.round(item.confidence * 100)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label={`${item.label}: ${Math.round(item.confidence * 100)}%`}
              >
                <div
                  className="prediction-display__bar-fill"
                  style={{ width: `${item.confidence * 100}%` }}
                />
              </div>
              <span className="prediction-display__item-confidence">
                {Math.round(item.confidence * 100)}%
              </span>
            </li>
          ))}
        </ul>
      )}

      <div className="prediction-display__controls">
        <button
          type="button"
          onClick={onToggleExpanded}
          aria-pressed={isExpanded}
          className="prediction-display__btn"
        >
          {isExpanded ? "Recolher" : "Expandir Top-5"}
        </button>
        <button
          type="button"
          onClick={onToggleSpeech}
          aria-pressed={isSpeechEnabled}
          className="prediction-display__btn"
        >
          {isSpeechEnabled ? "Voz: ligada" : "Voz: desligada"}
        </button>
      </div>
    </div>
  );
}
