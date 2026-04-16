// Shared types for NeuroSign frontend

export interface PredictionItem {
  label: string;
  confidence: number;
  rank: number;
}

export interface PredictionMessage {
  predictions: PredictionItem[];
  session_id: string;
  timestamp_ms: number;
}
