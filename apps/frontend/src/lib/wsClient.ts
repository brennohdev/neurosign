import type { PredictionMessage } from "../types";

export type { PredictionMessage };
export type { PredictionItem } from "../types";

/**
 * WebSocket client with persistent connection and exponential backoff reconnection.
 *
 * Requisitos: 2.1, 2.2, 2.3, 2.4
 */
export class WsClient {
  private readonly url: string;
  private ws: WebSocket | null = null;
  private messageHandler: ((msg: PredictionMessage) => void) | null = null;
  private reconnectAttempt = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private destroyed = false;

  constructor(url: string) {
    this.url = url;
    this.connect();
  }

  private connect(): void {
    if (this.destroyed) return;

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectAttempt = 0;
    };

    this.ws.onmessage = (event: MessageEvent) => {
      if (!this.messageHandler) return;
      try {
        const msg = JSON.parse(event.data as string) as PredictionMessage;
        this.messageHandler(msg);
      } catch {
        // ignore malformed messages
      }
    };

    this.ws.onclose = () => {
      if (this.destroyed) return;
      this.scheduleReconnect();
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror, so reconnect is handled there
    };
  }

  private scheduleReconnect(): void {
    const delay = Math.min(Math.pow(2, this.reconnectAttempt) * 1000, 30000);
    this.reconnectAttempt++;
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  /** Serializes the frame as JSON and sends it over the WebSocket. */
  send(frame: Float32Array): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ frame: Array.from(frame) }));
    }
  }

  /** Registers a handler for incoming prediction messages. */
  onMessage(handler: (msg: PredictionMessage) => void): void {
    this.messageHandler = handler;
  }

  /** Closes the connection and stops reconnection attempts. */
  destroy(): void {
    this.destroyed = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
