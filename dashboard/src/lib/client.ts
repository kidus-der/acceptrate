// EventSource client for GET /stats/stream. The browser's own retry is
// disabled in favour of an explicit close + capped exponential backoff, so a
// dead server does not hammer the loopback and the UI can show the state.
import { parseStats } from './stats';
import type { Stats } from './types';

export type ConnectionStatus = 'connecting' | 'open' | 'reconnecting' | 'closed';

export interface StatsHandlers {
  onStats(stats: Stats): void;
  onHeartbeat(stats: Stats): void;
  onStatus(status: ConnectionStatus, detail?: string): void;
  onError(message: string): void;
}

/** The subset of EventSource the client uses, so tests can inject a fake. */
export interface EventSourceLike {
  addEventListener(type: string, listener: (ev: Event) => void): void;
  close(): void;
}

export type EventSourceFactory = (url: string) => EventSourceLike;

export const BACKOFF_BASE_MS = 500;
export const BACKOFF_MAX_MS = 8000;

export function backoffMs(attempt: number): number {
  return Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** attempt);
}

const defaultFactory: EventSourceFactory = (url) => new EventSource(url);

export class StatsClient {
  #source: EventSourceLike | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  #attempt = 0;
  #running = false;

  constructor(
    private readonly url: string,
    private readonly handlers: StatsHandlers,
    private readonly factory: EventSourceFactory = defaultFactory,
  ) {}

  start(): void {
    if (this.#running) {
      return;
    }
    this.#running = true;
    this.#attempt = 0;
    this.#open();
  }

  stop(): void {
    this.#running = false;
    this.#closeSource();
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
    this.handlers.onStatus('closed');
  }

  #open(): void {
    this.handlers.onStatus(this.#attempt === 0 ? 'connecting' : 'reconnecting');
    const source = this.factory(this.url);
    this.#source = source;
    source.addEventListener('open', () => {
      this.#attempt = 0;
      this.handlers.onStatus('open');
    });
    source.addEventListener('stats', (ev) => this.#frame(ev, this.handlers.onStats));
    source.addEventListener('heartbeat', (ev) => this.#frame(ev, this.handlers.onHeartbeat));
    source.addEventListener('error', () => this.#reconnect());
  }

  #frame(ev: Event, deliver: (stats: Stats) => void): void {
    const data = (ev as MessageEvent<unknown>).data;
    try {
      deliver(parseStats(data));
    } catch (err) {
      this.handlers.onError((err as Error).message);
    }
  }

  #reconnect(): void {
    this.#closeSource();
    if (!this.#running) {
      return;
    }
    const delay = backoffMs(this.#attempt);
    this.#attempt += 1;
    this.handlers.onStatus('reconnecting', `retrying in ${delay} ms`);
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#open();
    }, delay);
  }

  #closeSource(): void {
    this.#source?.close();
    this.#source = null;
  }
}
