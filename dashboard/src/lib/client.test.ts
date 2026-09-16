// The SSE client against a fake EventSource: frames, heartbeats, backoff.
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

import { BACKOFF_BASE_MS, BACKOFF_MAX_MS, StatsClient, backoffMs } from './client';
import type { EventSourceLike, StatsHandlers } from './client';

const PAYLOAD = {
  model: 'target-fake',
  draft: 'draft-fake',
  busy: false,
  k_current: 3,
  alpha_ewma: null,
  tok_s_recent: null,
  windows_total: 0,
  accepted_total: 0,
  proposed_total: 0,
  last_windows: [],
};

class FakeSource implements EventSourceLike {
  readonly listeners = new Map<string, Array<(ev: Event) => void>>();
  closed = false;
  constructor(readonly url: string) {}
  addEventListener(type: string, listener: (ev: Event) => void): void {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener]);
  }
  close(): void {
    this.closed = true;
  }
  emit(type: string, data?: string): void {
    const ev = data === undefined ? new Event(type) : new MessageEvent(type, { data });
    for (const fn of this.listeners.get(type) ?? []) fn(ev);
  }
}

function harness() {
  const sources: FakeSource[] = [];
  const handlers: StatsHandlers = {
    onStats: vi.fn(),
    onHeartbeat: vi.fn(),
    onStatus: vi.fn(),
    onError: vi.fn(),
  };
  const client = new StatsClient('/stats/stream', handlers, (url) => {
    const s = new FakeSource(url);
    sources.push(s);
    return s;
  });
  return { sources, handlers, client };
}

describe('backoffMs', () => {
  test('doubles from the base and caps', () => {
    expect(backoffMs(0)).toBe(BACKOFF_BASE_MS);
    expect(backoffMs(1)).toBe(BACKOFF_BASE_MS * 2);
    expect(backoffMs(20)).toBe(BACKOFF_MAX_MS);
  });
});

describe('StatsClient', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  test('opens the stream URL and reports connecting then open', () => {
    const { sources, handlers, client } = harness();
    client.start();
    expect(sources).toHaveLength(1);
    expect(sources[0].url).toBe('/stats/stream');
    expect(handlers.onStatus).toHaveBeenLastCalledWith('connecting');
    sources[0].emit('open');
    expect(handlers.onStatus).toHaveBeenLastCalledWith('open');
  });

  test('routes stats and heartbeat frames through the parser', () => {
    const { sources, handlers, client } = harness();
    client.start();
    sources[0].emit('stats', JSON.stringify(PAYLOAD));
    sources[0].emit('heartbeat', JSON.stringify({ ...PAYLOAD, busy: true }));
    expect(handlers.onStats).toHaveBeenCalledWith(PAYLOAD);
    expect(handlers.onHeartbeat).toHaveBeenCalledWith({ ...PAYLOAD, busy: true });
  });

  test('a malformed frame is reported and does not drop the connection', () => {
    const { sources, handlers, client } = harness();
    client.start();
    sources[0].emit('stats', '{"model":');
    expect(handlers.onError).toHaveBeenCalledTimes(1);
    expect(handlers.onStats).not.toHaveBeenCalled();
    expect(sources[0].closed).toBe(false);
  });

  test('on error it closes the source and reconnects with exponential backoff', () => {
    const { sources, handlers, client } = harness();
    client.start();
    sources[0].emit('error');
    expect(sources[0].closed).toBe(true);
    expect(handlers.onStatus).toHaveBeenLastCalledWith('reconnecting', expect.any(String));
    vi.advanceTimersByTime(BACKOFF_BASE_MS - 1);
    expect(sources).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sources).toHaveLength(2);
    sources[1].emit('error');
    vi.advanceTimersByTime(BACKOFF_BASE_MS * 2 - 1);
    expect(sources).toHaveLength(2);
    vi.advanceTimersByTime(1);
    expect(sources).toHaveLength(3);
  });

  test('a successful open resets the backoff', () => {
    const { sources, client } = harness();
    client.start();
    sources[0].emit('error');
    vi.advanceTimersByTime(BACKOFF_BASE_MS);
    sources[1].emit('open');
    sources[1].emit('error');
    vi.advanceTimersByTime(BACKOFF_BASE_MS);
    expect(sources).toHaveLength(3);
  });

  test('stop closes the source, cancels the pending reconnect and reports closed', () => {
    const { sources, handlers, client } = harness();
    client.start();
    sources[0].emit('error');
    client.stop();
    expect(handlers.onStatus).toHaveBeenLastCalledWith('closed');
    vi.advanceTimersByTime(BACKOFF_MAX_MS * 2);
    expect(sources).toHaveLength(1);
    client.start();
    client.stop();
    expect(sources[1].closed).toBe(true);
  });

  test('start is idempotent while running', () => {
    const { sources, client } = harness();
    client.start();
    client.start();
    expect(sources).toHaveLength(1);
  });
});
