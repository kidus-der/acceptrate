// The boundary with `acceptrate serve`: validate every stats frame before it
// touches the store, and derive the one number the server does not send (c).
import type { Stats, WindowSummary } from './types';

export class StatsParseError extends Error {
  constructor(message: string) {
    super(`stats payload: ${message}`);
    this.name = 'StatsParseError';
  }
}

type Json = Record<string, unknown>;

function asRecord(value: unknown, what: string): Json {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new StatsParseError(`${what} must be an object`);
  }
  return value as Json;
}

function num(obj: Json, key: string): number {
  const v = obj[key];
  if (typeof v !== 'number' || !Number.isFinite(v)) {
    throw new StatsParseError(`${key} must be a finite number`);
  }
  return v;
}

function nullableNum(obj: Json, key: string): number | null {
  return obj[key] === null ? null : num(obj, key);
}

function str(obj: Json, key: string): string {
  const v = obj[key];
  if (typeof v !== 'string') {
    throw new StatsParseError(`${key} must be a string`);
  }
  return v;
}

function bool(obj: Json, key: string): boolean {
  const v = obj[key];
  if (typeof v !== 'boolean') {
    throw new StatsParseError(`${key} must be a boolean`);
  }
  return v;
}

function parseWindow(value: unknown): WindowSummary {
  const w = asRecord(value, 'window');
  return {
    k_proposed: num(w, 'k_proposed'),
    n_accepted: num(w, 'n_accepted'),
    draft_ms: num(w, 'draft_ms'),
    verify_ms: num(w, 'verify_ms'),
    window_ms: num(w, 'window_ms'),
  };
}

function decode(input: unknown): unknown {
  if (typeof input !== 'string') {
    return input;
  }
  try {
    return JSON.parse(input);
  } catch (err) {
    throw new StatsParseError(`invalid JSON (${(err as Error).message})`);
  }
}

/** Validate a /stats body or an SSE `data:` payload; throws StatsParseError on any drift. */
export function parseStats(input: unknown): Stats {
  const obj = asRecord(decode(input), 'stats');
  const windows = obj['last_windows'];
  if (!Array.isArray(windows)) {
    throw new StatsParseError('last_windows must be an array');
  }
  return {
    model: str(obj, 'model'),
    draft: obj['draft'] === null ? null : str(obj, 'draft'),
    busy: bool(obj, 'busy'),
    k_current: num(obj, 'k_current'),
    alpha_ewma: nullableNum(obj, 'alpha_ewma'),
    tok_s_recent: nullableNum(obj, 'tok_s_recent'),
    windows_total: num(obj, 'windows_total'),
    accepted_total: num(obj, 'accepted_total'),
    proposed_total: num(obj, 'proposed_total'),
    last_windows: windows.map(parseWindow),
  };
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Measured draft cost ratio c: median over speculative windows of
 * (draft_ms / k_proposed) / verify_ms. Null until a window has both times.
 */
export function costRatio(windows: readonly WindowSummary[]): number | null {
  const ratios = windows
    .filter((w) => w.k_proposed > 0 && w.verify_ms > 0)
    .map((w) => w.draft_ms / w.k_proposed / w.verify_ms);
  return ratios.length === 0 ? null : median(ratios);
}
