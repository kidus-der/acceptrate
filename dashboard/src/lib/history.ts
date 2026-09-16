// The streaming time-series: one sample per stats frame, x = windows_total so
// the axis is "window #", which stays ascending and unique as uPlot requires.

export interface Sample {
  /** The server's windows_total when the frame arrived. */
  readonly t: number;
  readonly tok_s: number | null;
  readonly alpha: number | null;
}

export const HISTORY_MAX = 120;

/** A new history with `sample` appended and the oldest dropped past `max`; unchanged if t does not advance. */
export function pushSample(
  history: readonly Sample[],
  sample: Sample,
  max: number = HISTORY_MAX,
): readonly Sample[] {
  const last = history[history.length - 1];
  if (last !== undefined && sample.t <= last.t) {
    return history;
  }
  const next = [...history, sample];
  return next.length > max ? next.slice(next.length - max) : next;
}

export type Columns = readonly [number[], (number | null)[], (number | null)[]];

/** uPlot's columnar data: [t, tok/s, alpha]. */
export function toColumns(history: readonly Sample[]): Columns {
  return [history.map((s) => s.t), history.map((s) => s.tok_s), history.map((s) => s.alpha)];
}
