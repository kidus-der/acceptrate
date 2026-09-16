// Pure uPlot option and data builders. Colours come in as a Palette read from
// the design tokens at render time (canvas cannot read CSS custom properties).
import type uPlot from 'uplot';

import { K_MAX, speedupCurve } from './speedup';

export interface Palette {
  readonly ink: string;
  readonly ink2: string;
  readonly muted: string;
  readonly line: string;
  readonly teal: string;
  readonly amber: string;
  readonly green: string;
  readonly red: string;
  readonly font: string;
}

export type CurveColumns = readonly [number[], number[], number[], (number | null)[]];

const LINE_WIDTH = 2;
const MARKER_SIZE = 11;
const BREAK_EVEN_DASH = [6, 4];

/** [K, speedup(K), 1.0 break-even, current-K marker] for uPlot. */
export function curveColumns(alpha: number, c: number, kCurrent: number, kMax: number = K_MAX): CurveColumns {
  const { ks, values } = speedupCurve(alpha, c, kMax);
  return [
    [...ks],
    [...values],
    ks.map(() => 1),
    ks.map((k, i) => (k === kCurrent ? values[i] : null)),
  ];
}

function axis(p: Palette, extra: Partial<uPlot.Axis> = {}): uPlot.Axis {
  return {
    stroke: p.muted,
    font: p.font,
    grid: { stroke: p.line, width: 1 },
    ticks: { stroke: p.line, width: 1 },
    ...extra,
  };
}

const fmt2 = (v: number | null): string => (v == null ? '' : v.toFixed(2));

/** Speedup vs K: the brief's calculator with the live operating point marked. */
export function curveOptions(p: Palette): Omit<uPlot.Options, 'width' | 'height'> {
  return {
    scales: { x: { time: false }, y: { range: (_u, min, max) => [Math.min(0.5, min), Math.max(2, max)] } },
    axes: [
      axis(p, { label: 'draft depth K', labelFont: p.font, incrs: [1] }),
      axis(p, { label: 'speedup ×', labelFont: p.font, values: (_u, vals) => vals.map(fmt2) }),
    ],
    legend: { show: false },
    cursor: { show: false },
    series: [
      {},
      { label: 'speedup', stroke: p.teal, width: LINE_WIDTH, points: { show: false }, value: (_u, v) => fmt2(v) },
      { label: 'break-even', stroke: p.muted, width: 1, dash: BREAK_EVEN_DASH, points: { show: false } },
      { label: 'K now', stroke: p.amber, width: 0, points: { show: true, size: MARKER_SIZE, fill: p.amber, stroke: p.amber } },
    ],
  };
}

const ALPHA_SCALE = 'alpha';

/** tok/s (left axis) and alpha (right axis, 0..1) over the last N windows. */
export function seriesOptions(p: Palette): Omit<uPlot.Options, 'width' | 'height'> {
  return {
    scales: {
      x: { time: false },
      y: { range: (_u, _min, max) => [0, Math.max(10, max)] },
      [ALPHA_SCALE]: { range: [0, 1] },
    },
    axes: [
      axis(p, { label: 'window #', labelFont: p.font }),
      axis(p, { label: 'tok/s', labelFont: p.font }),
      axis(p, { scale: ALPHA_SCALE, side: 1, label: 'α', labelFont: p.font, grid: { show: false } }),
    ],
    legend: { show: true },
    cursor: { show: false },
    series: [
      { label: 'window' },
      { label: 'tok/s', stroke: p.teal, width: LINE_WIDTH, points: { show: false }, value: (_u, v) => (v == null ? '' : v.toFixed(1)) },
      { label: 'α', scale: ALPHA_SCALE, stroke: p.green, width: LINE_WIDTH, dash: [2, 3], points: { show: false }, value: (_u, v) => fmt2(v) },
    ],
  };
}
