// A line-for-line port of acceptrate.model.speedup (Python is the source of truth):
//
//   speedup(alpha, K, c) = (1 - alpha^(K+1)) / ((1 - alpha) * (K*c + 1))
//
// alpha: per-token acceptance rate; K: draft depth; c: draft cost / target cost.

/** Largest draft depth the optimiser considers. */
export const K_MAX = 10;

/** alpha == 1 is the limit (K+1)/(Kc+1); computed via the geometric sum, not the ratio. */
const ALPHA_CAP = 0.999999;

const BREAK_EVEN_STEP = 0.001;

function validate(alpha: number, k: number, c: number): void {
  if (!(alpha >= 0 && alpha <= 1)) {
    throw new RangeError(`alpha must be in [0, 1], got ${alpha}`);
  }
  if (!Number.isInteger(k) || k < 0) {
    throw new RangeError(`K must be a non-negative integer, got ${k}`);
  }
  if (!(c >= 0)) {
    throw new RangeError(`cost ratio must be non-negative, got ${c}`);
  }
}

/** Expected tokens harvested per round: 1 + alpha + ... + alpha^K. */
export function expectedTokens(alpha: number, k: number): number {
  if (alpha >= ALPHA_CAP) {
    return k + 1;
  }
  return (1 - alpha ** (k + 1)) / (1 - alpha);
}

export function speedup(alpha: number, k: number, c: number): number {
  validate(alpha, k, c);
  if (k === 0) {
    return 1.0;
  }
  return expectedTokens(alpha, k) / (k * c + 1);
}

/** The draft depth that maximises speedup; 0 when no depth beats plain decoding. */
export function bestK(alpha: number, c: number, kMax: number = K_MAX): number {
  validate(alpha, 0, c);
  let best = 0;
  let bestValue = 1.0;
  for (let k = 1; k <= kMax; k++) {
    const value = speedup(alpha, k, c);
    if (value > bestValue) {
      best = k;
      bestValue = value;
    }
  }
  return best;
}

/** Smallest alpha < 1 at which depth K stops losing, or null if it never does. */
export function breakEvenAlpha(k: number, c: number): number | null {
  validate(0, k, c);
  const steps = Math.round(1 / BREAK_EVEN_STEP);
  for (let i = 0; i < steps; i++) {
    const alpha = i * BREAK_EVEN_STEP;
    if (speedup(alpha, k, c) >= 1.0) {
      return alpha;
    }
  }
  return null;
}

export interface Curve {
  readonly ks: readonly number[];
  readonly values: readonly number[];
}

/** The equation evaluated on K = 0..kMax, in uPlot's columnar shape. */
export function speedupCurve(alpha: number, c: number, kMax: number = K_MAX): Curve {
  const ks = Array.from({ length: kMax + 1 }, (_, k) => k);
  return { ks, values: ks.map((k) => speedup(alpha, k, c)) };
}

// --- The measured correction (docs/gates/P4.md) -------------------------------
// On Apple Silicon one verify pass over K+1 tokens is not one decode step.
// v(K) = verify pass / plain step, measured per machine by `acceptrate calibrate`
// and served in /stats as v_by_k. The brief's equation assumes v == 1.

export type VTable = Readonly<Record<number, number>>;

/** Measured on an M4 (16 GB), 8B 4-bit target — the default until /stats supplies one. */
export const M4_V_BY_K: VTable = { 1: 1.0, 2: 1.05, 3: 1.3, 4: 1.57, 5: 1.98, 6: 2.43, 7: 2.48, 8: 2.96 };

/** v(K) from the table, extrapolated linearly from its last two entries beyond it. */
export function vAt(table: VTable, k: number): number {
  if (k in table) {
    return table[k];
  }
  const ks = Object.keys(table)
    .map(Number)
    .sort((a, b) => a - b);
  if (ks.length === 0) {
    return 1.0;
  }
  if (ks.length === 1) {
    return table[ks[0]];
  }
  const [k1, k2] = [ks[ks.length - 2], ks[ks.length - 1]];
  const slope = (table[k2] - table[k1]) / (k2 - k1);
  return Math.max(1.0, table[k2] + slope * (k - k2));
}

/** E[tokens] / (K*cPlain + v): harvest over the measured window cost in plain-step units. */
export function speedupCorrected(alpha: number, k: number, cPlain: number, v: number): number {
  validate(alpha, k, cPlain);
  if (k === 0) {
    return 1.0;
  }
  return expectedTokens(alpha, k) / (k * cPlain + v);
}

export function bestKCorrected(alpha: number, cPlain: number, table: VTable, kMax: number = K_MAX): number {
  validate(alpha, 0, cPlain);
  let best = 0;
  let bestValue = 1.0;
  for (let k = 1; k <= kMax; k++) {
    const value = speedupCorrected(alpha, k, cPlain, vAt(table, k));
    if (value > bestValue) {
      best = k;
      bestValue = value;
    }
  }
  return best;
}

export function speedupCurveCorrected(alpha: number, cPlain: number, table: VTable, kMax: number = K_MAX): Curve {
  const ks = Array.from({ length: kMax + 1 }, (_, k) => k);
  return { ks, values: ks.map((k) => speedupCorrected(alpha, k, cPlain, vAt(table, k))) };
}
