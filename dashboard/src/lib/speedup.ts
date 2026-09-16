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
