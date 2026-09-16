// Pins the same values as tests/test_speedup.py so the port cannot drift
// from acceptrate.model.speedup.
import { describe, expect, test } from 'vitest';

import { K_MAX, bestK, breakEvenAlpha, speedup, speedupCurve } from './speedup';

describe('speedup', () => {
  test('K=0 is plain decoding', () => {
    expect(speedup(0.7, 0, 0.16)).toBe(1.0);
  });

  test("matches the brief's worked example: 1.71x at K=3, 1.69x at K=4", () => {
    expect(speedup(0.7, 3, 0.16)).toBeCloseTo(1.71, 2);
    expect(speedup(0.7, 4, 0.16)).toBeCloseTo(1.69, 2);
    expect(bestK(0.7, 0.16)).toBe(3);
  });

  test('alpha 0 always loses for any positive K', () => {
    expect(speedup(0.0, 4, 0.16)).toBeCloseTo(1 / (4 * 0.16 + 1), 12);
  });

  test('alpha 1 is the limit (K+1)/(Kc+1)', () => {
    expect(speedup(1.0, 4, 0.16)).toBeCloseTo(5 / (4 * 0.16 + 1), 12);
  });

  test('free drafts never lose', () => {
    for (let k = 1; k <= K_MAX; k++) {
      expect(speedup(0.3, k, 0)).toBeGreaterThanOrEqual(1 - 1e-12);
    }
  });

  test('rejects alpha outside [0, 1] and negative cost', () => {
    expect(() => speedup(-0.1, 4, 0.16)).toThrow(RangeError);
    expect(() => speedup(1.5, 4, 0.16)).toThrow(RangeError);
    expect(() => speedup(0.5, 4, -0.1)).toThrow(RangeError);
    expect(() => speedup(0.5, -1, 0.1)).toThrow(RangeError);
  });
});

describe('bestK', () => {
  test('is the argmax over the grid', () => {
    const [alpha, c] = [0.62, 0.11];
    const k = bestK(alpha, c);
    for (let j = 0; j <= K_MAX; j++) {
      expect(speedup(alpha, k, c)).toBeGreaterThanOrEqual(speedup(alpha, j, c) - 1e-12);
    }
  });

  test('is zero when no depth wins', () => {
    expect(bestK(0.1, 0.5)).toBe(0);
  });
});

describe('breakEvenAlpha', () => {
  test('is where speedup crosses one', () => {
    const alpha = breakEvenAlpha(4, 0.16);
    expect(alpha).not.toBeNull();
    expect(speedup(alpha as number, 4, 0.16)).toBeCloseTo(1.0, 1);
  });

  test('is null when drafting is too expensive', () => {
    expect(breakEvenAlpha(1, 1.0)).toBeNull();
  });
});

describe('speedupCurve', () => {
  test('is the equation evaluated on K = 0..K_MAX', () => {
    const curve = speedupCurve(0.7, 0.16);
    expect(curve.ks).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(curve.values[0]).toBe(1.0);
    expect(curve.values[3]).toBeCloseTo(1.71, 2);
  });
});

describe('corrected form (measured verify cost)', () => {
  test('matches the brief with v = 1 and pins the P4 table default', async () => {
    const { speedupCorrected, bestKCorrected, vAt, M4_V_BY_K, speedup, bestK } = await import('./speedup');
    expect(speedupCorrected(0.7, 4, 0.16, 1.0)).toBeCloseTo(speedup(0.7, 4, 0.16), 12);
    expect(vAt(M4_V_BY_K, 8)).toBe(2.96);
    expect(vAt(M4_V_BY_K, 10)).toBeCloseTo(2.96 + 2 * 0.48, 9);
    expect(bestK(0.85, 0.16)).toBeGreaterThanOrEqual(4);
    expect(bestKCorrected(0.85, 0.16, M4_V_BY_K, 8)).toBe(2);
  });
});
