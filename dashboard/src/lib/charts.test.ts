import { describe, expect, test } from 'vitest';

import { curveColumns } from './charts';
import { M4_V_BY_K } from './speedup';

/** v = 1 everywhere reproduces the brief's closed form (1.71x at K = 3 for alpha 0.7, c 0.16). */
const ONES = Object.fromEntries(Array.from({ length: 10 }, (_, i) => [i + 1, 1.0]));

describe('curveColumns', () => {
  test('is [K, speedup, break-even, marker] with the marker only at the current K', () => {
    const [ks, values, breakEven, marker] = curveColumns(0.7, 0.16, 3, ONES, 10);
    expect(ks).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(values[3]).toBeCloseTo(1.71, 2);
    expect(breakEven.every((v) => v === 1)).toBe(true);
    expect(marker.filter((v) => v !== null)).toHaveLength(1);
    expect(marker[3]).toBeCloseTo(1.71, 2);
  });

  test('has no marker when the current K is off the grid', () => {
    const [, , , marker] = curveColumns(0.7, 0.16, 42, M4_V_BY_K, 10);
    expect(marker.every((v) => v === null)).toBe(true);
  });
});
