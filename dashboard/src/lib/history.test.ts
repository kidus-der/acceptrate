import { describe, expect, test } from 'vitest';

import { HISTORY_MAX, pushSample, toColumns } from './history';
import type { Sample } from './history';

const s = (t: number, tokS: number | null = 20, alpha: number | null = 0.5): Sample => ({
  t,
  tok_s: tokS,
  alpha,
});

describe('pushSample', () => {
  test('returns a new array and leaves the original untouched', () => {
    const before = [s(1)];
    const after = pushSample(before, s(2));
    expect(before).toEqual([s(1)]);
    expect(after).toEqual([s(1), s(2)]);
  });

  test('drops the oldest sample past the cap', () => {
    const full = Array.from({ length: HISTORY_MAX }, (_, i) => s(i));
    const after = pushSample(full, s(HISTORY_MAX));
    expect(after).toHaveLength(HISTORY_MAX);
    expect(after[0].t).toBe(1);
    expect(after.at(-1)?.t).toBe(HISTORY_MAX);
  });

  test('honours a custom cap', () => {
    expect(pushSample([s(1), s(2)], s(3), 2)).toEqual([s(2), s(3)]);
  });

  test('ignores a sample whose t does not advance (heartbeats, replays)', () => {
    const history = [s(5)];
    expect(pushSample(history, s(5))).toBe(history);
    expect(pushSample(history, s(4))).toBe(history);
  });
});

describe('toColumns', () => {
  test('is uPlot columnar: [t[], tok_s[], alpha[]] with nulls kept as gaps', () => {
    expect(toColumns([s(1, 20, 0.5), s(2, null, null)])).toEqual([
      [1, 2],
      [20, null],
      [0.5, null],
    ]);
  });
});
