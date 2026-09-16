// Mirrors tui/internal/mock/scenario.go: alpha 0.45 -> 0.80, K 4 -> 6.
import { describe, expect, test } from 'vitest';

import { DEMO_DRAFT, DEMO_MODEL, chunkWords, demoScenario } from './demo';
import { parseStats } from './stats';

describe('chunkWords', () => {
  test('keeps whitespace attached to the preceding word so the join is exact', () => {
    const text = 'one two three\n\nfour  five';
    const chunks = chunkWords(text, 2);
    expect(chunks.join('')).toBe(text);
    expect(chunks[0]).toBe('one two ');
  });
});

describe('demoScenario', () => {
  const scenario = demoScenario();
  const { steps, idle } = scenario;
  const first = steps[0];
  const last = steps[steps.length - 1];

  test('names the real model pair', () => {
    expect(first.model).toBe(DEMO_MODEL);
    expect(first.draft).toBe(DEMO_DRAFT);
    expect(DEMO_MODEL).toBe('mlx-community/Llama-3.1-8B-Instruct-4bit');
  });

  test('alpha climbs 0.45 -> 0.80 and tok/s 18 -> 31', () => {
    expect(first.alpha_ewma).toBeCloseTo(0.45, 12);
    expect(last.alpha_ewma).toBeCloseTo(0.8, 12);
    expect(first.tok_s_recent).toBeCloseTo(18, 12);
    expect(last.tok_s_recent).toBeCloseTo(31, 12);
  });

  test('K steps from 4 to 6 at 40% of the way through, once', () => {
    const ks = steps.map((s) => s.k_current);
    const stepAt = ks.indexOf(6);
    expect(ks.slice(0, stepAt).every((k) => k === 4)).toBe(true);
    expect(ks.slice(stepAt).every((k) => k === 6)).toBe(true);
    expect(stepAt / (steps.length - 1)).toBeGreaterThanOrEqual(0.4);
    expect((stepAt - 1) / (steps.length - 1)).toBeLessThan(0.4);
  });

  test('every step is busy, counts windows, and keeps at most 16 last windows', () => {
    steps.forEach((s, i) => {
      expect(s.busy).toBe(true);
      expect(s.windows_total).toBe(i + 1);
      expect(s.last_windows.length).toBe(Math.min(i + 1, 16));
    });
  });

  test('totals are the running sums of the fabricated windows', () => {
    const [accepted, proposed] = steps.reduce(
      ([a, p], s) => {
        const w = s.last_windows[s.last_windows.length - 1];
        return [a + w.n_accepted, p + w.k_proposed];
      },
      [0, 0],
    );
    expect(last.accepted_total).toBe(accepted);
    expect(last.proposed_total).toBe(proposed);
  });

  test('windows wobble around alpha*k with real rejection points', () => {
    const kinds = new Set(steps.map((s) => s.last_windows.at(-1)!.n_accepted < s.k_current));
    expect(kinds.has(true)).toBe(true);
    for (const s of steps) {
      const w = s.last_windows.at(-1)!;
      expect(w.n_accepted).toBeGreaterThanOrEqual(0);
      expect(w.n_accepted).toBeLessThanOrEqual(w.k_proposed);
      expect(w.window_ms).toBeCloseTo(((w.n_accepted + 1) / s.tok_s_recent!) * 1000, 9);
      expect(w.draft_ms).toBeCloseTo(w.window_ms * 0.15, 9);
      expect(w.verify_ms).toBeCloseTo(w.window_ms * 0.75, 9);
    }
  });

  test('idle is the last step with busy cleared', () => {
    expect(idle).toEqual({ ...last, busy: false });
  });

  test('every step round-trips through the strict parser', () => {
    for (const s of steps) {
      expect(parseStats(JSON.stringify(s))).toEqual(s);
    }
  });
});
