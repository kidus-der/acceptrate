// The payload shape is the one tests/test_serve_app.py asserts on /stats and
// the `event: stats` frames of /stats/stream.
import { describe, expect, test } from 'vitest';

import { M4_V_BY_K } from './speedup';

import { StatsParseError, costRatio, parseStats } from './stats';
import type { WindowSummary } from './types';

const WINDOW = { k_proposed: 3, n_accepted: 2, draft_ms: 1.5, verify_ms: 4.0, window_ms: 6.2 };

const PAYLOAD = {
  model: 'target-fake',
  draft: 'draft-fake',
  busy: false,
  k_current: 3,
  alpha_ewma: 0.0,
  tok_s_recent: 41.7,
  windows_total: 9,
  accepted_total: 0,
  proposed_total: 27,
  last_windows: [WINDOW],
  v_by_k: M4_V_BY_K,
};

describe('parseStats', () => {
  test('accepts the exact /stats payload as a JSON string', () => {
    const stats = parseStats(JSON.stringify(PAYLOAD));
    expect(stats).toEqual(PAYLOAD);
  });

  test('accepts an already-decoded object', () => {
    expect(parseStats(PAYLOAD)).toEqual(PAYLOAD);
  });

  test('keeps the nullable fields null: idle server before any window', () => {
    const idle = { ...PAYLOAD, draft: null, alpha_ewma: null, tok_s_recent: null, last_windows: [] };
    const stats = parseStats(idle);
    expect(stats.draft).toBeNull();
    expect(stats.alpha_ewma).toBeNull();
    expect(stats.tok_s_recent).toBeNull();
    expect(stats.last_windows).toEqual([]);
  });

  test.each([
    ['not json', '{"model":'],
    ['missing field', { ...PAYLOAD, k_current: undefined }],
    ['wrong type', { ...PAYLOAD, busy: 'yes' }],
    ['bad window', { ...PAYLOAD, last_windows: [{ k_proposed: 'three' }] }],
    ['array root', []],
    ['null root', null],
  ])('rejects %s with StatsParseError', (_name, bad) => {
    expect(() => parseStats(bad)).toThrow(StatsParseError);
  });
});

describe('costRatio', () => {
  const w = (k: number, draft: number, verify: number): WindowSummary => ({
    k_proposed: k,
    n_accepted: 0,
    draft_ms: draft,
    verify_ms: verify,
    window_ms: draft + verify,
  });

  test('is the median over windows of per-drafted-token cost over verify cost', () => {
    // per-token costs: (1.2/4)/2 = 0.15, (2/4)/2.5 = 0.2, (0.3/3)/1 = 0.1 -> median 0.15
    expect(costRatio([w(4, 1.2, 2), w(4, 2, 2.5), w(3, 0.3, 1)])).toBeCloseTo(0.15, 12);
  });

  test('averages the middle pair for an even count', () => {
    expect(costRatio([w(4, 1.2, 2), w(4, 2, 2.5)])).toBeCloseTo(0.175, 12);
  });

  test('ignores baseline windows (k = 0) and windows without a verify time', () => {
    expect(costRatio([w(0, 0, 3), w(4, 1.2, 2), w(4, 1, 0)])).toBeCloseTo(0.15, 12);
  });

  test('is null when nothing is measurable yet', () => {
    expect(costRatio([])).toBeNull();
    expect(costRatio([w(0, 0, 3)])).toBeNull();
  });
});
