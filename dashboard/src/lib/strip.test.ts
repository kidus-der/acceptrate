// Mirrors tui/internal/ui/strip_test.go: the strip reads without colour.
import { describe, expect, test } from 'vitest';

import { GLYPH, cells, tail } from './strip';
import type { WindowSummary } from './types';

const w = (k: number, n: number): WindowSummary => ({
  k_proposed: k,
  n_accepted: n,
  draft_ms: 1,
  verify_ms: 2,
  window_ms: 3,
});

describe('cells', () => {
  test('a partially accepted window is n accepted glyphs then the rejection point', () => {
    expect(cells([w(4, 2)])).toEqual(['accepted', 'accepted', 'rejected']);
  });

  test('a fully accepted window ends in the bonus token', () => {
    expect(cells([w(3, 3)])).toEqual(['accepted', 'accepted', 'accepted', 'bonus']);
  });

  test('a window rejected at the first token is just the rejection point', () => {
    expect(cells([w(4, 0)])).toEqual(['rejected']);
  });

  test('baseline windows (k = 0) add nothing', () => {
    expect(cells([w(0, 0), w(2, 1)])).toEqual(['accepted', 'rejected']);
  });

  test('windows concatenate in order', () => {
    expect(cells([w(2, 2), w(2, 0)])).toEqual(['accepted', 'accepted', 'bonus', 'rejected']);
  });
});

describe('tail', () => {
  test('keeps the last n cells', () => {
    expect(tail(['accepted', 'rejected', 'bonus'], 2)).toEqual(['rejected', 'bonus']);
    expect(tail(['accepted'], 5)).toEqual(['accepted']);
  });
});

describe('GLYPH', () => {
  test('uses the design tokens: ● accepted, ○ rejected, ◆ bonus', () => {
    expect(GLYPH.accepted).toBe('●');
    expect(GLYPH.rejected).toBe('○');
    expect(GLYPH.bonus).toBe('◆');
  });
});
