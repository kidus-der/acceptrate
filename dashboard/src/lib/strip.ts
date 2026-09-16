// The window strip: recent windows expanded into tokens, one glyph each.
// Mirrors tui/internal/ui/strip.go so both clients read the same way.
import type { WindowSummary } from './types';

export type CellKind = 'accepted' | 'rejected' | 'bonus';

/** The design tokens' glyphs (design/tokens.json → --glyph-*), so cells read without colour. */
export const GLYPH: Readonly<Record<CellKind, string>> = {
  accepted: '●',
  rejected: '○',
  bonus: '◆',
};

/**
 * Expand windows into tokens: n accepted glyphs, then either the rejection
 * point or the bonus token. Tokens drafted past the rejection were discarded
 * and are not shown. Baseline windows (k = 0) add nothing.
 */
export function cells(windows: readonly WindowSummary[]): CellKind[] {
  return windows.flatMap((w): CellKind[] => {
    if (w.k_proposed <= 0) {
      return [];
    }
    const accepted = Array.from({ length: w.n_accepted }, (): CellKind => 'accepted');
    return [...accepted, w.n_accepted < w.k_proposed ? 'rejected' : 'bonus'];
  });
}

/** The last n cells (all of them when there are fewer). */
export function tail<T>(items: readonly T[], n: number): T[] {
  return items.slice(Math.max(0, items.length - n));
}
