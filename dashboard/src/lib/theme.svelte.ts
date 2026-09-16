// Colour scheme + reduced-motion tracking, so canvas charts can re-read the
// tokens when the OS theme flips (CSS handles everything else on its own).
import type { Palette } from './charts';

export const theme = $state({ dark: false, reducedMotion: false });

const DARK = '(prefers-color-scheme: dark)';
const REDUCED = '(prefers-reduced-motion: reduce)';

/** Start tracking the media queries; returns a teardown. */
export function watchTheme(): () => void {
  const dark = window.matchMedia(DARK);
  const reduced = window.matchMedia(REDUCED);
  const sync = (): void => {
    theme.dark = dark.matches;
    theme.reducedMotion = reduced.matches;
  };
  sync();
  dark.addEventListener('change', sync);
  reduced.addEventListener('change', sync);
  return () => {
    dark.removeEventListener('change', sync);
    reduced.removeEventListener('change', sync);
  };
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** The current token colours, read from the generated custom properties. */
export function readPalette(): Palette {
  return {
    ink: cssVar('--color-ink'),
    ink2: cssVar('--color-ink-2'),
    muted: cssVar('--color-muted'),
    line: cssVar('--color-line'),
    teal: cssVar('--color-teal'),
    amber: cssVar('--color-amber'),
    green: cssVar('--color-green'),
    red: cssVar('--color-red'),
    font: `11px ${cssVar('--font-data-stack') || 'monospace'}`,
  };
}
