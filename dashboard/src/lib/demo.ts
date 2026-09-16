// The brief's scripted generation, replayed when no server is reachable.
// A port of tui/internal/mock/scenario.go: alpha climbs 0.45 -> 0.80 while K
// steps 4 -> 6 and throughput pulls away from the baseline.
import type { Stats, WindowSummary } from './types';

export const DEMO_MODEL = 'mlx-community/Llama-3.1-8B-Instruct-4bit';
export const DEMO_DRAFT = 'mlx-community/Llama-3.2-1B-Instruct-4bit';

const ALPHA_START = 0.45;
const ALPHA_END = 0.8;
const TOK_S_START = 18.0;
const TOK_S_END = 31.0;
const K_LOW = 4;
const K_HIGH = 6;
/** Fraction of the way through when K adapts. */
const K_STEP_AT = 0.4;
/** Words streamed per window. */
const WORDS_PER_WINDOW = 2;
/** Mirrors serve/stats.py LAST_WINDOWS. */
const LAST_WINDOWS = 16;
const MS_PER_S = 1000.0;
const DRAFT_SHARE = 0.15;
const VERIFY_SHARE = 0.75;

/** The answer flows from prose into a code block: the brief's money shot. */
const DEMO_ANSWER =
  'Speculative decoding pays off when the draft model agrees with the target ' +
  'often enough that verifying K tokens in one batched pass costs less than K sequential ' +
  'decode steps. In prose the acceptance rate drifts; in code it climbs, because the ' +
  'syntax is predictable:\n\n```python\ndef adaptive_k(alpha: float, k_max: int = 8) -> int:\n' +
  '    expected = (1 - alpha ** (k_max + 1)) / (1 - alpha)\n' +
  '    return max(1, min(k_max, round(expected)))\n```\n\n' +
  'That is why the draft depth springs from 4 to 6 as the block streams.';

export interface Scenario {
  /** The stats snapshot after each scripted draft window. */
  readonly steps: readonly Stats[];
  /** The stats shown once the generation has finished. */
  readonly idle: Stats;
}

/**
 * Split text into groups of n words, keeping the whitespace (including
 * newlines) attached to the preceding word so the join is exact.
 */
export function chunkWords(text: string, n: number): string[] {
  const words: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const next = text[i + 1];
    if (text[i] === ' ' && i + 1 < text.length && next !== ' ' && next !== '\n') {
      words.push(text.slice(start, i + 1));
      start = i + 1;
    }
  }
  words.push(text.slice(start));
  const chunks: string[] = [];
  for (let i = 0; i < words.length; i += n) {
    chunks.push(words.slice(i, i + n).join(''));
  }
  return chunks;
}

/** One window whose acceptance count wobbles around alpha*k so the strip shows real rejection points. */
function demoWindow(i: number, k: number, alpha: number, tokS: number): WindowSummary {
  const wobble = 0.5 * Math.sin(i * 1.7);
  const nAccepted = Math.max(0, Math.min(k, Math.round(alpha * k + wobble)));
  const committed = nAccepted + 1; // the corrected token or the bonus token
  const windowMs = (committed / tokS) * MS_PER_S;
  return {
    k_proposed: k,
    n_accepted: nAccepted,
    draft_ms: windowMs * DRAFT_SHARE,
    verify_ms: windowMs * VERIFY_SHARE,
    window_ms: windowMs,
  };
}

export function demoScenario(): Scenario {
  const chunks = chunkWords(DEMO_ANSWER, WORDS_PER_WINDOW);
  const n = chunks.length;
  const windows: WindowSummary[] = [];
  let accepted = 0;
  let proposed = 0;
  const steps = chunks.map((_, i): Stats => {
    const frac = i / (n - 1);
    const alpha = ALPHA_START + (ALPHA_END - ALPHA_START) * frac;
    const tokS = TOK_S_START + (TOK_S_END - TOK_S_START) * frac;
    const k = frac >= K_STEP_AT ? K_HIGH : K_LOW;
    const w = demoWindow(i, k, alpha, tokS);
    windows.push(w);
    accepted += w.n_accepted;
    proposed += w.k_proposed;
    return {
      model: DEMO_MODEL,
      draft: DEMO_DRAFT,
      busy: true,
      k_current: k,
      alpha_ewma: alpha,
      tok_s_recent: tokS,
      windows_total: i + 1,
      accepted_total: accepted,
      proposed_total: proposed,
      last_windows: windows.slice(Math.max(0, windows.length - LAST_WINDOWS)),
    };
  });
  const last = steps[n - 1];
  return { steps, idle: { ...last, busy: false } };
}
