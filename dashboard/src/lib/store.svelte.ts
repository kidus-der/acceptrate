// The one reactive store: whatever the page shows comes from here, whether
// the stats arrive over SSE or from the replayed demo scenario.
import { StatsClient } from './client';
import type { ConnectionStatus } from './client';
import { demoScenario } from './demo';
import { pushSample } from './history';
import type { Sample } from './history';
import type { Stats } from './types';

export type Mode = 'live' | 'demo';

export const live = $state({
  mode: 'live' as Mode,
  status: 'closed' as ConnectionStatus,
  detail: '',
  stats: null as Stats | null,
  history: [] as readonly Sample[],
  error: '',
});

/** Relative to the page, so /dashboard/ reaches /stats/stream wherever the app is mounted. */
export function streamUrl(): string {
  return new URL('../stats/stream', window.location.href).toString();
}

const DEMO_STEP_MS = 160;
const DEMO_IDLE_MS = 2500;

let client: StatsClient | null = null;
let demoTimer: ReturnType<typeof setTimeout> | null = null;

function applyStats(stats: Stats): void {
  live.stats = stats;
  live.history = pushSample(live.history, {
    t: stats.windows_total,
    tok_s: stats.tok_s_recent,
    alpha: stats.alpha_ewma,
  });
}

export function connect(): void {
  if (client !== null) {
    return;
  }
  live.mode = 'live';
  live.history = [];
  client = new StatsClient(streamUrl(), {
    onStats: applyStats,
    onHeartbeat: (stats) => {
      live.stats = stats;
    },
    onStatus: (status, detail) => {
      live.status = status;
      live.detail = detail ?? '';
    },
    onError: (message) => {
      live.error = message;
    },
  });
  client.start();
}

export function disconnect(): void {
  client?.stop();
  client = null;
}

function clearDemoTimer(): void {
  if (demoTimer !== null) {
    clearTimeout(demoTimer);
    demoTimer = null;
  }
}

function playDemo(): void {
  const { steps, idle } = demoScenario();
  live.history = [];
  const tick = (i: number): void => {
    if (i < steps.length) {
      applyStats(steps[i]);
      demoTimer = setTimeout(() => tick(i + 1), DEMO_STEP_MS);
      return;
    }
    live.stats = idle;
    demoTimer = setTimeout(playDemo, DEMO_IDLE_MS);
  };
  tick(0);
}

export function startDemo(): void {
  disconnect();
  clearDemoTimer();
  live.mode = 'demo';
  live.status = 'closed';
  live.detail = '';
  live.error = '';
  playDemo();
}

export function stopDemo(): void {
  clearDemoTimer();
  live.stats = null;
  connect();
}

export function toggleDemo(): void {
  if (live.mode === 'demo') {
    stopDemo();
  } else {
    startDemo();
  }
}
