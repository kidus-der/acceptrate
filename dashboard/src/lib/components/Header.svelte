<script lang="ts">
  import type { ConnectionStatus } from '../client';
  import type { Mode } from '../store.svelte';

  interface Props {
    model: string | null;
    draft: string | null;
    busy: boolean;
    status: ConnectionStatus;
    detail: string;
    mode: Mode;
    onToggleDemo: () => void;
  }

  let { model, draft, busy, status, detail, mode, onToggleDemo }: Props = $props();

  const unavailable = $derived(mode === 'live' && status === 'reconnecting');
  const feed = $derived(
    mode === 'demo' ? 'demo scenario' : status === 'open' ? 'live feed' : status,
  );
</script>

<header class="flex flex-wrap items-baseline gap-x-6 gap-y-2">
  <div class="min-w-0 grow">
    <h1 class="text-lg font-semibold tracking-tight">AcceptRate</h1>
    <p class="data truncate text-sm" style="color: var(--color-ink-2)">
      {model ?? '—'}
      <span style="color: var(--color-muted)">+ draft</span>
      {draft ?? '—'}
    </p>
  </div>
  <div class="flex items-center gap-4 text-sm">
    <span class="data flex items-center gap-2" aria-live="polite">
      <span class="dot" class:busy aria-hidden="true"></span>
      {busy ? 'generating' : 'idle'}
    </span>
    <span class="data" style="color: var(--color-muted)" title={detail}>{feed}</span>
    <button type="button" class="toggle" aria-pressed={mode === 'demo'} onclick={onToggleDemo}>
      {mode === 'demo' ? 'stop demo' : 'demo'}
    </button>
  </div>
  {#if unavailable}
    <p class="basis-full text-sm" style="color: var(--color-cost)">
      No server on this origin ({detail}). The demo replays the brief's scenario without one.
    </p>
  {/if}
</header>

<style>
  .dot {
    display: inline-block;
    width: 0.6rem;
    height: 0.6rem;
    border-radius: 999px;
    background: var(--color-line-2);
  }
  .dot.busy {
    background: var(--color-measurement);
    animation: pulse 1.2s ease-in-out infinite;
  }
  @media (prefers-reduced-motion: reduce) {
    .dot.busy {
      animation: none;
    }
  }
  @keyframes pulse {
    50% {
      opacity: 0.35;
    }
  }
  .toggle {
    font: inherit;
    padding: 0.2rem 0.7rem;
    border: 1px solid var(--color-line-2);
    border-radius: 4px;
    background: var(--color-surface-2);
    color: var(--color-ink);
    cursor: pointer;
  }
  .toggle[aria-pressed='true'] {
    border-color: var(--color-measurement);
    color: var(--color-measurement);
  }
</style>
