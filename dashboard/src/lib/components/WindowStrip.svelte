<script lang="ts">
  // Recent windows as tokens: ● accepted, ○ rejection point, ◆ bonus.
  // Glyph and colour both carry the meaning, so the strip reads without either.
  import { GLYPH, cells, tail } from '../strip';
  import type { WindowSummary } from '../types';

  interface Props {
    windows: readonly WindowSummary[];
  }

  const STRIP_WIDTH = 96;
  let { windows }: Props = $props();

  const shown = $derived(tail(cells(windows), STRIP_WIDTH));
  const counts = $derived({
    accepted: shown.filter((k) => k === 'accepted').length,
    rejected: shown.filter((k) => k === 'rejected').length,
    bonus: shown.filter((k) => k === 'bonus').length,
  });
</script>

<section class="panel min-w-0 p-4" aria-labelledby="strip-title">
  <h2 id="strip-title" class="panel-title mb-3">window strip · last {windows.length} windows</h2>
  <div
    class="strip data"
    role="img"
    aria-label="{counts.accepted} accepted, {counts.rejected} rejected, {counts.bonus} bonus tokens"
  >
    {#each shown as kind, i (i)}
      <span class={kind}>{GLYPH[kind]}</span>
    {:else}
      <span class="empty">no windows yet</span>
    {/each}
  </div>
  <p class="legend data mt-2">
    <span class="accepted">{GLYPH.accepted}</span> accepted
    <span class="rejected">{GLYPH.rejected}</span> rejection point
    <span class="bonus">{GLYPH.bonus}</span> bonus token
  </p>
</section>

<style>
  .strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.15rem 0.2rem;
    min-height: 1.6rem;
    font-size: 0.95rem;
    line-height: 1;
  }
  .accepted {
    color: var(--color-accepted);
  }
  .rejected {
    color: var(--color-rejected);
  }
  .bonus {
    color: var(--color-measurement);
  }
  .empty,
  .legend {
    color: var(--color-muted);
    font-size: 0.78rem;
  }
  .legend span {
    margin-left: 0.6rem;
  }
  .legend span:first-child {
    margin-left: 0;
  }
</style>
