<script lang="ts">
  import { seriesOptions } from '../charts';
  import type { Palette } from '../charts';
  import { toColumns } from '../history';
  import type { Sample } from '../history';
  import Chart from './Chart.svelte';

  interface Props {
    history: readonly Sample[];
    palette: Palette;
  }

  let { history, palette }: Props = $props();

  const opts = $derived(seriesOptions(palette));
  const data = $derived(toColumns(history));
</script>

<section class="panel p-4" aria-labelledby="ts-title">
  <h2 id="ts-title" class="panel-title mb-3">
    tok/s and α · last {history.length} windows
    <span class="key data"><span class="tok">—</span> tok/s (left) <span class="alpha">···</span> α (right)</span>
  </h2>
  {#if history.length > 1}
    <Chart {opts} {data} height={180} label="Tokens per second and acceptance rate over recent windows" />
  {:else}
    <p class="waiting data" role="status">waiting for data</p>
  {/if}
</section>

<style>
  .key {
    float: right;
    letter-spacing: 0;
  }
  .tok {
    color: var(--color-measurement);
  }
  .alpha {
    color: var(--color-accepted);
  }
  .waiting {
    display: grid;
    place-items: center;
    height: 180px;
    border: 1px dashed var(--color-line-2);
    border-radius: 4px;
    color: var(--color-muted);
    font-size: 0.85rem;
  }
</style>
