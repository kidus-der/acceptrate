<script lang="ts">
  // The brief's calculator made live: alpha and c are measured, K is marked.
  import { curveColumns, curveOptions } from '../charts';
  import type { Palette } from '../charts';
  import { bestK, speedup } from '../speedup';
  import Chart from './Chart.svelte';
  import Readout from './Readout.svelte';

  interface Props {
    alpha: number | null;
    c: number | null;
    k: number;
    tokS: number | null;
    palette: Palette;
  }

  let { alpha, c, k, tokS, palette }: Props = $props();

  const ready = $derived(alpha !== null && c !== null);
  const opts = $derived(curveOptions(palette));
  const data = $derived(ready ? curveColumns(alpha as number, c as number, k) : null);
  const best = $derived(ready ? bestK(alpha as number, c as number) : null);
  const now = $derived(ready && k >= 0 ? speedup(alpha as number, k, c as number) : null);
</script>

<section class="panel min-w-0 p-4" aria-labelledby="calc-title">
  <h2 id="calc-title" class="panel-title mb-3">speedup vs K · measured α and c</h2>
  {#if data !== null}
    <Chart {opts} {data} label="Speedup versus draft depth K, current K marked" />
  {:else}
    <p class="waiting data" role="status">waiting for data — the curve appears after the first draft window</p>
  {/if}
  <div class="mt-4">
    <Readout {alpha} {c} {k} bestK={best} {tokS} speedupNow={now} />
  </div>
</section>

<style>
  .waiting {
    display: grid;
    place-items: center;
    height: 220px;
    border: 1px dashed var(--color-line-2);
    border-radius: 4px;
    color: var(--color-muted);
    font-size: 0.85rem;
  }
</style>
