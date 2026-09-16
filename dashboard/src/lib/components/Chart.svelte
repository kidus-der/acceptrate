<script lang="ts">
  // One uPlot instance bound to a host div: recreated when the options change
  // (palette flip), resized with the host, and fed new data in place.
  import { untrack } from 'svelte';
  import uPlot from 'uplot';

  interface Props {
    opts: Omit<uPlot.Options, 'width' | 'height'>;
    data: uPlot.AlignedData;
    height?: number;
    label: string;
  }

  const MIN_WIDTH = 240;
  let { opts, data, height = 220, label }: Props = $props();
  let host: HTMLDivElement;
  let chart: uPlot | null = null;

  $effect(() => {
    const width = Math.max(host.clientWidth, MIN_WIDTH);
    const initial = untrack(() => data);
    chart = new uPlot({ ...opts, width, height }, initial, host);
    const observer = new ResizeObserver(() => {
      chart?.setSize({ width: Math.max(host.clientWidth, MIN_WIDTH), height });
    });
    observer.observe(host);
    return () => {
      observer.disconnect();
      chart?.destroy();
      chart = null;
    };
  });

  $effect(() => {
    chart?.setData(data);
  });
</script>

<div bind:this={host} class="chart w-full" role="img" aria-label={label}></div>

<style>
  .chart :global(.u-wrap) {
    color: var(--color-ink);
  }
</style>
