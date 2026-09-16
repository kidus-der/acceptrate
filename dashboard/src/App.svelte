<script lang="ts">
  import { onMount } from 'svelte';

  import Calculator from './lib/components/Calculator.svelte';
  import Explainer from './lib/components/Explainer.svelte';
  import Header from './lib/components/Header.svelte';
  import Timeseries from './lib/components/Timeseries.svelte';
  import WindowStrip from './lib/components/WindowStrip.svelte';
  import { costRatio } from './lib/stats';
  import { connect, disconnect, live, startDemo, toggleDemo } from './lib/store.svelte';
  import { readPalette, theme, watchTheme } from './lib/theme.svelte';

  const stats = $derived(live.stats);
  const alpha = $derived(stats?.alpha_ewma ?? null);
  const c = $derived(stats === null ? null : costRatio(stats.last_windows));
  const palette = $derived.by(() => {
    void theme.dark; // re-read the tokens when the scheme flips
    return readPalette();
  });

  onMount(() => {
    const untrack = watchTheme();
    const wantsDemo = new URLSearchParams(window.location.search).has('demo');
    if (wantsDemo) {
      startDemo();
    } else {
      connect();
    }
    return () => {
      untrack();
      disconnect();
    };
  });
</script>

<main class="mx-auto flex max-w-5xl flex-col gap-4 p-4 sm:p-6">
  <Header
    model={stats?.model ?? null}
    draft={stats?.draft ?? null}
    busy={stats?.busy ?? false}
    status={live.status}
    detail={live.detail}
    mode={live.mode}
    onToggleDemo={toggleDemo}
  />
  {#if live.error}
    <p class="data text-sm" role="alert" style="color: var(--color-rejected)">{live.error}</p>
  {/if}
  <div class="grid gap-4 lg:grid-cols-[3fr_2fr]">
    <Calculator {alpha} {c} k={stats?.k_current ?? 0} tokS={stats?.tok_s_recent ?? null} {palette} />
    <Explainer />
  </div>
  <Timeseries history={live.history} {palette} />
  <WindowStrip windows={stats?.last_windows ?? []} />
  <footer class="data text-xs" style="color: var(--color-muted)">
    {stats?.windows_total ?? 0} windows · {stats?.accepted_total ?? 0} / {stats?.proposed_total ?? 0} drafted tokens accepted
  </footer>
</main>
