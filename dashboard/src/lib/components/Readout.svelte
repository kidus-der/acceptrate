<script lang="ts">
  interface Props {
    alpha: number | null;
    c: number | null;
    k: number;
    bestK: number | null;
    tokS: number | null;
    speedupNow: number | null;
  }

  let { alpha, c, k, bestK, tokS, speedupNow }: Props = $props();

  const f2 = (v: number | null): string => (v == null ? '—' : v.toFixed(2));
  const f1 = (v: number | null): string => (v == null ? '—' : v.toFixed(1));
</script>

<dl class="data grid grid-cols-3 gap-x-4 gap-y-2 text-sm sm:grid-cols-6">
  <div><dt>α</dt><dd>{f2(alpha)}</dd></div>
  <div><dt>c</dt><dd>{f2(c)}</dd></div>
  <div><dt>K</dt><dd>{k}</dd></div>
  <div><dt>best K</dt><dd>{bestK ?? '—'}</dd></div>
  <div><dt>speedup</dt><dd>{speedupNow == null ? '—' : `${f2(speedupNow)}×`}</dd></div>
  <div><dt>tok/s</dt><dd>{f1(tokS)}</dd></div>
</dl>

<style>
  dt {
    color: var(--color-muted);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
  }
  dd {
    color: var(--color-ink);
    font-size: 1.15rem;
  }
</style>
