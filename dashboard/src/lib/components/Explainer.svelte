<section class="panel min-w-0 p-4 text-sm leading-relaxed" aria-labelledby="explainer-title">
  <h2 id="explainer-title" class="panel-title mb-3">the equation</h2>
  <p class="data equation" aria-label="speedup of alpha, K and c equals one minus alpha to the K plus one, over one minus alpha times K c plus one">
    speedup(α, K, c) = (1 − α<sup>K+1</sup>) / ((1 − α) · (K·c + 1))
  </p>
  <dl class="mt-3 grid gap-2">
    <div>
      <dt class="data">α</dt>
      <dd>
        The per-token acceptance rate: how often the target agrees with the draft.
        Measured live as an EWMA over recent windows — the same number the adaptive scheduler drives K from.
      </dd>
    </div>
    <div>
      <dt class="data">K</dt>
      <dd>
        The draft depth: tokens proposed per window. Deeper drafts harvest more
        when α is high and waste more when it is not; the marker on the curve is the K in use now.
      </dd>
    </div>
    <div>
      <dt class="data">c</dt>
      <dd>
        The cost ratio: one drafted token's time over one verify pass, measured
        as the median of (draft_ms / K) / verify_ms over the last windows. Below the dashed 1.0 line, speculation loses.
      </dd>
    </div>
  </dl>
</section>

<style>
  .equation {
    white-space: nowrap;
    overflow-x: auto;
    padding: 0.6rem 0.8rem;
    border-radius: 4px;
    background: var(--color-surface-2);
    color: var(--color-ink);
  }
  dt {
    display: inline;
    color: var(--color-measurement);
  }
  dt::after {
    content: ' — ';
    color: var(--color-muted);
  }
  dd {
    display: inline;
    color: var(--color-ink-2);
  }
</style>
