// Mirrors acceptrate.serve.stats.Stats / WindowSummary (Pydantic is the source of truth).

export interface WindowSummary {
  readonly k_proposed: number;
  readonly n_accepted: number;
  readonly draft_ms: number;
  readonly verify_ms: number;
  readonly window_ms: number;
}

export interface Stats {
  readonly model: string;
  readonly draft: string | null;
  readonly busy: boolean;
  readonly k_current: number;
  readonly alpha_ewma: number | null;
  readonly tok_s_recent: number | null;
  readonly windows_total: number;
  readonly accepted_total: number;
  readonly proposed_total: number;
  readonly last_windows: readonly WindowSummary[];
  /** verify pass over K+1 tokens relative to one plain step, per K (docs/gates/P4.md) */
  readonly v_by_k: Readonly<Record<number, number>>;
}
