"""serve/stats.py — pure statistics over the rolling window of recent draft windows.

These feed GET /stats and the TUI's SSE feed. Everything here is a function of
(rolling window, running totals); no I/O, no state.
"""

from __future__ import annotations

import pytest

from acceptrate.serve.stats import (
    EWMA_DECAY,
    Totals,
    WindowStat,
    compute_stats,
    ewma_alpha,
    tokens_per_second,
    window_stat_from_row,
)
from acceptrate.trace.schema import make_row


def _ws(k: int, n: int, window_ms: float = 10.0, committed: int | None = None) -> WindowStat:
    return WindowStat(
        k_proposed=k,
        n_accepted=n,
        n_committed=n + 1 if committed is None else committed,
        draft_ms=2.0,
        verify_ms=5.0,
        window_ms=window_ms,
    )


def test_window_stat_from_row_reads_the_schema_fields_by_name() -> None:
    row = make_row("r", 3, 10, 4, 2, 1.5, 6.0, "chat", 0, 0, 0, "p", 0, 9.0)

    stat = window_stat_from_row(row, n_committed=3)

    assert stat == WindowStat(4, 2, 3, 1.5, 6.0, 9.0)


def test_ewma_alpha_is_none_without_speculative_windows() -> None:
    assert ewma_alpha(()) is None
    assert ewma_alpha((_ws(0, 0),)) is None


def test_ewma_alpha_starts_at_the_first_rate_and_decays_towards_later_ones() -> None:
    window = (_ws(4, 4), _ws(4, 0))

    alpha = ewma_alpha(window)

    assert alpha == pytest.approx((1 - EWMA_DECAY) * 1.0 + EWMA_DECAY * 0.0)


def test_ewma_alpha_skips_plain_windows() -> None:
    assert ewma_alpha((_ws(0, 0), _ws(2, 1), _ws(0, 0))) == pytest.approx(0.5)


def test_tokens_per_second_uses_committed_tokens_over_wall_clock() -> None:
    window = (_ws(4, 4, window_ms=50.0), _ws(4, 1, window_ms=50.0))  # 5 + 2 tokens in 100 ms

    assert tokens_per_second(window) == pytest.approx(70.0)
    assert tokens_per_second(()) is None
    assert tokens_per_second((_ws(1, 0, window_ms=0.0),)) is None


def test_compute_stats_assembles_the_json_shape() -> None:
    window = tuple(_ws(4, n) for n in (4, 3, 0))

    stats = compute_stats(
        window,
        Totals(windows=30, accepted=70, proposed=120),
        model="target",
        draft="draft",
        busy=True,
        k_current=4,
        last_n=2,
    )

    body = stats.model_dump()
    assert body["model"] == "target"
    assert body["draft"] == "draft"
    assert body["busy"] is True
    assert body["k_current"] == 4
    assert body["windows_total"] == 30
    assert body["accepted_total"] == 70
    assert body["proposed_total"] == 120
    assert body["alpha_ewma"] == pytest.approx(ewma_alpha(window))
    assert body["tok_s_recent"] == pytest.approx(tokens_per_second(window))
    assert [w["n_accepted"] for w in body["last_windows"]] == [3, 0]
    assert set(body["last_windows"][0]) == {
        "k_proposed",
        "n_accepted",
        "draft_ms",
        "verify_ms",
        "window_ms",
    }


def test_compute_stats_on_an_empty_window_is_all_nulls_and_zeros() -> None:
    stats = compute_stats((), Totals(0, 0, 0), model="m", draft=None, busy=False, k_current=0)

    assert stats.alpha_ewma is None
    assert stats.tok_s_recent is None
    assert stats.last_windows == []
    assert stats.draft is None


def test_ewma_alpha_is_per_token_so_a_rejection_counts_as_one_examined() -> None:
    # 2 accepted of 4 proposed: the third was examined and rejected, the fourth never examined
    assert ewma_alpha((_ws(4, 2),)) == pytest.approx(2 / 3)


def test_stats_carry_the_machine_verify_cost_table() -> None:
    from acceptrate.model.speedup import M4_V_BY_K
    from acceptrate.serve.stats import Totals, compute_stats

    stats = compute_stats((), Totals(0, 0, 0), model="m", draft=None, busy=False, k_current=0)

    assert stats.v_by_k == {str(k): v for k, v in M4_V_BY_K.items()}
