"""The trace schema: one row per draft window. The seam bench/ and runtime/ share."""

from __future__ import annotations

import polars as pl

from acceptrate.trace.schema import (
    FIELD_NAMES,
    SCHEMA,
    WORKLOAD_TAGS,
    WindowRow,
    frame_from_rows,
    make_row,
)


def test_schema_has_every_field_from_the_brief_in_order() -> None:
    brief = (
        "run_id", "window_idx", "token_pos", "k_proposed", "n_accepted",
        "draft_ms", "verify_ms", "workload_tag", "mem_pressure", "page_ins", "thermal_level",
    )  # fmt: skip
    assert FIELD_NAMES[: len(brief)] == brief


def test_schema_adds_the_three_fields_needed_to_disambiguate_and_time_windows() -> None:
    assert FIELD_NAMES[-3:] == ("prompt_id", "rep", "window_ms")


def test_workload_tags_are_the_six_from_the_brief() -> None:
    assert WORKLOAD_TAGS == ("code", "json", "prose", "chat", "summarize", "reason")


def test_make_row_produces_a_plain_tuple_in_schema_order() -> None:
    row = make_row(
        run_id="r", window_idx=3, token_pos=17, k_proposed=4, n_accepted=2,
        draft_ms=8.5, verify_ms=55.0, workload_tag="code", mem_pressure=0,
        page_ins=0, thermal_level=0, prompt_id="p1", rep=0, window_ms=66.0,
    )  # fmt: skip

    assert type(row) is tuple
    assert len(row) == len(FIELD_NAMES)
    assert row[0] == "r"
    assert row[-1] == 66.0


def test_frame_from_rows_matches_the_polars_schema_exactly() -> None:
    rows: list[WindowRow] = [
        make_row("r", 0, 0, 0, 0, 0.0, 50.0, "prose", 0, 0, 0, "p1", 0, 51.0),
        make_row("r", 1, 1, 4, 3, 9.0, 55.0, "prose", 1, 2, 0, "p1", 0, 66.0),
    ]

    df = frame_from_rows(rows)

    assert df.schema == SCHEMA
    assert df.height == 2
    assert df["n_accepted"].dtype == pl.Int8
    assert df["page_ins"].dtype == pl.Int64
    assert df["draft_ms"].dtype == pl.Float32


def test_empty_rows_give_an_empty_frame_with_the_schema() -> None:
    df = frame_from_rows([])

    assert df.schema == SCHEMA
    assert df.height == 0
