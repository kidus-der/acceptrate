"""verify/reference.py — framework-free comparison of two token sequences.

The model-backed cross-check against mlx-lm lives in tests/test_reference.py;
these tests pin the pure comparison logic with plain lists so CI covers it.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from acceptrate.verify.reference import (
    ReferenceReport,
    compare_sequences,
    describe,
    first_divergence,
)

TOKENS = st.lists(st.integers(min_value=0, max_value=128_255), max_size=40)


def test_identical_sequences_have_no_divergence() -> None:
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None


def test_empty_sequences_have_no_divergence() -> None:
    assert first_divergence([], []) is None


def test_divergence_is_the_first_mismatching_index() -> None:
    assert first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2


def test_a_shorter_prefix_diverges_where_the_longer_one_continues() -> None:
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert first_divergence([1, 2, 3], [1, 2]) == 2


@given(TOKENS, TOKENS)
def test_divergence_index_is_symmetric_and_a_prefix_bound(a: list[int], b: list[int]) -> None:
    index = first_divergence(a, b)

    assert index == first_divergence(b, a)
    if index is None:
        assert a == b
    else:
        assert a[:index] == b[:index]
        assert index <= min(len(a), len(b))


def test_compare_sequences_reports_a_match() -> None:
    report = compare_sequences("code-001", [5, 6, 7], [5, 6, 7])

    assert report == ReferenceReport(
        prompt_id="code-001",
        ours=(5, 6, 7),
        reference=(5, 6, 7),
        first_divergence=None,
        matched=True,
    )


def test_compare_sequences_reports_a_mismatch_with_its_index() -> None:
    report = compare_sequences("prose-002", [5, 6, 7], [5, 8, 7])

    assert not report.matched
    assert report.first_divergence == 1


def test_describe_names_the_index_and_both_tokens() -> None:
    report = compare_sequences("prose-002", [5, 6, 7], [5, 8, 7])

    text = describe(report)

    assert "prose-002" in text
    assert "index 1" in text
    assert "ours=6" in text
    assert "reference=8" in text


def test_describe_a_match_says_so() -> None:
    assert "matched" in describe(compare_sequences("json-003", [1], [1]))
