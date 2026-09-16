"""End-to-end distributional losslessness on fakes: plain vs speculative sampling.

For a fixed prefix, N next-token samples are drawn each way and the two
empirical laws must sit under tv_bound. Token index 1 is compared, not 0:
index 0 comes straight from the target's prefill logits in both paths and
would pass trivially; index 1 is the first token that goes through
draft -> accept/reject -> residual/bonus.

The negative control forces every draft to be accepted, which makes the
speculative law the draft's law; with an unrelated draft that must exceed
the bound, or the test would be unable to fail.
"""

from __future__ import annotations

import numpy as np
import pytest

import acceptrate.runtime.sampled as sampled_module
from acceptrate.verify.distributional import passes, run_distributional_check
from tests.fakes_prob import ProbFake

V = 32
N = 4000
PREFIX = [3]


def _perturbed_draft() -> ProbFake:
    return ProbFake.perturbed(ProbFake(V, seed=0), seed=1, noise=0.5)


def _unrelated_draft() -> ProbFake:
    return ProbFake(V, seed=9)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("make_draft", [_perturbed_draft, _unrelated_draft])
def test_speculative_sampling_matches_plain_sampling_in_tv(temperature, make_draft) -> None:
    report, bound = run_distributional_check(
        ProbFake(V, seed=0), make_draft(), PREFIX, k=4, temperature=temperature,
        n_samples=N, seed=42, vocab_size=V,
    )  # fmt: skip

    assert report.n_plain == N and report.n_spec == N
    assert 0.0 < bound < 0.2
    assert passes(report, bound), (report, bound)


def test_always_accepting_drafts_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampled_module, "accept_probability", lambda p, q: 1.0)

    report, bound = run_distributional_check(
        ProbFake(V, seed=0), _unrelated_draft(), PREFIX, k=4, temperature=1.0,
        n_samples=N, seed=42, vocab_size=V,
    )  # fmt: skip

    assert not passes(report, bound), (report, bound)
    assert report.tv > 2 * bound


def test_check_is_reproducible_in_seed_and_independent_between_paths() -> None:
    kwargs = dict(k=2, temperature=1.0, n_samples=500, seed=7, vocab_size=V)

    a, _ = run_distributional_check(ProbFake(V, seed=0), _perturbed_draft(), PREFIX, **kwargs)
    b, _ = run_distributional_check(ProbFake(V, seed=0), _perturbed_draft(), PREFIX, **kwargs)
    c, _ = run_distributional_check(
        ProbFake(V, seed=0), _perturbed_draft(), PREFIX, **{**kwargs, "seed": 8}
    )

    assert a == b
    assert a != c


def test_check_uses_the_observed_support_for_the_bound() -> None:
    from acceptrate.verify.distributional import tv_bound

    report, bound = run_distributional_check(
        ProbFake(V, seed=0), _perturbed_draft(), PREFIX, k=2, temperature=0.3,
        n_samples=300, seed=1, vocab_size=V,
    )  # fmt: skip

    # at T = 0.3 with 300 draws far fewer than 32 tokens appear, so the bound is tighter
    assert bound < tv_bound(300, 300, V)
    assert np.isfinite(report.tv)


def test_check_rejects_non_positive_sample_count() -> None:
    with pytest.raises(ValueError):
        run_distributional_check(
            ProbFake(V, seed=0), _perturbed_draft(), PREFIX, k=2, temperature=1.0,
            n_samples=0, seed=1, vocab_size=V,
        )  # fmt: skip
