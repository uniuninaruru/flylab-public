"""flylab.experiments.stats: hand-checkable cases plus a coverage simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flylab.experiments.stats import (
    bootstrap_interval,
    derive_seed,
    describe,
    holm_adjust,
    paired_comparison,
    sign_flip_test,
    stat_dz,
    stat_mean,
)


def test_interval_is_reproducible_and_seed_dependent() -> None:
    # continuous values: bootstrap means of a few integers sit on a coarse grid, so two seeds
    # could legitimately give the same quantiles
    x = [3.14, 1.59, 2.65, 3.58, 9.79, 3.23, 8.46, 2.64]
    a = bootstrap_interval(x, seed=7, n_boot=2000)
    b = bootstrap_interval(x, seed=7, n_boot=2000)
    c = bootstrap_interval(x, seed=8, n_boot=2000)
    assert a == b
    assert (a.low, a.high) != (c.low, c.high)
    assert a.method == "bca"
    assert a.low is not None
    assert a.high is not None
    assert a.low < float(np.mean(x)) < a.high


def test_degenerate_and_undefined_intervals() -> None:
    same = bootstrap_interval([2.0, 2.0, 2.0, 2.0], n_boot=500)
    assert same.method == "degenerate"
    assert (same.low, same.high) == (2.0, 2.0)
    one = bootstrap_interval([2.0], n_boot=500)
    assert one.method == "undefined"
    assert one.low is None


def test_rejects_non_finite_and_bad_arguments() -> None:
    with pytest.raises(ValueError, match="finite"):
        bootstrap_interval([1.0, math.nan])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval([1.0, 2.0], confidence=1.0)
    with pytest.raises(ValueError, match="method"):
        bootstrap_interval([1.0, 2.0], method="bootstrap-t")


def test_percentile_matches_numpy_quantiles_of_the_same_resamples() -> None:
    x = np.array([0.5, 1.5, 2.0, 7.0, 3.0, 3.5])
    iv = bootstrap_interval(x, method="percentile", seed=3, n_boot=1000)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(3)))
    boot = x[rng.integers(0, x.size, size=(1000, x.size))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert (iv.low, iv.high) == (float(lo), float(hi))


def test_bca_equals_percentile_for_symmetric_data_without_bias() -> None:
    """For a sample symmetric about its mean the jackknife skewness is 0 (a = 0); with an
    even resample count around theta z0 is ~0, so BCa ~ percentile."""
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    bca = bootstrap_interval(x, seed=1, n_boot=20_000)
    pct = bootstrap_interval(x, seed=1, n_boot=20_000, method="percentile")
    assert bca.low == pytest.approx(pct.low, abs=0.1)
    assert bca.high == pytest.approx(pct.high, abs=0.1)


def test_bca_coverage_on_normal_samples() -> None:
    """Nominal 95 %: the observed coverage of the true mean must be close to it (n = 20)."""
    rng = np.random.default_rng(2024)
    trials, hits = 300, 0
    for t in range(trials):
        x = rng.normal(5.0, 2.0, size=20)
        iv = bootstrap_interval(x, seed=t, n_boot=1000)
        assert iv.low is not None
        assert iv.high is not None
        hits += iv.low <= 5.0 <= iv.high
    coverage = hits / trials
    assert 0.89 <= coverage <= 0.99, coverage


def test_dz_statistic_and_describe() -> None:
    d = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    out = stat_dz(d)
    assert out[0] == pytest.approx(2.0 / 1.0)
    assert math.isnan(out[1])
    assert stat_mean(d)[0] == pytest.approx(2.0)
    desc = describe([1.0, 2.0, 3.0, 4.0], n_boot=500)
    assert desc["n"] == 4
    assert desc["mean"] == pytest.approx(2.5)
    assert desc["sd"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert desc["median"] == pytest.approx(2.5)


def test_sign_flip_exact_p_values() -> None:
    # all 5 differences positive: only the identity and the all-flipped assignment are as extreme
    assert sign_flip_test([1.0, 2.0, 3.0, 4.0, 5.0])["p_value"] == pytest.approx(2 / 32)
    # all zero: every assignment ties -> p = 1
    assert sign_flip_test([0.0, 0.0, 0.0])["p_value"] == pytest.approx(1.0)
    # [1, -1]: |mean| is 0 for 2 of 4 assignments and 1 for the other 2; observed 0 -> p = 1
    assert sign_flip_test([1.0, -1.0])["p_value"] == pytest.approx(1.0)
    res = sign_flip_test(np.ones(20), seed=4, n_resamples=999)
    assert res["method"] == "monte_carlo"
    # exact p = 2/2^20; the Monte Carlo estimate is bounded below by 1/(1 + 999)
    assert res["p_value"] == pytest.approx(1 / 1000, abs=3 / 1000)


def test_holm_known_example() -> None:
    # Holm 1979 style: p sorted 0.01, 0.02, 0.03, 0.04 with m = 4 -> 0.04, 0.06, 0.06, 0.06
    adj = holm_adjust([0.03, None, 0.01, 0.04, 0.02])
    assert adj == pytest.approx([0.06, None, 0.04, 0.06, 0.06])
    assert holm_adjust([]) == []
    assert holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        holm_adjust([1.5])


def test_paired_comparison_by_hand() -> None:
    cond = [12.0, 14.0, 11.0, 15.0]
    ref = [10.0, 10.0, 10.0, 10.0]
    res = paired_comparison(cond, ref, n_boot=2000, seed=5)
    d = np.array(cond) - np.array(ref)
    assert res["n_pairs"] == 4
    assert res["mean_difference"] == pytest.approx(d.mean())
    assert res["d_z"] == pytest.approx(d.mean() / d.std(ddof=1))
    assert res["relative_change"] == pytest.approx(d.mean() / 10.0)
    test = res["sign_flip_test"]
    assert isinstance(test, dict)
    assert test["p_value"] == pytest.approx(2 / 16)


def test_paired_comparison_undefined_parts() -> None:
    res = paired_comparison([1.0, 2.0, 3.0], [0.0, 1.0, 2.0], n_boot=500)
    assert res["d_z"] is None  # every difference is 1 -> sd 0
    assert "d_z_note" in res
    zero_ref = paired_comparison([1.0, 2.0], [0.0, 0.0], n_boot=500)
    assert zero_ref["relative_change"] is None
    with pytest.raises(ValueError, match="pair"):
        paired_comparison([1.0, 2.0], [1.0])


def test_derive_seed_is_stable_and_label_specific() -> None:
    assert derive_seed(0, "a", "b") == derive_seed(0, "a", "b")
    assert derive_seed(0, "a", "b") != derive_seed(0, "b", "a")
    assert 0 <= derive_seed(123, "x") < 2**63
