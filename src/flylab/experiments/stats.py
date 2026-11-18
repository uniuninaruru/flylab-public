"""Replicate statistics (spec 8.3): bootstrap intervals, paired effect sizes, sign-flip tests.

The unit of replication is chosen by the caller. In replicate groups it is one stimulus seed:
one realisation of the Poisson input (and, for null-graph conditions, of the shuffled graph) on
one fixed connectome and parameter set. Seeds are **not** animals; the spread they show is the
model's sensitivity to its own input noise, not biological variability.

Methods (all ENGINEERED choices, recorded with every result):

- Confidence interval of a mean: bias-corrected and accelerated (BCa) bootstrap
  (Efron 1987, JASA 82:171-185), jackknife acceleration, ties counted half (the rule
  `scipy.stats.bootstrap` uses). Falls back to the percentile interval when bootstrap replicates
  are not all finite, and reports a degenerate interval when every replicate is equal.
- Paired comparison (same seed in condition and reference): mean difference with a BCa interval,
  Cohen's d_z = mean(d) / sd(d) with a percentile interval, relative change
  mean(d) / mean(reference) with a paired percentile interval, and a two-sided sign-flip
  permutation test of mean(d) (exact up to `EXACT_SIGN_FLIP_MAX_N` pairs).
- Multiple comparisons: Holm (1979) step-down adjustment over a stated family.

Quantiles of the normal distribution come from `statistics.NormalDist` (no SciPy dependency).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_N_BOOT = 10_000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_PERMUTATIONS = 20_000
EXACT_SIGN_FLIP_MAX_N = 16

Statistic = Callable[[NDArray[np.float64]], NDArray[np.float64]]  # reduces the last axis
_N = NormalDist()


def stat_mean(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(a.mean(axis=-1), dtype=np.float64)


def stat_dz(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cohen's d_z of paired differences along the last axis (nan where sd is 0)."""
    sd = a.std(axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(np.where(sd > 0, a.mean(axis=-1) / sd, np.nan), dtype=np.float64)


def derive_seed(base: int, *parts: str) -> int:
    """Deterministic 63-bit seed for one statistic: sha256 of the base seed and labels."""
    h = hashlib.sha256("|".join([str(base), *parts]).encode()).digest()
    return int.from_bytes(h[:8], "big") & (2**63 - 1)


def _vector(values: ArrayLike, name: str = "values") -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


@dataclass(frozen=True)
class Interval:
    low: float | None
    high: float | None
    confidence: float
    method: str  # bca | percentile | degenerate | undefined
    n_boot: int
    seed: int
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _quantiles(boot: NDArray[np.float64], probs: Sequence[float]) -> tuple[float, float]:
    lo, hi = np.quantile(boot, probs)
    return float(lo), float(hi)


def bootstrap_interval(
    values: ArrayLike,
    statistic: Statistic = stat_mean,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    method: str = "bca",
) -> Interval:
    """Bootstrap confidence interval of `statistic` over the replicates in `values`."""
    if method not in ("bca", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    x = _vector(values)
    n = x.size
    if n < 2:
        return Interval(None, None, confidence, "undefined", n_boot, seed, "needs >= 2 replicates")
    theta = float(statistic(x[None, :])[0])
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    boot = statistic(x[rng.integers(0, n, size=(n_boot, n))])
    finite = boot[np.isfinite(boot)]
    note = None
    if finite.size < boot.size:
        note = f"{boot.size - finite.size} of {boot.size} bootstrap replicates not finite; excluded"
        if finite.size < 0.5 * boot.size:
            return Interval(None, None, confidence, "undefined", n_boot, seed, note)
        method = "percentile"
    if not math.isfinite(theta):
        return Interval(None, None, confidence, "undefined", n_boot, seed, "statistic not finite")
    if np.all(finite == finite[0]):
        v = float(finite[0])
        return Interval(v, v, confidence, "degenerate", n_boot, seed, "all replicates equal")
    alpha = 1.0 - confidence
    if method == "percentile":
        lo, hi = _quantiles(finite, (alpha / 2, 1 - alpha / 2))
        return Interval(lo, hi, confidence, "percentile", n_boot, seed, note)
    # BCa. Bias correction with ties counted half (as scipy.stats.bootstrap does).
    prop = float(np.count_nonzero(finite < theta) + np.count_nonzero(finite <= theta)) / (
        2.0 * finite.size
    )
    prop = min(max(prop, 1.0 / (finite.size + 1)), finite.size / (finite.size + 1.0))
    z0 = _N.inv_cdf(prop)
    loo = np.array([np.delete(x, i) for i in range(n)])
    jack = statistic(loo)
    if not np.isfinite(jack).all():
        lo, hi = _quantiles(finite, (alpha / 2, 1 - alpha / 2))
        return Interval(lo, hi, confidence, "percentile", n_boot, seed, "jackknife not finite")
    dev = jack.mean() - jack
    den = 6.0 * float((dev**2).sum()) ** 1.5
    a = float((dev**3).sum()) / den if den > 0 else 0.0
    probs = []
    for zq in (_N.inv_cdf(alpha / 2), _N.inv_cdf(1 - alpha / 2)):
        w = z0 + zq
        probs.append(_N.cdf(z0 + w / (1.0 - a * w)))
    lo, hi = _quantiles(finite, probs)
    return Interval(lo, hi, confidence, "bca", n_boot, seed, note)


def describe(
    values: ArrayLike,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict[str, object]:
    """n, mean, sd, median, range, per-replicate values and a BCa interval of the mean."""
    x = _vector(values)
    if x.size == 0:
        return {"n": 0, "values": []}
    return {
        "n": int(x.size),
        "values": [float(v) for v in x],
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else None,
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "ci_mean": bootstrap_interval(x, n_boot=n_boot, confidence=confidence, seed=seed).to_dict(),
    }


def sign_flip_test(
    differences: ArrayLike, *, seed: int = 0, n_resamples: int = DEFAULT_N_PERMUTATIONS
) -> dict[str, object]:
    """Two-sided paired permutation test of mean(d) = 0 by flipping signs of the differences."""
    d = _vector(differences, "differences")
    n = d.size
    out: dict[str, object] = {
        "statistic": "|mean paired difference|",
        "alternative": "two-sided",
    }
    if n == 0:
        return out | {"p_value": None, "method": "undefined", "note": "no pairs"}
    obs = abs(float(d.mean()))
    tol = 1e-12 * max(1.0, obs)
    if n <= EXACT_SIGN_FLIP_MAX_N:
        bits = (np.arange(2**n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
        stats = np.abs(((2 * bits - 1) * d).mean(axis=1))
        p = float(np.count_nonzero(stats >= obs - tol) / stats.size)
        return out | {"p_value": p, "method": "exact", "n_resamples": int(stats.size)}
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    signs = rng.integers(0, 2, size=(n_resamples, n)) * 2 - 1
    stats = np.abs((signs * d).mean(axis=1))
    p = (1 + int(np.count_nonzero(stats >= obs - tol))) / (1 + n_resamples)
    return out | {"p_value": float(p), "method": "monte_carlo", "n_resamples": n_resamples,
                  "seed": seed}  # fmt: skip


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values; `None` entries are skipped and stay `None`."""
    known = [(p, i) for i, p in enumerate(p_values) if p is not None]
    for p, _ in known:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range: {p}")
    m = len(known)
    out: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(known)):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def paired_comparison(
    condition: ArrayLike,
    reference: ArrayLike,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> dict[str, object]:
    """Effect of a condition against its matched reference; arrays are aligned by replicate."""
    x, y = _vector(condition, "condition"), _vector(reference, "reference")
    if x.shape != y.shape:
        raise ValueError("condition and reference must have one value per pair")
    n = int(x.size)
    d = x - y
    res: dict[str, object] = {
        "n_pairs": n,
        "condition_mean": float(x.mean()) if n else None,
        "reference_mean": float(y.mean()) if n else None,
        "mean_difference": float(d.mean()) if n else None,
        "sd_difference": float(d.std(ddof=1)) if n > 1 else None,
        "ci_mean_difference": bootstrap_interval(
            d, n_boot=n_boot, confidence=confidence, seed=seed
        ).to_dict(),
    }
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        res["d_z"] = float(d.mean() / sd)
        res["ci_d_z"] = bootstrap_interval(
            d, stat_dz, n_boot=n_boot, confidence=confidence, seed=seed + 1, method="percentile"
        ).to_dict()
    else:
        res["d_z"] = None
        res["ci_d_z"] = None
        res["d_z_note"] = "undefined: fewer than 2 pairs or all paired differences equal"
    if n and float(y.mean()) != 0.0:
        res["relative_change"] = float(d.mean() / y.mean())
        # Pairs are resampled jointly: the same replicate indices for d and the reference.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed + 2)))
        idx = rng.integers(0, n, size=(n_boot, n))
        num = d[idx].mean(axis=1)
        den = y[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / den
        ok = np.isfinite(ratio)
        alpha = 1.0 - confidence
        if ok.sum() >= 0.5 * n_boot and n > 1:
            lo, hi = _quantiles(ratio[ok], (alpha / 2, 1 - alpha / 2))
            res["ci_relative_change"] = Interval(
                lo, hi, confidence, "percentile", n_boot, seed + 2,
                None if ok.all() else f"{int((~ok).sum())} resamples with reference mean 0",
            ).to_dict()  # fmt: skip
        else:
            res["ci_relative_change"] = None
    else:
        res["relative_change"] = None
        res["ci_relative_change"] = None
    res["sign_flip_test"] = sign_flip_test(d, seed=seed + 3)
    return res
