"""`w_syn` calibration against the criterion of Shiu et al. (2024) (PR-04, OPEN_QUESTIONS Q16).

Shiu et al., Methods: "We chose Wsyn such that activation of sugar GRNs at 100 Hz resulted in
roughly 80% of maximal MN9 firing". The paper does not define "maximal MN9 firing" numerically,
so FlyLab fixes a definition (ENGINEERED) and reports two alternatives:

- **D1 (primary)**: the largest seed-mean readout rate over the swept input rates, at the same
  `w_syn`: `F(w) = M(w, r_ref) / max_r M(w, r)`; `M` = mean over seeds of the readout
  population's whole-run mean rate; `r_ref` = 100 Hz requested Poisson rate.
- **D2**: the readout rate at the highest swept input rate: `F2(w) = M(w, r_ref) / M(w, r_max)`.
- **D3**: as D1, but evaluated at the requested rate whose *achieved* stimulus-population rate is
  closest to `r_ref` (FlyLab's forced spikes respect the 2.2 ms refractory period, so a requested
  100 Hz gives fewer spikes; Shiu et al.'s Poisson input set the targets' refractory period to 0).

Uncertainty: seeds are resampled jointly across input rates at one `w_syn` (the same seeds run at
every rate), `F` is recomputed per resample, and the percentile interval is reported. The
selected `w_syn` is the swept level with `F` closest to the target (no interpolation is used for
the selection; a linear interpolation between bracketing levels is reported for information).

The result is labelled LEARNED: it is fit to a stated criterion on stated simulations. It is a
value calibrated for this criterion on this graph, **not** a measured synaptic strength.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flylab import SCHEMA_VERSION
from flylab.experiments.replicates import GroupConfig, member_config, parse_group
from flylab.experiments.stats import DEFAULT_N_BOOT, derive_seed
from flylab.provenance.labels import Evidence

SHIU_CRITERION = (
    "Shiu et al. 2024, Nature 634:210-219, doi:10.1038/s41586-024-07763-9, Methods: 'We chose "
    "Wsyn such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 "
    "firing' (refs 64, 65 = Dahanukar et al. 2007; Inagaki et al. 2012)"
)


class CalibrationError(ValueError):
    """The group cannot be read as a w_syn x rate sweep."""


@dataclass(frozen=True)
class Cell:
    condition: str
    w_syn: float
    rate_hz: float
    readout: tuple[float, ...]  # per seed, group seed order
    stimulus: tuple[float, ...]
    whole: tuple[float, ...]


def _rate_of(cfg_stimuli: Sequence[Any], population: str) -> float:
    rates = [s.rate_hz for s in cfg_stimuli if s.population == population and s.rate_hz]
    if len(rates) != 1:
        raise CalibrationError(f"expected one Poisson stimulus on {population!r}, got {rates}")
    return float(rates[0])


def sweep_cells(
    group: GroupConfig,
    stats: Mapping[str, Any],
    readout: str,
    stimulus: str,
) -> list[Cell]:
    cells = []
    for c in group.conditions:
        cfg = member_config(group, c, group.seeds[0])
        pops = stats["conditions"][c.name]["populations"]
        for p in (readout, stimulus, "__all__"):
            if p not in pops:
                raise CalibrationError(f"population {p!r} not in the group statistics")

        by_seed = {
            p: tuple(float(pops[p]["mean_rate_hz"]["values_by_seed"][str(s)]) for s in group.seeds)
            for p in (readout, stimulus, "__all__")
        }
        cells.append(
            Cell(
                condition=c.name,
                w_syn=float(cfg.model.parameters().w_syn.value),
                rate_hz=_rate_of(cfg.stimuli, stimulus),
                readout=by_seed[readout],
                stimulus=by_seed[stimulus],
                whole=by_seed["__all__"],
            )
        )
    return cells


def _fraction(m: np.ndarray, rates: np.ndarray, ref_i: int, definition: str) -> np.ndarray:
    """m: (..., n_rates) seed-mean readout. Returns F along the leading axes (nan if max is 0)."""
    denom = m.max(axis=-1) if definition == "max" else m[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, m[..., ref_i] / denom, np.nan)


def _interval(boot: np.ndarray, confidence: float) -> dict[str, object]:
    ok = boot[np.isfinite(boot)]
    a = 1.0 - confidence
    if ok.size < 0.5 * boot.size or ok.size == 0:
        return {"low": None, "high": None, "method": "undefined",
                "n_undefined": int(boot.size - ok.size)}  # fmt: skip
    lo, hi = np.quantile(ok, [a / 2, 1 - a / 2])
    return {"low": float(lo), "high": float(hi), "method": "percentile",
            "n_undefined": int(boot.size - ok.size)}  # fmt: skip


def calibrate(
    group: GroupConfig,
    stats: Mapping[str, Any],
    *,
    readout: str,
    stimulus: str,
    reference_rate_hz: float = 100.0,
    target_fraction: float = 0.8,
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = 0.95,
    seed: int = 0,
    extra: Sequence[tuple[GroupConfig, Mapping[str, Any]]] = (),
) -> dict[str, object]:
    """Calibrate on one sweep group, or on several (`extra`) that use the same seeds and
    cover disjoint w_syn levels with the same rates (a coarse sweep plus a refinement)."""
    cells = sweep_cells(group, stats, readout, stimulus)
    for g2, s2 in extra:
        if g2.seeds != group.seeds:
            raise CalibrationError(f"group {g2.group_id} uses seeds {g2.seeds}, not {group.seeds}")
        cells += sweep_cells(g2, s2, readout, stimulus)
    ws = sorted({c.w_syn for c in cells})
    rates = np.array(sorted({c.rate_hz for c in cells}))
    grid: dict[tuple[float, float], Cell] = {}
    for c in cells:
        if (c.w_syn, c.rate_hz) in grid:
            raise CalibrationError(f"w_syn {c.w_syn} x rate {c.rate_hz} appears in two groups")
        grid[(c.w_syn, c.rate_hz)] = c
    missing = [(w, float(r)) for w in ws for r in rates if (w, float(r)) not in grid]
    if missing:
        raise CalibrationError(f"sweep is not a full w_syn x rate grid; missing {missing[:5]}")
    if reference_rate_hz not in rates:
        raise CalibrationError(f"reference rate {reference_rate_hz} Hz was not swept")
    ref_i = int(np.flatnonzero(rates == reference_rate_hz)[0])
    n_seeds = len(group.seeds)
    achieved = np.array(
        [np.mean([np.mean(grid[(w, float(r))].stimulus) for w in ws]) for r in rates]
    )
    d3_i = int(np.argmin(np.abs(achieved - reference_rate_hz)))

    per_w: list[dict[str, object]] = []
    # (definition) -> list of (w, fraction or None, ci) for the selection below
    found: dict[str, list[tuple[float, float | None, dict[str, object]]]] = {
        "D1": [], "D2": [], "D3": []
    }  # fmt: skip
    for w in ws:
        x = np.array([grid[(w, float(r))].readout for r in rates]).T  # (seeds, rates)
        m = x.mean(axis=0)
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(derive_seed(seed, "calibrate", str(w))))
        )
        idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
        mb = x[idx].mean(axis=1)  # (boot, rates)
        entry: dict[str, object] = {
            "w_syn_mV": w,
            "rates_hz": [float(r) for r in rates],
            "readout_mean_hz": [float(v) for v in m],
            "readout_by_seed_hz": {
                str(s): [float(v) for v in x[k]] for k, s in enumerate(group.seeds)
            },
            "stimulus_achieved_mean_hz": [
                float(np.mean(grid[(w, float(r))].stimulus)) for r in rates
            ],
            "whole_brain_mean_hz": [float(np.mean(grid[(w, float(r))].whole)) for r in rates],
            "whole_brain_by_seed_hz": {
                str(s): [float(grid[(w, float(r))].whole[k]) for r in rates]
                for k, s in enumerate(group.seeds)
            },
            "max_readout_hz": float(m.max()),
            "argmax_rate_hz": float(rates[int(np.argmax(m))]),
        }
        for name, definition, ri in (("D1", "max", ref_i), ("D2", "last", ref_i),
                                     ("D3", "max", d3_i)):  # fmt: skip
            f = float(_fraction(m, rates, ri, definition))
            ci = _interval(_fraction(mb, rates, ri, definition), confidence)
            frac = None if np.isnan(f) else f
            entry[name] = {"fraction": frac, "ci": ci}
            found[name].append((w, frac, ci))
        per_w.append(entry)

    def pick(name: str) -> dict[str, object]:
        defined = [(w, f, ci) for w, f, ci in found[name] if f is not None]
        if not defined:
            return {"w_syn_mV": None, "note": "readout never fired: criterion undefined"}
        w_best, f_best, ci_best = min(defined, key=lambda t: abs(t[1] - target_fraction))
        contains = []
        for w, _, ci in found[name]:
            lo, hi = ci.get("low"), ci.get("high")
            if isinstance(lo, float) and isinstance(hi, float) and lo <= target_fraction <= hi:
                contains.append(w)
        crossings: list[float] = []
        for (w0, f0, _), (w1, f1, _) in itertools.pairwise(defined):
            if f0 == target_fraction:
                crossings.append(w0)
            elif (f0 - target_fraction) * (f1 - target_fraction) < 0:
                crossings.append(w0 + (target_fraction - f0) * (w1 - w0) / (f1 - f0))
        if defined[-1][1] == target_fraction:
            crossings.append(defined[-1][0])
        return {
            "w_syn_mV": w_best,
            "fraction": f_best,
            "ci": ci_best,
            "levels_whose_ci_contains_target": contains,
            "linear_interpolation_crossings_mV": crossings,
            "note": "selected = swept level nearest the target; interpolation for information "
            "only (F need not be monotonic in w_syn)",
        }

    selected = pick("D1")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_wsyn_calibration",
        "group_id": group.group_id,
        "group_config_sha256": stats.get("group_config_sha256"),
        "groups": [
            {"group_id": g.group_id, "group_config_sha256": st.get("group_config_sha256")}
            for g, st in ((group, stats), *extra)
        ],
        "criterion": {
            "source": SHIU_CRITERION,
            "evidence": Evidence.LITERATURE_DERIVED.value,
            "target_fraction": target_fraction,
            "reference_rate_hz": reference_rate_hz,
            "readout_population": readout,
            "stimulus_population": stimulus,
            "note": "Shiu et al. applied it to FlyWire v630 (female brain, Brian2 PoissonInput "
            "with the targets' refractory period set to 0); here it is applied to MaleCNS with "
            "FlyLab forced spikes",
        },
        "definitions": {
            "evidence": Evidence.ENGINEERED.value,
            "M": "mean over seeds of the readout population's whole-run mean rate (Hz)",
            "D1": "F = M(w, r_ref) / max over swept rates of M(w, r)  [primary]",
            "D2": "F = M(w, r_ref) / M(w, r_max) with r_max the highest swept rate",
            "D3": "as D1 but at the requested rate whose achieved stimulus rate is closest to "
            f"r_ref: {float(rates[d3_i])} Hz requested -> {float(achieved[d3_i]):.1f} Hz achieved",
            "uncertainty": f"percentile interval over {n_boot} joint resamples of the "
            f"{n_seeds} seeds (same seeds at every rate), confidence {confidence}",
        },
        "seeds": list(group.seeds),
        "per_w_syn": per_w,
        "selected": {
            **selected,
            "definition": "D1",
            "evidence": Evidence.LEARNED.value,
            "rationale": "w_syn calibrated FOR THIS CRITERION on this graph and model; not a "
            "physiological synaptic strength",
        },
        "selected_D2": pick("D2"),
        "selected_D3": pick("D3"),
        "stimulus_achieved_mean_hz_by_rate": dict(
            zip([str(float(r)) for r in rates], [float(a) for a in achieved], strict=True)
        ),
    }


def _read_group_dir(group_dir: str | Path) -> tuple[GroupConfig, dict[str, Any]]:
    gd = Path(group_dir)
    group = parse_group(yaml.safe_load((gd / "group.yaml").read_text(encoding="utf-8")))
    stats: dict[str, Any] = json.loads((gd / "group_stats.json").read_text(encoding="utf-8"))
    return group, stats


def calibrate_group_dir(group_dir: str | Path, **kwargs: Any) -> dict[str, object]:
    return calibrate_group_dirs([group_dir], **kwargs)


def calibrate_group_dirs(group_dirs: Sequence[str | Path], **kwargs: Any) -> dict[str, object]:
    first, *rest = [_read_group_dir(d) for d in group_dirs]
    return calibrate(first[0], first[1], extra=rest, **kwargs)
