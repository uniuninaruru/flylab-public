"""w_syn calibration arithmetic (flylab.experiments.calibration) on hand-made rate tables.

The statistics tables below are constructed test inputs, not simulation results.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from flylab.data.synthetic import SYNTHETIC_ID_BASE
from flylab.experiments.calibration import CalibrationError, calibrate
from flylab.experiments.replicates import GroupConfig, parse_group

BASE: dict[str, Any] = {
    "schema_version": 1,
    "experiment_id": "cal",
    "dataset": {"kind": "synthetic_random", "n_neurons": 50, "mean_out_degree": 5,
                "graph_seed": 1},
    "backend": "numpy",
    "dt_ms": 0.1,
    "steps": 100,
    "seed": 1,
    "model": {"parameter_set": "shiu2024_code_v1"},
    "synapse_sign": {"source": "graph", "unknown_sign_policy": "error"},
    "populations": {"grn": {"body_ids": [SYNTHETIC_ID_BASE]},
                    "mn": {"body_ids": [SYNTHETIC_ID_BASE + 1]}},
    "stimuli": [{"kind": "forced_spikes", "population": "grn", "onset_ms": 0.0,
                 "offset_ms": 10.0, "rate_hz": 100.0}],
    "interventions": [],
    "recording": {"spike_events": False},
}  # fmt: skip
WS = (0.1, 0.2, 0.3)
RATES = (50.0, 100.0, 200.0)


def _group() -> GroupConfig:
    def wl(w: float) -> dict[str, Any]:
        o = {"value": w, "units": "mV", "evidence": "ASSUMED", "rationale": "test"}
        return {"label": f"w{w}", "patch": {"model": {"overrides": {"w_syn": o}}}}

    def rl(r: float) -> dict[str, Any]:
        s = [dict(BASE["stimuli"][0], rate_hz=r)]
        return {"label": f"r{int(r)}", "patch": {"stimuli": s}}

    g = {
        "schema_version": 1,
        "kind": "flylab_replicate_group",
        "group_id": "cal",
        "base_config": "unused.yaml",
        "seeds": [1, 2],
        "sweep": [{"axis": "w", "levels": [wl(w) for w in WS]},
                  {"axis": "r", "levels": [rl(r) for r in RATES]}],
    }  # fmt: skip
    return parse_group(g, base_text=yaml.safe_dump(BASE))


def _stats(readout: dict[tuple[float, float], tuple[float, float]]) -> dict[str, Any]:
    """Fake group statistics: readout (mn) per seed; stimulus achieves 0.8 x requested."""
    conds: dict[str, Any] = {}
    for w in WS:
        for r in RATES:
            a, b = readout[(w, r)]

            def entry(x: float, y: float) -> dict[str, Any]:
                return {"mean_rate_hz": {"values_by_seed": {"1": x, "2": y}}}

            conds[f"w{w}_r{int(r)}"] = {
                "populations": {
                    "mn": entry(a, b),
                    "grn": entry(0.8 * r, 0.8 * r),
                    "__all__": entry(1.0, 1.0),
                }
            }
    return {"conditions": conds, "group_config_sha256": "test"}


def test_fraction_selection_and_definitions() -> None:
    # w=0.1: 100 Hz gives 4 of max 10 -> 0.4; w=0.2: 8 of 10 -> 0.8; w=0.3: 10 of 10 -> 1.0
    readout = {
        (0.1, 50.0): (1.0, 1.0), (0.1, 100.0): (4.0, 4.0), (0.1, 200.0): (10.0, 10.0),
        (0.2, 50.0): (2.0, 2.0), (0.2, 100.0): (7.0, 9.0), (0.2, 200.0): (10.0, 10.0),
        (0.3, 50.0): (9.0, 9.0), (0.3, 100.0): (10.0, 10.0), (0.3, 200.0): (5.0, 5.0),
    }  # fmt: skip
    res = calibrate(_group(), _stats(readout), readout="mn", stimulus="grn", n_boot=500)
    per = {e["w_syn_mV"]: e for e in res["per_w_syn"]}  # type: ignore[attr-defined]
    assert per[0.1]["D1"]["fraction"] == pytest.approx(0.4)
    assert per[0.2]["D1"]["fraction"] == pytest.approx(0.8)
    assert per[0.3]["D1"]["fraction"] == pytest.approx(1.0)
    # D2 divides by the highest swept rate (200 Hz): w=0.3 -> 10/5 = 2.0
    assert per[0.3]["D2"]["fraction"] == pytest.approx(2.0)
    sel = res["selected"]
    assert isinstance(sel, dict)
    assert sel["w_syn_mV"] == 0.2
    assert sel["evidence"] == "LEARNED"
    assert 0.2 in sel["levels_whose_ci_contains_target"]
    assert sel["linear_interpolation_crossings_mV"] == pytest.approx([0.2])
    # stimulus achieves 0.8 x requested: 125 Hz is not swept, so D3 uses the rate whose
    # achieved value (40, 80, 160) is closest to 100 -> 100 Hz requested (80 Hz achieved)
    assert "100.0 Hz requested" in res["definitions"]["D3"]  # type: ignore[index]


def test_silent_readout_is_undefined_not_zero() -> None:
    readout = {(w, r): (0.0, 0.0) for w in WS for r in RATES}
    res = calibrate(_group(), _stats(readout), readout="mn", stimulus="grn", n_boot=200)
    sel = res["selected"]
    assert isinstance(sel, dict)
    assert sel["w_syn_mV"] is None
    assert all(e["D1"]["fraction"] is None for e in res["per_w_syn"])  # type: ignore[attr-defined]


def test_rejects_unswept_reference_and_missing_population() -> None:
    readout = {(w, r): (1.0, 1.0) for w in WS for r in RATES}
    with pytest.raises(CalibrationError, match="reference rate"):
        calibrate(_group(), _stats(readout), readout="mn", stimulus="grn",
                  reference_rate_hz=125.0, n_boot=200)  # fmt: skip
    with pytest.raises(CalibrationError, match="population"):
        calibrate(_group(), _stats(readout), readout="nope", stimulus="grn", n_boot=200)


def test_two_groups_combine_and_overlap_is_refused() -> None:
    readout = {(w, r): (1.0, 1.0) for w in WS for r in RATES}
    g = _group()
    res = calibrate(g, _stats(readout), readout="mn", stimulus="grn", n_boot=200)
    assert len(res["per_w_syn"]) == 3  # type: ignore[arg-type]
    with pytest.raises(CalibrationError, match="two groups"):
        calibrate(g, _stats(readout), readout="mn", stimulus="grn", n_boot=200,
                  extra=[(g, _stats(readout))])  # fmt: skip
