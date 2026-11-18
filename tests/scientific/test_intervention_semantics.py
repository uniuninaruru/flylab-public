"""Intervention semantics of docs/MODEL_ASSUMPTIONS.md section 6, checked as equivalences.

Synthetic random graphs only: these test that each intervention does what its definition says,
not anything about flies.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import yaml

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import SYNTHETIC_ID_BASE, random_graph
from flylab.experiments.config import parse_config
from flylab.experiments.dataset import load_dataset
from flylab.experiments.populations import resolve_populations
from flylab.experiments.runner import RunResult, run_experiment
from flylab.experiments.schedule import select_removed_edges

N = 400
SILENCED = list(range(0, 20))
STIM = list(range(20, 40))


def _ids(idx: list[int]) -> list[int]:
    return [SYNTHETIC_ID_BASE + i for i in idx]


def _cfg(
    interventions: list[dict[str, Any]], stimuli: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "iv",
        "backend": "numpy",
        "dt_ms": 0.1,
        "steps": 1500,
        "seed": 9,
        "dataset": {
            "kind": "synthetic_random",
            "n_neurons": N,
            "mean_out_degree": 20,
            "graph_seed": 3,
        },
        "model": {
            "parameter_set": "shiu2024_code_v1",
            "overrides": {
                "w_syn": {
                    "value": 1.8,
                    "units": "mV",
                    "evidence": "ASSUMED",
                    "rationale": "test activity",
                }
            },
        },
        "synapse_sign": {"source": "graph", "unknown_sign_policy": "error"},
        "populations": {
            "silenced": {"body_ids": _ids(SILENCED)},
            "stim": {"body_ids": _ids(STIM)},
            "both": {"body_ids": _ids(SILENCED + STIM)},
        },
        "stimuli": stimuli
        if stimuli is not None
        else [
            {
                "kind": "injected_current",
                "population": "both",
                "onset_ms": 0.0,
                "offset_ms": 150.0,
                "amplitude_mV": 18.0,
            },
            {
                "kind": "forced_spikes",
                "population": "stim",
                "onset_ms": 0.0,
                "offset_ms": 150.0,
                "rate_hz": 40.0,
            },
        ],
        "interventions": interventions,
        "recording": {"population_rate_bin_ms": 10.0},
    }


def _run(cfg: dict[str, Any], tmp: Path, name: str) -> RunResult:
    path = tmp / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return run_experiment(path, tmp / "runs", name, progress=None)[0]


def _spikes(r: RunResult) -> list[tuple[int, int, str, bool]]:
    t = pq.read_table(r.run_dir / "spikes.parquet")
    return list(
        zip(
            t["step"].to_pylist(),
            t["idx"].to_pylist(),
            t["cause"].to_pylist(),
            t["propagated"].to_pylist(),
            strict=True,
        )
    )


def _outside(spikes: list[tuple[int, int, str, bool]], members: list[int]) -> list[tuple[int, int]]:
    m = set(members)
    return [(s, i) for s, i, _, _ in spikes if i not in m]


def test_block_output_equals_removing_all_outgoing_edges(tmp_path: Path) -> None:
    blocked = _run(
        _cfg([{"kind": "silence_neurons", "mode": "block_output", "population": "silenced"}]),
        tmp_path,
        "blocked",
    )
    removed = _run(
        _cfg([{"kind": "remove_synapses", "pre_population": "silenced"}]), tmp_path, "removed"
    )
    sb, sr = _spikes(blocked), _spikes(removed)
    assert len(sb) > 500
    # Same spikes everywhere, including inside the silenced population (it keeps integrating).
    assert [(s, i) for s, i, _, _ in sb] == [(s, i) for s, i, _, _ in sr]
    silenced = set(SILENCED)
    assert all(p is (i not in silenced) for _, i, _, p in sb)
    assert all(p for *_, p in sr)
    assert any(i in silenced for _, i, _, _ in sb)


def test_clamp_and_block_differ_only_inside_the_population(tmp_path: Path) -> None:
    blocked = _run(
        _cfg([{"kind": "silence_neurons", "mode": "block_output", "population": "silenced"}]),
        tmp_path,
        "b",
    )
    clamped = _run(
        _cfg([{"kind": "silence_neurons", "mode": "clamp_to_rest", "population": "silenced"}]),
        tmp_path,
        "c",
    )
    sb, sc = _spikes(blocked), _spikes(clamped)
    # Neither sends anything, so the rest of the network sees the same input.
    assert _outside(sb, SILENCED) == _outside(sc, SILENCED)
    # block_output: membrane integration continues, so the driven population still fires;
    # clamp_to_rest: no integration, no spikes.
    assert sum(1 for _, i, _, _ in sb if i in set(SILENCED)) > 0
    assert sum(1 for _, i, _, _ in sc if i in set(SILENCED)) == 0


def test_matched_baseline_is_identical_before_onset(tmp_path: Path) -> None:
    iv = [
        {
            "kind": "silence_neurons",
            "mode": "clamp_to_rest",
            "population": "silenced",
            "onset_ms": 60.0,
            "offset_ms": 100.0,
        }
    ]
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump(_cfg(iv)), encoding="utf-8")
    base, main = run_experiment(path, tmp_path / "runs", "w", with_baseline=True, progress=None)
    sb, sm = _spikes(base), _spikes(main)
    onset = 600
    assert [x for x in sb if x[0] < onset] == [x for x in sm if x[0] < onset]
    assert sb != sm


def test_injected_current_and_forced_spikes_are_distinct_modes(tmp_path: Path) -> None:
    stimuli = [
        {
            "kind": "injected_current",
            "population": "silenced",
            "onset_ms": 20.0,
            "offset_ms": 80.0,
            "amplitude_mV": 20.0,
        },
        {
            "kind": "forced_spikes",
            "population": "stim",
            "onset_ms": 20.0,
            "offset_ms": 80.0,
            "times_ms": [20.0, 40.0],
        },
    ]
    r = _run(_cfg([], stimuli), tmp_path, "modes")
    sp = _spikes(r)
    forced = {(s, i) for s, i, c, _ in sp if c == "forced"}
    requested = {(st, i) for st in (200, 400) for i in STIM}
    assert forced <= requested
    # Requests on refractory neurons are dropped and counted, never silently lost.
    totals = r.metrics["totals"]
    assert isinstance(totals, dict)
    assert len(forced) + totals["n_forced_dropped_refractory_or_clamped"] == len(requested)
    assert {(200, i) for i in STIM} <= forced  # nothing is refractory at the first request
    assert all(c == "threshold" for _, i, c, _ in sp if i in set(SILENCED))
    # No input before onset: nothing spikes before step 200.
    assert min(s for s, *_ in sp) >= 200
    t = pq.read_table(r.run_dir / "interventions.parquet")
    kinds = t.filter(pc.equal(t["category"], "stimulus"))["kind"].to_pylist()
    assert kinds.count("injected_current") == len(SILENCED)
    assert kinds.count("forced_spikes") == len(STIM)


def test_remove_synapses_subset_is_reproducible(tmp_path: Path) -> None:
    def removal(fraction: float, seed: int) -> np.ndarray[Any, np.dtype[np.bool_]]:
        c = parse_config(
            _cfg(
                [
                    {
                        "kind": "remove_synapses",
                        "pre_population": "stim",
                        "fraction": fraction,
                        "seed": seed,
                    }
                ]
            )
        )
        data = load_dataset(c.dataset)
        pops = resolve_populations(c.populations, data)
        return select_removed_edges(data.graph, c.interventions, pops).mask

    g, _ = random_graph(N, 20, seed=3)
    n_candidates = int(np.isin(g.src, STIM).sum())
    for fraction in (0.29, 0.5, 1.0):
        m = removal(fraction, 4)
        assert int(m.sum()) == math.floor(fraction * n_candidates + 1e-9)
        assert np.isin(g.src[m], STIM).all()
        np.testing.assert_array_equal(m, removal(fraction, 4))
    assert not np.array_equal(removal(0.5, 4), removal(0.5, 5))

    r = _run(
        _cfg([{"kind": "remove_synapses", "pre_population": "stim", "fraction": 0.5, "seed": 4}]),
        tmp_path,
        "rm",
    )
    t = pq.read_table(r.run_dir / "interventions.parquet")
    edges = t.filter(pc.equal(t["kind"], "remove_synapses"))
    m = removal(0.5, 4)
    assert sorted(
        zip(edges["idx"].to_pylist(), edges["dst_idx"].to_pylist(), strict=True)
    ) == sorted(zip(g.src[m].tolist(), g.dst[m].tolist(), strict=True))
    manifest_net = json.loads((r.run_dir / "manifest.json").read_text())["network"]
    assert manifest_net["n_removed_edges"] == int(m.sum())
    assert manifest_net["n_edges"] == g.n_edges - int(m.sum())


def test_removing_synapses_is_not_killing_the_neuron(tmp_path: Path) -> None:
    """Cutting a population's outputs leaves the population itself spiking (unlike clamping)."""
    r = _run(_cfg([{"kind": "remove_synapses", "pre_population": "silenced"}]), tmp_path, "cut")
    assert any(i in set(SILENCED) for _, i, _, _ in _spikes(r))


@pytest.mark.parametrize("policy", [UnknownSignPolicy.EXCLUDE, UnknownSignPolicy.AS_INHIBITORY])
def test_policy_is_recorded_for_every_run(tmp_path: Path, policy: UnknownSignPolicy) -> None:
    cfg = copy.deepcopy(_cfg([]))
    cfg["synapse_sign"]["unknown_sign_policy"] = policy.value
    r = _run(cfg, tmp_path, policy.value)
    manifest = json.loads((r.run_dir / "manifest.json").read_text())
    assert manifest["synapse_sign"]["unknown_sign_policy"] == policy.value
    assert manifest["network"]["unknown_sign_policy"] == policy.value
