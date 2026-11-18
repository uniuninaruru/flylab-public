"""Host-side parts of the fused CUDA engine (no GPU needed): input packing, ring layout,
integer-weight detection and the `engine_impl` setting."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.experiments.config import ConfigError, parse_config
from flylab.neural.engine import StepDrive
from flylab.neural.lif_fused import (
    aggregate_drive,
    fused_status,
    network_supports_fused,
    pack_inputs,
    ring_to_state,
    state_to_ring,
)
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize

DISC = discretize(SHIU2024_CODE_V1, 0.1)


def test_drive_aggregation_matches_add_at_bitwise() -> None:
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 50, 400).astype(np.int32)
    mv = rng.normal(0, 7, 400) * 1e3 + 1e-9  # values where summation order matters
    ref = np.zeros(50)
    np.add.at(ref, idx, mv)
    uniq, total = aggregate_drive(idx, mv)
    assert np.array_equal(uniq, np.unique(idx))
    assert np.array_equal(total, ref[uniq])  # bitwise, not approximately


def test_pack_inputs_layout() -> None:
    packed, lengths = pack_inputs(
        np.array([3, 1, 3], np.int32), np.array([1.5, 2.0, 0.25]),
        np.array([7], np.int32), np.zeros(0, np.int32), np.array([2, 9], np.int32),
    )  # fmt: skip
    assert lengths == (2, 1, 0, 2)
    assert packed[:2].tolist() == [1, 3]
    assert packed[2:4].view(np.float64).tolist() == [2.0, 1.75]
    assert packed[4:].tolist() == [7, 2, 9]
    none = np.zeros(0, np.int32)
    empty, zero = pack_inputs(none, np.zeros(0, np.float64), none, none, none)
    assert zero == (0, 0, 0, 0)
    assert empty.size == 0


def test_ring_layout_round_trips_the_reference_state() -> None:
    g, _ = random_graph(200, 20, seed=3)
    e = NumpyLIFEngine(build_network(g, UnknownSignPolicy.ERROR, 1.5), DISC)
    e.reset(0)
    rng = np.random.default_rng(1)
    for _ in range(37):  # more than one delay, so every reference slot is in use
        e.step(StepDrive(forced_idx=np.flatnonzero(rng.random(200) < 0.05).astype(np.int32)))
    s = e.state()
    D = DISC.delay_steps
    assert sum(a.size for a in s.in_flight) > 0
    slots = state_to_ring(s.step, D, s.in_flight)
    assert len(slots) == D + 1
    assert slots[s.step % (D + 1)].size == 0  # the slot the next step writes is free
    back = ring_to_state(s.step, D, slots)
    assert all(np.array_equal(a, b) for a, b in zip(back, s.in_flight, strict=True))


def test_ring_layout_before_the_first_delay() -> None:
    D = DISC.delay_steps
    slots = [np.zeros(0, np.int32) for _ in range(D + 1)]
    slots[1] = np.array([5, 2], np.int32)  # emitted at step 1
    state = ring_to_state(3, D, slots)
    assert state[1].tolist() == [2, 5]
    assert sum(a.size for a in state) == 2


def test_integer_weights_are_detected() -> None:
    g, _ = random_graph(100, 10, seed=2)
    net = build_network(g, UnknownSignPolicy.ERROR, 0.275)
    assert net.signed_count is not None
    assert net.weight_scale_mV == 0.275
    assert network_supports_fused(net)
    bent = net.weight_mV.copy()
    bent[0] = np.nextafter(bent[0], np.inf)
    assert not network_supports_fused(dataclasses.replace(net, weight_mV=bent))
    assert not network_supports_fused(dataclasses.replace(net, signed_count=None))


def test_fused_status_explains_why_not() -> None:
    status = fused_status()
    if status["available"]:
        pytest.skip("this host can run the fused engine; tests/gpu covers it")
    assert isinstance(status["reason"], str)
    assert status["reason"]


def _cfg(tiny_dir: Path, **changes: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "fused_cfg_test",
        "dataset": {
            "kind": "csv_fixture",
            "neurons": str(tiny_dir / "neurons.csv"),
            "edges": str(tiny_dir / "edges.csv"),
            "duplicate_rule": "sum",
            "dataset_id": "synthetic:tiny:v1",
        },
        "backend": "numpy",
        "dt_ms": 0.1,
        "steps": 100,
        "seed": 1,
        "model": {"parameter_set": "shiu2024_code_v1"},
        "synapse_sign": {"source": "graph", "unknown_sign_policy": "exclude"},
        "populations": {"a": {"body_ids": [900001]}},
        "stimuli": [],
        "interventions": [],
        "recording": {"population_rate_bin_ms": 10.0},
    }
    cfg.update(copy.deepcopy(changes))
    return cfg


def test_engine_impl_setting(tiny_dir: Path) -> None:
    assert parse_config(_cfg(tiny_dir)).engine_impl == "auto"
    c = parse_config(_cfg(tiny_dir, backend="torch-cuda", engine_impl="fused"))
    assert c.engine_impl == "fused"
    assert yaml.safe_load(c.to_yaml())["engine_impl"] == "fused"
    assert "engine_impl" not in yaml.safe_load(parse_config(_cfg(tiny_dir)).to_yaml())
    with pytest.raises(ConfigError, match="engine_impl: 'fused' applies to backend"):
        parse_config(_cfg(tiny_dir, engine_impl="fused"))
    with pytest.raises(ConfigError, match="engine_impl: 'turbo' is not one of"):
        parse_config(_cfg(tiny_dir, backend="torch-cuda", engine_impl="turbo"))


def test_explicit_fused_without_a_gpu_is_a_run_error(tiny_dir: Path, tmp_path: Path) -> None:
    from flylab.experiments.runner import RunError, run_experiment

    if fused_status()["available"]:
        pytest.skip("this host can run the fused engine")
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(_cfg(tiny_dir, backend="torch-cuda", engine_impl="fused")))
    with pytest.raises(RunError):
        run_experiment(path, tmp_path / "runs", "x", progress=None, validate_only=True)
