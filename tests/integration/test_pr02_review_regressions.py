"""Regression tests for the PR-02 code review findings (uniuninaruru/flylab#15)."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import yaml

from flylab.data.processed import write_processed
from flylab.data.synthetic import random_graph
from flylab.experiments.artifacts import RunRecorder
from flylab.experiments.config import DatasetConfig
from flylab.experiments.dataset import DatasetError, load_dataset
from flylab.experiments.runner import PreparedRun, RunError, run_experiment
from flylab.neural.engine import NeuralStep
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize
from flylab.provenance.labels import Evidence, Labeled


def _processed(tmp: Path, dataset_id: str | None, manifest_extra: dict[str, Any]) -> Path:
    g, r = random_graph(40, 3, seed=2)
    if dataset_id is not None:
        g = dataclasses.replace(g, dataset_id=dataset_id)
    return write_processed(tmp / "proc", g, r, {"type": pa.array(["x"] * 40)}, manifest_extra)


@pytest.mark.parametrize(
    ("dataset_id", "extra", "expected"),
    [
        (None, {"release": None}, False),  # synthetic graph, synthetic import
        (None, {}, False),  # synthetic id, older manifest without `release`
        ("male-cns:v1.0", {"release": {"dataset_id": "male-cns:v1.0"}}, True),
        ("male-cns:v1.0", {}, True),  # manifest written before `release` existed
        ("male-cns:v1.0", {"release": None}, False),  # registry check was skipped
    ],
)
def test_biological_data_comes_from_the_processed_manifest(
    tmp_path: Path, dataset_id: str | None, extra: dict[str, Any], expected: bool
) -> None:
    proc = _processed(tmp_path, dataset_id, extra)
    info = load_dataset(DatasetConfig(kind="processed", path=str(proc))).info
    assert info["biological_data"] is expected


def test_loader_requires_the_files_it_reads_to_be_listed(tmp_path: Path) -> None:
    proc = _processed(tmp_path, None, {})
    lines = (proc / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not ln.endswith("graph.npz")]
    (proc / "checksums.sha256").write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="does not list"):
        load_dataset(DatasetConfig(kind="processed", path=str(proc)))


def _spike(rec: RunRecorder, step: int) -> None:
    rec.record(
        NeuralStep(
            step=step, t_ms=step * 0.1, dt_ms=0.1, spike_idx=np.array([0], np.int32),
            forced=np.array([False]), propagated=np.array([True]), n_forced_dropped=0,
            population_spike_counts={},
        )
    )  # fmt: skip


def test_latency_uses_the_first_spike_after_the_reference(tmp_path: Path) -> None:
    pops = {"p": np.array([0], dtype=np.int32)}
    rec = RunRecorder(tmp_path, np.arange(3, dtype=np.int64), pops, 200, 10, False,
                      reference_step=100)  # fmt: skip
    _spike(rec, 5)  # spontaneous, before the stimulus at 10 ms
    _spike(rec, 150)  # first spike after the stimulus
    _spike(rec, 180)
    lat = rec.population_metrics(0.1, 200, reference_ms=10.0)["p"]["first_spike_latency_ms"]
    assert isinstance(lat, dict)
    assert lat["n_neurons"] == 1
    assert lat["median"] == pytest.approx(5.0)
    assert rec.population_metrics(0.1, 200, 10.0)["p"]["fraction_spiking"] == 1.0
    with pytest.raises(ValueError, match="reference_ms"):
        rec.population_metrics(0.1, 200, reference_ms=3.0)


def _tiny(repo_root: Path, tmp_path: Path, **changes: Any) -> Path:
    cfg = yaml.safe_load((repo_root / "configs" / "tiny_fixture.yaml").read_text())
    cfg.update(changes)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def at_repo_root(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not (repo_root / "configs").is_dir():
        pytest.skip("configs/ is not part of this checkout (e.g. the CPU Docker image)")
    monkeypatch.chdir(repo_root)  # the example's fixture paths are repo-relative
    return repo_root


def test_validate_only_checks_torch_availability(
    at_repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flylab.neural.lif_torch as lt

    monkeypatch.setattr(lt, "torch_available", lambda: False)
    path = _tiny(at_repo_root, tmp_path, backend="torch-cpu")
    with pytest.raises(RunError, match="needs PyTorch"):
        run_experiment(path, tmp_path / "runs", validate_only=True, progress=None)


def test_run_restores_global_rng_state_and_labels_process_peak(
    at_repo_root: Path, tmp_path: Path
) -> None:
    np.random.seed(1234)  # noqa: NPY002
    random.seed(99)
    np_before = np.random.get_state()  # noqa: NPY002
    py_before = random.getstate()
    [res] = run_experiment(_tiny(at_repo_root, tmp_path), tmp_path / "runs", "t", progress=None)
    np_after = np.random.get_state()  # noqa: NPY002
    assert py_before == random.getstate()
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    bench = json.loads((res.run_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert "peak_rss_bytes" not in bench["measured"]
    assert bench["measured"]["process_peak_rss_bytes"] > 0
    assert "not attributable" in bench["notes"]["process_peak_rss_bytes"]


def test_prepared_runs_do_not_hold_a_network() -> None:
    assert "network" not in {f.name for f in dataclasses.fields(PreparedRun)}


def test_failed_restore_leaves_the_engine_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    g, _ = random_graph(30, 3, seed=4)
    from flylab.data.graph import UnknownSignPolicy

    net = build_network(g, UnknownSignPolicy.ERROR, 0.275)
    disc = discretize(SHIU2024_CODE_V1, 0.1)
    eng = NumpyLIFEngine(net, disc)
    eng.reset(1)
    eng.simulate(5)
    ckpt = tmp_path / "c.npz"
    eng.checkpoint(ckpt)
    eng.simulate(3)
    assert eng.current_step == 8

    def boom(state: object) -> None:
        raise ValueError("rejected")

    monkeypatch.setattr(eng, "_load_state", boom)
    with pytest.raises(ValueError, match="rejected"):
        eng.restore(ckpt)
    assert eng.current_step == 8


def _with_tau_syn(tau: float) -> Any:
    return SHIU2024_CODE_V1.with_overrides(
        {"tau_syn": Labeled(tau, Evidence.ASSUMED, rationale="numerics test", units="ms")}
    )


def test_propagator_b_is_continuous_at_equal_time_constants() -> None:
    tau_m = SHIU2024_CODE_V1.values()["tau_m"]
    exact = discretize(_with_tau_syn(tau_m), 0.1).b
    for eps in (1e-12, 1e-9, 1e-6):
        near = discretize(_with_tau_syn(tau_m * (1 + eps)), 0.1).b
        assert near == pytest.approx(exact, rel=10 * eps + 1e-14)


def test_propagator_b_unchanged_for_the_shiu_preset() -> None:
    v = SHIU2024_CODE_V1.values()
    dt, tm, ts = 0.1, v["tau_m"], v["tau_syn"]
    general = ts / (ts - tm) * (math.exp(-dt / ts) - math.exp(-dt / tm))
    assert discretize(SHIU2024_CODE_V1, dt).b == general  # bit-for-bit
