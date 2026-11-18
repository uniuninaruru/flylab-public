"""`flylab parity`: the config's backend against the NumPy reference, in lockstep (spec 7.4).

Spike-by-spike policy: `flylab.neural.parity`. After every individually justified threshold
flip the tested engine is resynchronized to the reference; comparison continues at every step.
Population counts and a reference run with the next seed are supplementary descriptions, not
an independent gate or a statistical test. Test counts describe the resynchronized trajectory.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable
from pathlib import Path

from flylab import SCHEMA_VERSION
from flylab.experiments.benchmark import BenchmarkError, check_tag
from flylab.experiments.config import ExperimentConfig
from flylab.experiments.runner import _engine, _process_globals
from flylab.experiments.workload import load_workload
from flylab.neural.engine import NeuralStep
from flylab.neural.parity import lockstep, validate_margin
from flylab.provenance.environment import capture_environment
from flylab.provenance.hardware import capture_hardware

PARITY_KIND = "flylab_backend_parity"
REFERENCE_BACKENDS = ("numpy", "torch-cpu")


class _PopCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.spikes = 0

    def record(self, step: NeuralStep) -> None:
        self.spikes += int(step.spike_idx.size)
        for name, c in step.population_spike_counts.items():
            self.counts[name] = self.counts.get(name, 0) + c


def run_parity(
    config: ExperimentConfig | str | Path,
    *,
    reference: str = "numpy",
    steps: int | None = None,
    backend: str | None = None,
    dtype: str | None = None,
    margin_mV: float | None = None,
    state_every: int = 100,
    seed_control: bool = True,
    max_edges_per_chunk: int | None = None,
    engine_impl: str | None = None,
    tag: str | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    if reference not in REFERENCE_BACKENDS:
        raise BenchmarkError(f"reference must be one of {REFERENCE_BACKENDS}")
    if state_every < 1:
        raise BenchmarkError("state_every must be >= 1")
    if margin_mV is not None:
        try:
            validate_margin(margin_mV)
        except ValueError as exc:
            raise BenchmarkError(str(exc)) from exc
    check_tag(tag)
    say = progress or (lambda _msg: None)
    started = dt.datetime.now(dt.UTC)
    wl = load_workload(config, steps=steps, backend=backend, dtype=dtype, engine_impl=engine_impl)
    cfg = wl.config
    if cfg.backend == reference:
        raise BenchmarkError(f"the backend under test is the reference {reference!r} itself")
    ref_cfg = dataclasses.replace(cfg, backend=reference)
    say(f"[parity] {cfg.backend} vs {reference} ({cfg.dtype}), {cfg.steps} steps, "
        f"{wl.network.n_neurons:,} neurons, {wl.network.n_edges:,} edges")  # fmt: skip
    every = max(1, cfg.steps // 10)

    def tick(k: int) -> None:
        if k % every == 0:
            say(f"[parity] step {k}/{cfg.steps}")

    with _process_globals(cfg):
        test = _engine(cfg, wl.prep, wl.network, wl.pops, max_edges_per_chunk)
        ref = _engine(ref_cfg, wl.prep, wl.network, wl.pops)
        ref.reset(cfg.seed)
        test.reset(cfg.seed)
        rep = lockstep(ref, test, cfg.steps, wl.schedule(), margin_mV, state_every, tick)
        engines = {"reference": ref.describe(), "test": test.describe()}
        control: dict[str, object] | None = None
        if seed_control and rep.divergence is not None:
            say(f"[parity] diverged at step {rep.divergence.step}; running the seed control")
            counter = _PopCounter()
            ref.reset(cfg.seed + 1)
            ref.simulate(cfg.steps, counter, wl.schedule(cfg.seed + 1))
            control = {"seed": cfg.seed + 1, "n_spikes": counter.spikes,
                        "population_spikes": counter.counts}  # fmt: skip

    pops: dict[str, dict[str, object]] = {}
    for name, r in rep.population_spikes_reference.items():
        t = rep.population_spikes_test.get(name, 0)
        row: dict[str, object] = {"reference": r, "test": t, "abs_diff": abs(t - r)}
        if control is not None:
            alt = int(control["population_spikes"].get(name, 0))  # type: ignore[attr-defined]
            row |= {
                "reference_next_seed": alt,
                "seed_to_seed_abs_diff": abs(alt - r),
                "within_seed_to_seed_diff": abs(t - r) <= abs(alt - r),
            }
        pops[name] = row
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": PARITY_KIND,
        "tag": tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "status": "PASS" if rep.ok else "FAIL",
        "workload": {
            "config_path": wl.config_path,
            "config_sha256": wl.config_sha256,
            "experiment_id": cfg.experiment_id,
            "dtype": cfg.dtype,
            "dt_ms": cfg.dt_ms,
            "steps": cfg.steps,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "dataset_id": wl.data.graph.dataset_id,
            "biological_data": wl.data.info.get("biological_data"),
            "n_neurons": wl.network.n_neurons,
            "n_edges": wl.network.n_edges,
            "network_fingerprint_sha256": wl.network.fingerprint(),
        },
        "policy": "docs/MODEL_ASSUMPTIONS.md section 2.3 (flylab.neural.parity)",
        "lockstep": rep.to_dict(),
        "populations": pops,
        "seed_control": control,
        "engines": engines,
        "environment": capture_environment(),
        "hardware": capture_hardware(),
    }
