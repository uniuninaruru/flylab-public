"""Fused (Triton) CUDA LIF engine vs the NumPy reference. RTX host only: `--run-gpu`.

Policy: MODEL_ASSUMPTIONS 2.3 / `flylab.neural.parity`. Arrivals are exact integer sums scaled
once by w_syn, so states may differ from the reference by float64 rounding when a neuron receives
several arrivals in a step; spikes must match except for justified near-threshold flips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import NeuralStep, StepDrive, StepInterventions
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize
from flylab.neural.parity import lockstep

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from flylab.neural.lif_fused import FusedCudaLIFEngine, fused_status  # noqa: E402
from flylab.neural.lif_torch import TorchLIFEngine  # noqa: E402

DISC = discretize(SHIU2024_CODE_V1, 0.1)


class ListSchedule:
    def __init__(self, inputs: list[tuple[StepDrive | None, StepInterventions | None]]) -> None:
        self._inputs = inputs

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        return self._inputs[step]


def _net(n: int = 2000, seed: int = 1, w: float = 0.6) -> SynapticNetwork:
    g, _ = random_graph(n, 30, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, w)


def _inputs(n: int, steps: int, seed: int) -> ListSchedule:
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 14, n)
    out: list[tuple[StepDrive | None, StepInterventions | None]] = []
    for k in range(steps):
        idx = np.arange(n, dtype=np.int32)
        if k % 7 == 3:  # duplicate entries: drive must add like np.add.at
            idx = np.concatenate([idx, idx[:50]])
        drive = StepDrive(
            current_idx=idx,
            current_mV=np.concatenate([base, base[:50]])[: idx.size] + rng.normal(0, 2, idx.size),
            forced_idx=np.flatnonzero(rng.random(n) < 0.002).astype(np.int32),
        )
        iv = None
        if 500 <= k < 900:
            iv = StepInterventions(
                block_output_idx=np.arange(0, 40, dtype=np.int32),
                clamp_idx=np.arange(40, 60, dtype=np.int32),
            )
        out.append((drive if k % 11 else None, iv))  # some steps without any drive
    return ListSchedule(out)


class _Keep:
    def __init__(self) -> None:
        self.rows: list[tuple[int, list[int], list[bool], list[bool], int]] = []

    def record(self, s: NeuralStep) -> None:
        self.rows.append((s.step, s.spike_idx.tolist(), s.forced.tolist(),
                          s.propagated.tolist(), s.n_forced_dropped))  # fmt: skip


@pytest.fixture(autouse=True)
def _available() -> None:
    status = fused_status()
    assert status["available"], status


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_fused_matches_numpy_in_lockstep(dtype: str) -> None:
    net = _net()
    pops = {"all": np.arange(net.n_neurons), "first100": np.arange(100)}
    ref = NumpyLIFEngine(net, DISC, dtype, pops)
    test = FusedCudaLIFEngine(net, DISC, dtype, pops)
    ref.reset(1)
    test.reset(1)
    rep = lockstep(ref, test, 1500, _inputs(net.n_neurons, 1500, 5), state_every=50)
    print(rep.to_dict())  # the host log keeps the measured differences
    assert rep.n_spikes_reference > 2000, "the comparison needs substantial recurrent activity"
    assert rep.ok, rep.to_dict()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_batched_simulate_equals_single_steps(dtype: str) -> None:
    """simulate() keeps up to `batch_steps` steps on the GPU; results must not depend on it."""
    net = _net(seed=2)
    runs = []
    for batch in (1, 7, 32):
        e = FusedCudaLIFEngine(net, DISC, dtype, {"all": np.arange(net.n_neurons)},
                               batch_steps=batch)  # fmt: skip
        e.reset(3)
        rec = _Keep()
        summary = e.simulate(1000, rec, _inputs(net.n_neurons, 1000, 8))
        runs.append((rec.rows, e.state(), summary))
    (r1, s1, m1), *others = runs
    assert sum(len(x[1]) for x in r1) > 1000
    for r, s, m in others:
        assert r == r1
        assert np.array_equal(s.v, s1.v)
        assert np.array_equal(s.g, s1.g)
        assert m.n_spikes == m1.n_spikes


def test_fused_is_deterministic_and_close_to_eager_cuda() -> None:
    net = _net(seed=3)
    states = []
    for _ in range(2):
        e = FusedCudaLIFEngine(net, DISC, "float32")
        e.reset(2)
        e.simulate(800, None, _inputs(net.n_neurons, 800, 9))
        states.append(e.state())
    assert np.array_equal(states[0].v, states[1].v)
    assert np.array_equal(states[0].g, states[1].g)
    eager = TorchLIFEngine(net, DISC, "float32", device="cuda")
    fused = FusedCudaLIFEngine(net, DISC, "float32")
    eager.reset(2)
    fused.reset(2)
    rep = lockstep(eager, fused, 800, _inputs(net.n_neurons, 800, 9))
    assert rep.ok, rep.to_dict()


def test_checkpoints_move_between_fused_and_numpy(tmp_path: Path) -> None:
    net = _net(seed=5)
    sched = _inputs(net.n_neurons, 600, 7)
    gpu = FusedCudaLIFEngine(net, DISC)
    gpu.reset(4)
    gpu.simulate(300, None, sched)
    gpu.checkpoint(tmp_path / "a.npz")
    cpu = NumpyLIFEngine(net, DISC)
    cpu.restore(tmp_path / "a.npz")
    rep = lockstep(cpu, gpu, 150, sched)
    assert rep.ok, rep.to_dict()
    cpu.checkpoint(tmp_path / "b.npz")  # and back: NumPy state into a fresh fused engine
    gpu2 = FusedCudaLIFEngine(net, DISC)
    gpu2.restore(tmp_path / "b.npz")
    assert gpu2.current_step == 450
    rep2 = lockstep(cpu, gpu2, 150, sched)
    assert rep2.ok, rep2.to_dict()


def test_description_and_memory() -> None:
    net = _net(n=500)
    e = FusedCudaLIFEngine(net, DISC, "float32")
    e.reset(0)
    d = e.describe()
    assert d["impl"] == "fused"
    assert d["backend"] == "torch-cuda"
    assert d["fp_fusion"] is False
    assert e.retained_bytes()["edges"] == 8 * (net.n_neurons + 1) + 8 * net.n_edges
    e.reset_peak_device_memory()
    e.simulate(50)
    mem = e.device_memory_stats()
    assert mem is not None
    assert mem["max_allocated_bytes"] > 0
