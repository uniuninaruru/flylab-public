"""CUDA LIF engine vs the NumPy reference. RTX host only: `pytest tests/gpu --run-gpu`.

Tolerance policy: docs/MODEL_ASSUMPTIONS.md section 2.3 and `flylab.neural.parity`. Every
operation except the float64 additions inside `index_add_` is the same IEEE-754 operation on
both devices, so states may differ only by reordered float64 sums; spikes must be identical
unless a flip is explained by a reference potential within the margin of threshold.
A skip here is not a pass: GPU status stays UNVERIFIED until the host log exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import StepDrive, StepInterventions
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize
from flylab.neural.parity import lockstep

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
from flylab.neural.lif_torch import TorchLIFEngine  # noqa: E402

DISC = discretize(SHIU2024_CODE_V1, 0.1)
REPO = Path(__file__).resolve().parents[2]


class ListSchedule:
    def __init__(self, inputs: list[tuple[StepDrive | None, StepInterventions | None]]) -> None:
        self._inputs = inputs

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        return self._inputs[step]


def _net(n: int = 2000, seed: int = 1) -> SynapticNetwork:
    g, _ = random_graph(n, 30, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, 0.6)


def _inputs(n: int, steps: int, seed: int) -> ListSchedule:
    """Noisy drive to every neuron, sparse forced spikes, a block + clamp window."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 14, n)
    out: list[tuple[StepDrive | None, StepInterventions | None]] = []
    for k in range(steps):
        drive = StepDrive(
            current_idx=np.arange(n, dtype=np.int32),
            current_mV=base + rng.normal(0, 2, n),
            forced_idx=np.flatnonzero(rng.random(n) < 0.002).astype(np.int32),
        )
        iv = None
        if 500 <= k < 900:
            iv = StepInterventions(
                block_output_idx=np.arange(0, 40, dtype=np.int32),
                clamp_idx=np.arange(40, 60, dtype=np.int32),
            )
        out.append((drive, iv))
    return ListSchedule(out)


@pytest.fixture(autouse=True)
def _cuda() -> None:
    assert torch.cuda.is_available(), "CUDA unavailable"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_cuda_matches_numpy_in_lockstep(dtype: str, deterministic: bool) -> None:
    net = _net()
    pops = {"all": np.arange(net.n_neurons), "first100": np.arange(100)}
    ref = NumpyLIFEngine(net, DISC, dtype, pops)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        test = TorchLIFEngine(net, DISC, dtype, pops, device="cuda")
        assert test.backend == "torch-cuda"
        ref.reset(1)
        test.reset(1)
        rep = lockstep(ref, test, 1500, _inputs(net.n_neurons, 1500, 5), state_every=50)
    finally:
        torch.use_deterministic_algorithms(previous)
    print(rep.to_dict())  # the host log keeps the measured differences
    assert rep.n_spikes_reference > 2000, "the comparison needs substantial recurrent activity"
    assert rep.ok, rep.to_dict()
    assert rep.state_checks >= 1


def test_deterministic_mode_repeats_bitwise() -> None:
    net = _net(seed=3)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        states = []
        for _ in range(2):
            e = TorchLIFEngine(net, DISC, "float32", device="cuda")
            e.reset(2)
            sched = _inputs(net.n_neurons, 800, 9)
            for k in range(800):
                e.step(*sched.inputs(k))
            states.append(e.state())
    finally:
        torch.use_deterministic_algorithms(previous)
    a, b = states
    assert np.array_equal(a.v, b.v)
    assert np.array_equal(a.g, b.g)
    assert np.array_equal(a.last_spike, b.last_spike)


def test_chunked_arrivals_match_on_cuda() -> None:
    net = _net(seed=4)
    whole = TorchLIFEngine(net, DISC, device="cuda")
    parts = TorchLIFEngine(net, DISC, device="cuda", max_edges_per_chunk=97)
    whole.reset(1)
    parts.reset(1)
    rep = lockstep(whole, parts, 800, _inputs(net.n_neurons, 800, 6))
    assert rep.ok, rep.to_dict()


def test_cuda_checkpoint_resumes_on_numpy(tmp_path: Path) -> None:
    net = _net(seed=5)
    sched = _inputs(net.n_neurons, 600, 7)
    gpu = TorchLIFEngine(net, DISC, device="cuda")
    gpu.reset(4)
    for k in range(300):
        gpu.step(*sched.inputs(k))
    gpu.checkpoint(tmp_path / "ck.npz")
    cpu = NumpyLIFEngine(net, DISC)
    cpu.restore(tmp_path / "ck.npz")
    assert cpu.current_step == gpu.current_step == 300
    rep = lockstep(cpu, gpu, 300, sched)  # the schedule is indexed by absolute step
    assert rep.ok, rep.to_dict()


def test_device_memory_and_settings_are_reported() -> None:
    net = _net(n=500)
    e = TorchLIFEngine(net, DISC, "float32", device="cuda")
    e.reset(0)
    d = e.describe()
    assert str(d["device"]).startswith("cuda")
    assert d["cuda_capability"] == "12.0"
    assert "sm_120" in d["cuda_arch_list"]  # type: ignore[operator]
    e.reset_peak_device_memory()
    e.simulate(50)
    mem = e.device_memory_stats()
    assert mem is not None
    assert mem["allocated_bytes"] >= e.retained_bytes()["edges"]
    assert mem["max_reserved_bytes"] >= mem["max_allocated_bytes"] > 0
