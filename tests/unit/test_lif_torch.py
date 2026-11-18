"""PyTorch CPU engine vs the NumPy reference.

Skipped when torch is not installed, unless FLYLAB_REQUIRE_TORCH=1 (set by the dedicated CI
workflow), in which case a missing torch is a failure, not a skip.

Tolerances. Both engines evaluate the same IEEE-754 operations (+, -, *) in the same order with
propagator constants rounded once to the state dtype, and both sum synaptic arrivals in float64
before one rounding to the state dtype. Bitwise equality is therefore expected and is what these
tests observed. The state tolerances below only allow for a different float64 summation order
inside `index_add_` on another platform: with at most ~10^3 terms of |w| <= 100 mV the float64
sum error is < 1e-10 mV (atol 1e-9 for float64 state); after rounding to float32 such an error
can flip at most one unit in the last place of |g| < 128 mV, i.e. 2**-16 mV, and propagates into
v within a few ulps (atol 4 * 2**-16 ~ 6.1e-5 mV for float32 state). Spike trains are compared
exactly: a flip would need v within that margin of threshold on the tested inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import NeuralStep, StepDrive, StepInterventions
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize

if os.environ.get("FLYLAB_REQUIRE_TORCH") == "1":
    import torch
else:
    torch = pytest.importorskip("torch")

from flylab.neural.lif_torch import (
    CudaUnavailableError,
    TorchLIFEngine,
    TorchUnavailableError,
    check_backend,
    cuda_status,
    torch_settings,
)

DISC = discretize(SHIU2024_CODE_V1, 0.1)
ATOL = {"float64": 1e-9, "float32": 4 * 2.0**-16}


def _net(seed: int = 1, n: int = 400) -> SynapticNetwork:
    g, _ = random_graph(n, 30, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, 0.6)


def _inputs(n: int, steps: int, seed: int) -> list[tuple[StepDrive, StepInterventions | None]]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 14, n)
    out: list[tuple[StepDrive, StepInterventions | None]] = []
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
    return out


def _spikes(steps: list[NeuralStep]) -> list[tuple[int, int, bool, bool]]:
    return [
        (s.step, int(i), bool(f), bool(p))
        for s in steps
        for i, f, p in zip(s.spike_idx, s.forced, s.propagated, strict=True)
    ]


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_numpy_and_torch_agree(dtype: str) -> None:
    net = _net()
    inputs = _inputs(net.n_neurons, 1500, seed=5)
    results = {}
    for cls in (NumpyLIFEngine, TorchLIFEngine):
        e = cls(net, DISC, dtype)
        e.reset(1)
        results[cls.backend] = (_spikes([e.step(d, iv) for d, iv in inputs]), e.state())
    (sp_np, st_np), (sp_t, st_t) = results["numpy"], results["torch-cpu"]
    assert len(sp_np) > 2000, "the comparison needs substantial recurrent activity"
    assert any(not p for *_, p in sp_np), "the blocked window must produce unpropagated spikes"
    assert sp_np == sp_t
    np.testing.assert_allclose(st_t.v, st_np.v, rtol=0, atol=ATOL[dtype])
    np.testing.assert_allclose(st_t.g, st_np.g, rtol=0, atol=ATOL[dtype])
    np.testing.assert_array_equal(st_t.last_spike, st_np.last_spike)
    assert st_t.v.dtype == np.dtype(dtype)


def test_closed_form_first_spike_and_isi() -> None:
    g, _ = random_graph(1, 0, seed=0)
    e = TorchLIFEngine(build_network(g, UnknownSignPolicy.ERROR, 1.0), DISC)
    e.reset(0)
    drive = StepDrive(np.array([0], np.int32), np.array([20.0]))
    spikes = [s.step for s in (e.step(drive) for _ in range(500)) if s.spike_idx.size]
    assert spikes[:2] == [86, 194]  # same as NumPy reference and Brian2 (MODEL_ASSUMPTIONS 3.2)


def test_checkpoint_is_backend_neutral(tmp_path: Path) -> None:
    net = _net(n=200)
    inputs = _inputs(net.n_neurons, 300, seed=6)
    ref = NumpyLIFEngine(net, DISC)
    ref.reset(3)
    full = [ref.step(d, iv) for d, iv in inputs]

    first = NumpyLIFEngine(net, DISC)
    first.reset(3)
    for d, iv in inputs[:120]:
        first.step(d, iv)
    first.checkpoint(tmp_path / "ck.npz")
    resumed = TorchLIFEngine(net, DISC)
    resumed.restore(tmp_path / "ck.npz")
    rest = [resumed.step(d, iv) for d, iv in inputs[120:]]
    assert _spikes(rest) == _spikes(full[120:])


def test_cuda_without_a_device_fails_clearly() -> None:
    if torch.cuda.is_available():
        pytest.skip("this host has CUDA; tests/gpu covers the CUDA engine")
    with pytest.raises(CudaUnavailableError, match="needs a CUDA device"):
        TorchLIFEngine(_net(n=10), DISC, device="cuda")
    with pytest.raises(TorchUnavailableError, match="needs a CUDA device"):
        check_backend("torch-cuda")
    assert cuda_status()["available"] is False


def test_device_and_chunk_arguments_are_validated() -> None:
    with pytest.raises(ValueError, match="device must be"):
        TorchLIFEngine(_net(n=10), DISC, device="mps")
    with pytest.raises(ValueError, match="max_edges_per_chunk"):
        TorchLIFEngine(_net(n=10), DISC, max_edges_per_chunk=0)


def test_settings_and_memory_accounting_are_recorded() -> None:
    s = torch_settings()
    assert s["torch_version"] == torch.__version__
    assert isinstance(s["num_threads"], int)
    net = _net(n=50)
    e = TorchLIFEngine(net, DISC, "float32")
    assert e.describe()["device"] == "cpu"
    assert e.backend == "torch-cpu"
    assert e.device_memory_stats() is None
    e.reset(0)
    got = e.retained_bytes()
    # int64 row pointers + int32 targets + float64 weights; float32 v, g; int64 last; bool mask
    assert got["edges"] == 8 * (net.n_neurons + 1) + 12 * net.n_edges
    assert got["state"] == (4 + 4 + 8 + 1) * net.n_neurons
    assert got["in_flight"] == 0


@pytest.mark.parametrize("chunk", [1, 7, 1000])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_edge_chunking_does_not_change_results(chunk: int, dtype: str) -> None:
    """Chunked arrivals add the same terms per target in the same order: bitwise equal."""
    net = _net(seed=2, n=300)
    inputs = _inputs(net.n_neurons, 700, seed=8)
    whole = TorchLIFEngine(net, DISC, dtype)
    parts = TorchLIFEngine(net, DISC, dtype, max_edges_per_chunk=chunk)
    for e in (whole, parts):
        e.reset(1)
    sw = [whole.step(d, iv) for d, iv in inputs]
    sp = [parts.step(d, iv) for d, iv in inputs]
    assert sum(s.spike_idx.size for s in sw) > 500, "needs arrivals larger than the chunks"
    assert _spikes(sw) == _spikes(sp)
    a, b = whole.state(), parts.state()
    assert np.array_equal(a.v, b.v)
    assert np.array_equal(a.g, b.g)
