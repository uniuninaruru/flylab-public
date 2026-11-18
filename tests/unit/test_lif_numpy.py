"""Hand-checkable tests of the NumPy reference engine (docs/MODEL_ASSUMPTIONS.md section 2.1).

Expected values are closed-form solutions of the scheme, not outputs of the code under test.
With the Shiu et al. preset at dt = 0.1 ms: a = exp(-dt/20), d = exp(-dt/5), R = 22, D = 18.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from flylab.data.csv_io import read_edges_csv, read_neurons_csv
from flylab.data.graph import DuplicateRule, UnknownSignPolicy, build_graph
from flylab.neural.engine import EngineInputError, NeuralStep, StepDrive, StepInterventions
from flylab.neural.lif_numpy import NumpyLIFEngine, outgoing_edges
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, LIFParameters, discretize
from flylab.provenance.labels import Evidence, Labeled

DT = 0.1
V0, VTH = -52.0, -45.0
A, D_ = math.exp(-DT / 20), math.exp(-DT / 5)
R, DELAY = 22, 18


def _net(n: int, edges: list[tuple[int, int, int, int]], w_syn: float = 1.0) -> SynapticNetwork:
    """edges: (pre_idx, post_idx, synapse_count, sign); body ids are 1..n."""
    ids = np.arange(1, n + 1)
    pre = [ids[e[0]] for e in edges]
    post = [ids[e[1]] for e in edges]
    g, _ = build_graph(
        ids, pre, post, [e[2] for e in edges], [e[3] for e in edges], dataset_id="test"
    )
    return build_network(g, UnknownSignPolicy.ERROR, w_syn)


def _engine(
    net: SynapticNetwork, params: LIFParameters = SHIU2024_CODE_V1, dtype: str = "float64"
) -> NumpyLIFEngine:
    e = NumpyLIFEngine(net, discretize(params, DT), dtype)
    e.reset(0)
    return e


def _drive(idx: list[int], mv: list[float]) -> StepDrive:
    return StepDrive(current_idx=np.array(idx, dtype=np.int32), current_mV=np.array(mv))


def _forced(idx: list[int]) -> StepDrive:
    return StepDrive(forced_idx=np.array(idx, dtype=np.int32))


def _spike_steps(steps: list[NeuralStep], neuron: int) -> list[int]:
    return [s.step for s in steps if neuron in s.spike_idx]


def _k_star(e: float) -> int:
    """Integrations from rest needed to cross threshold under constant drive e."""
    return next(k for k in range(1, 100_000) if V0 + e * (1 - A**k) > VTH)


def test_rest_is_a_fixed_point() -> None:
    e = _engine(_net(3, []))
    for _ in range(500):
        out = e.step()
        assert out.spike_idx.size == 0
    s = e.state()
    np.testing.assert_array_equal(s.v, np.full(3, V0))
    np.testing.assert_array_equal(s.g, np.zeros(3))


def test_subthreshold_constant_drive_matches_closed_form() -> None:
    e = _engine(_net(1, []))
    drive = 5.0  # steady state V0 + 5 = -47 mV, below threshold
    for n in range(2000):
        e.step(_drive([0], [drive]))
        expected = V0 + drive * (1 - A ** (n + 1))
        assert e.state().v[0] == pytest.approx(expected, abs=1e-12)


def test_first_spike_and_isi_match_closed_form_and_brian2() -> None:
    """Closed form: first spike at step k* - 1, ISI R + k* - 1, k* = 87. Brian2 2.10.1 running the
    Shiu et al. equations gave the same 86 and 108 (docs/MODEL_ASSUMPTIONS.md 3.2)."""
    e = _engine(_net(1, []))
    out = [e.step(_drive([0], [20.0])) for _ in range(1000)]
    k = _k_star(20.0)
    assert k == 87
    spikes = _spike_steps(out, 0)
    assert spikes[0] == k - 1 == 86
    assert set(np.diff(spikes)) == {R + k - 1} == {108}
    assert all(not s.forced.any() for s in out)


def test_zero_refractory_period() -> None:
    p = SHIU2024_CODE_V1.with_overrides(
        {"t_ref": Labeled(0.0, Evidence.ASSUMED, rationale="test", units="ms")}
    )
    e = _engine(_net(1, []), p)
    spikes = _spike_steps([e.step(_drive([0], [20.0])) for _ in range(1000)], 0)
    assert set(np.diff(spikes)) == {_k_star(20.0)}  # integration resumes the step after a spike


def test_single_arrival_matches_closed_form_and_delay() -> None:
    w = 10.0  # mV; peak PSP = 0.1575 * w < 7 mV, so no spike
    e = _engine(_net(2, [(0, 1, 1, +1)], w_syn=w))
    e.step(_forced([0]))  # step 0: neuron 0 spikes; its spike arrives in step D
    for n in range(1, DELAY):
        e.step()
        assert e.state().g[1] == 0.0, f"arrived early at step {n}"
    assert e.step().step == DELAY
    for k in range(400):  # k integrations after the arrival step
        if k:
            assert e.step().spike_idx.size == 0
        s = e.state()
        assert s.g[1] == pytest.approx(w * D_**k, abs=1e-12)
        assert s.v[1] == pytest.approx(V0 + w * 5 / (5 - 20) * (D_**k - A**k), abs=1e-12)


def test_inhibitory_and_excitatory_signs() -> None:
    e = _engine(_net(3, [(0, 1, 2, +1), (0, 2, 2, -1)], w_syn=3.0))
    e.step(_forced([0]))
    for _ in range(DELAY + 50):
        e.step()
    s = e.state()
    assert s.v[1] > V0
    assert s.v[2] < V0
    assert s.v[1] - V0 == pytest.approx(V0 - s.v[2], abs=1e-12)  # symmetric


def test_dense_reference_gives_the_same_arrivals(tiny_dir: Path) -> None:
    rows = read_edges_csv(tiny_dir / "edges.csv")
    g, _ = build_graph(
        read_neurons_csv(tiny_dir / "neurons.csv"),
        rows.pre_body_id,
        rows.post_body_id,
        rows.synapse_count,
        rows.sign,
        dataset_id="synthetic:tiny:v1",
        duplicate_rule=DuplicateRule.SUM,
    )
    net = build_network(g, UnknownSignPolicy.AS_EXCITATORY, 0.5)
    w = net.dense_for_tests()
    for pre in range(net.n_neurons):
        eid = outgoing_edges(net.row_ptr, np.array([pre], dtype=np.int32))
        sparse = np.bincount(net.dst[eid], weights=net.weight_mV[eid], minlength=net.n_neurons)
        np.testing.assert_array_equal(sparse, w[:, pre])
    # Hand value from the fixture README: 900002 -> 900004 merged to 5 synapses, sign -1.
    assert w[3, 1] == -2.5


def test_arrivals_at_refractory_or_spiking_neurons_are_discarded() -> None:
    w = 10.0
    e = _engine(_net(2, [(0, 1, 1, +1)], w_syn=w))
    e.step(_forced([0]))  # neuron 0 at step 0 -> arrival at step D
    for _ in range(1, DELAY):
        e.step()
    out = e.step(_forced([1]))  # step D: neuron 1 spikes in the arrival step
    assert out.step == DELAY
    assert list(out.spike_idx) == [1]
    s = e.state()
    assert s.g[1] == 0.0  # arrival discarded (Brian2 conditional write), then reset
    assert s.v[1] == V0

    # Arrival in the middle of the refractory period is discarded too.
    e2 = _engine(_net(2, [(0, 1, 1, +1)], w_syn=w))
    e2.step(_forced([1]))  # step 0: neuron 1 spikes, refractory for steps 1..R-1
    for _ in range(1, 5):
        e2.step()
    e2.step(_forced([0]))  # step 5 -> arrival at step 5 + D = 23 >= R: accepted
    for _ in range(6, 5 + DELAY + 1):
        e2.step()
    assert e2.state().g[1] == pytest.approx(w, abs=1e-12)
    e3 = _engine(_net(2, [(0, 1, 1, +1)], w_syn=w))
    e3.step(_forced([0, 1]))  # both at step 0 -> arrival at step D = 18 < R = 22: discarded
    for _ in range(1, DELAY + 1):
        e3.step()
    assert e3.state().g[1] == 0.0


def test_forced_spikes_respect_refractoriness_and_win_ties() -> None:
    e = _engine(_net(1, []))
    out = [e.step(_forced([0]))]
    out += [e.step(_forced([0])) for _ in range(1, R + 1)]
    spikes = _spike_steps(out, 0)
    assert spikes == [0, R]  # steps 1..R-1 refractory
    assert sum(s.n_forced_dropped for s in out) == R - 1

    # Forced request on a neuron that also crosses threshold: one spike, cause "forced".
    e2 = _engine(_net(1, []))
    for _ in range(86):
        e2.step(_drive([0], [20.0]))
    both = e2.step(StepDrive(np.array([0], np.int32), np.array([20.0]), np.array([0], np.int32)))
    assert both.step == 86
    assert list(both.spike_idx) == [0]
    assert list(both.forced) == [True]


def test_block_output_keeps_dynamics_but_stops_propagation() -> None:
    e = _engine(_net(2, [(0, 1, 1, +1)], w_syn=10.0))
    blocked = StepInterventions(block_output_idx=np.array([0], dtype=np.int32))
    out = e.step(_forced([0]), blocked)
    assert list(out.spike_idx) == [0]
    assert list(out.propagated) == [False]
    for _ in range(DELAY + 10):
        e.step()
    assert e.state().g[1] == 0.0


def test_clamp_stops_integration_spikes_and_input() -> None:
    e = _engine(_net(2, [(0, 1, 1, +1)], w_syn=10.0))
    clamp = StepInterventions(clamp_idx=np.array([1], dtype=np.int32))
    out = [e.step(_forced([0]), clamp)]
    for _ in range(1, 300):
        out.append(
            e.step(
                StepDrive(np.array([1], np.int32), np.array([50.0]), np.array([1], np.int32)), clamp
            )
        )
        assert e.state().v[1] == V0
        assert e.state().g[1] == 0.0
    assert _spike_steps(out, 1) == []
    assert sum(s.n_forced_dropped for s in out) == 299
    # After the window the neuron integrates again.
    after = [e.step(_drive([1], [20.0])) for _ in range(250)]
    first = 300 + _k_star(20.0) - 1
    assert _spike_steps(after, 1) == [first, first + R + _k_star(20.0) - 1]


def test_population_counts_in_step_summary() -> None:
    net = _net(4, [])
    e = NumpyLIFEngine(
        net,
        discretize(SHIU2024_CODE_V1, DT),
        populations={"ab": np.array([0, 1]), "d": np.array([3])},
    )
    e.reset(1)
    out = e.step(_forced([0, 1, 2]))
    assert dict(out.population_spike_counts) == {"ab": 2, "d": 0}
    d = out.to_dict()
    assert d["schema_version"] == 1
    assert d["units"] == {"t_ms": "ms", "dt_ms": "ms", "v": "mV", "g": "mV"}
    assert d["spike_idx"] == [0, 1, 2]


def test_input_validation() -> None:
    e = NumpyLIFEngine(_net(2, []), discretize(SHIU2024_CODE_V1, DT))
    with pytest.raises(RuntimeError, match="reset"):
        e.step()
    e.reset(0)
    with pytest.raises(EngineInputError, match="outside"):
        e.step(_forced([2]))
    with pytest.raises(EngineInputError, match="align"):
        e.step(StepDrive(np.array([0], np.int32), np.array([1.0, 2.0])))
    with pytest.raises(EngineInputError, match="finite"):
        e.step(_drive([0], [float("nan")]))
    with pytest.raises(ValueError, match="dtype"):
        NumpyLIFEngine(_net(2, []), discretize(SHIU2024_CODE_V1, DT), "float16")


def _random_inputs(n: int, steps: int, seed: int) -> list[StepDrive]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 14, n)
    return [
        StepDrive(
            current_idx=np.arange(n, dtype=np.int32),
            current_mV=base + rng.normal(0, 2, n),
            forced_idx=np.flatnonzero(rng.random(n) < 0.002).astype(np.int32),
        )
        for _ in range(steps)
    ]


def _random_net(n: int = 300, seed: int = 1) -> SynapticNetwork:
    from flylab.data.synthetic import random_graph

    g, _ = random_graph(n, 20, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, 0.6)


def _spikes_of(steps: list[NeuralStep]) -> list[tuple[int, int]]:
    return [(s.step, int(i)) for s in steps for i in s.spike_idx]


def test_checkpoint_restore_continues_identically(tmp_path: Path) -> None:
    net = _random_net()
    inputs = _random_inputs(net.n_neurons, 400, seed=3)
    ref = _engine(net)
    full = [ref.step(d) for d in inputs]
    assert len(_spikes_of(full)) > 100  # the test needs recurrent activity

    first = _engine(net)
    for d in inputs[:150]:
        first.step(d)
    ckpt = tmp_path / "state.npz"
    first.checkpoint(ckpt)
    resumed = NumpyLIFEngine(net, discretize(SHIU2024_CODE_V1, DT))
    resumed.restore(ckpt)
    assert resumed.current_step == 150
    rest = [resumed.step(d) for d in inputs[150:]]
    assert _spikes_of(rest) == _spikes_of(full[150:])
    np.testing.assert_array_equal(resumed.state().v, ref.state().v)

    other = NumpyLIFEngine(_random_net(seed=2), discretize(SHIU2024_CODE_V1, DT))
    with pytest.raises(ValueError, match="network_fingerprint"):
        other.restore(ckpt)


def test_same_inputs_same_output() -> None:
    net = _random_net()
    inputs = _random_inputs(net.n_neurons, 300, seed=4)
    runs: list[list[tuple[int, int]]] = []
    for _ in range(2):
        e = _engine(net)
        runs.append(_spikes_of([e.step(d) for d in inputs]))
    assert runs[0] == runs[1]


def test_sparse_only_on_a_large_sparse_graph() -> None:
    """200,000 neurons: a dense float64 W would need 320 GB. Only edges are touched."""
    n = 200_000
    ids: NDArray[np.int64] = np.arange(1, n + 1, dtype=np.int64)
    g, _ = build_graph(
        ids, [1, 2, 199_999], [2, 3, 200_000], [5, 5, 5], [1, 1, 1], dataset_id="big"
    )
    net = build_network(g, UnknownSignPolicy.ERROR, 1.0)  # 5 mV jump: subthreshold PSP
    e = _engine(net)
    e.step(_forced([0, 199_998]))
    for _ in range(DELAY + 100):
        e.step()
    s = e.state()
    assert s.v[1] > V0
    assert s.v[199_999] > V0
    assert int((s.v != V0).sum()) == 2
