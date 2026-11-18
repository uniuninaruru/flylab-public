"""Lockstep backend comparison (flylab.neural.parity): identical, justified, unjustified."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from numpy.typing import NDArray

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import EngineState, StepDrive, StepInterventions, _Advanced
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, Discretization, discretize
from flylab.neural.parity import DEFAULT_MARGIN_MV, lockstep, threshold_margins

DISC = discretize(SHIU2024_CODE_V1, 0.1)


class ListSchedule:
    def __init__(self, inputs: list[tuple[StepDrive | None, StepInterventions | None]]) -> None:
        self._inputs = inputs

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        return self._inputs[step]


def _net(n: int = 300, seed: int = 1) -> SynapticNetwork:
    g, _ = random_graph(n, 30, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, 0.6)


def _inputs(n: int, steps: int, seed: int) -> ListSchedule:
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 14, n)
    out: list[tuple[StepDrive | None, StepInterventions | None]] = []
    for k in range(steps):
        drive = StepDrive(
            current_idx=np.arange(n, dtype=np.int32),
            current_mV=base + rng.normal(0, 2, n),
            forced_idx=np.flatnonzero(rng.random(n) < 0.002).astype(np.int32),
        )
        iv = StepInterventions(block_output_idx=np.arange(10, dtype=np.int32)) if k > 200 else None
        out.append((drive, iv))
    return ListSchedule(out)


def _pair(
    net: SynapticNetwork,
    test_cls: type[NumpyLIFEngine] = NumpyLIFEngine,
    disc: Discretization = DISC,
) -> tuple[NumpyLIFEngine, NumpyLIFEngine]:
    pops = {"all": np.arange(net.n_neurons)}
    ref = NumpyLIFEngine(net, DISC, populations=pops)
    test = test_cls(net, disc, populations=pops)
    ref.reset(1)
    test.reset(1)
    return ref, test


def test_identical_engines_are_identical() -> None:
    net = _net()
    ref, test = _pair(net)
    rep = lockstep(ref, test, 400, _inputs(net.n_neurons, 400, 3), state_every=50)
    assert rep.status == "identical"
    assert rep.ok
    assert rep.steps_identical == 400
    assert rep.state_checks == 9  # initial state plus eight periodic checks
    assert rep.max_abs_v_diff_mV == 0.0
    assert rep.n_spikes_reference == rep.n_spikes_test > 100
    assert rep.population_spikes_reference == rep.population_spikes_test
    assert rep.to_dict()["divergence"] is None


def test_torch_cpu_matches_numpy_in_lockstep() -> None:
    if os.environ.get("FLYLAB_REQUIRE_TORCH") != "1":
        pytest.importorskip("torch")
    from flylab.neural.lif_torch import TorchLIFEngine

    net = _net()
    ref = NumpyLIFEngine(net, DISC, "float32")
    test = TorchLIFEngine(net, DISC, "float32", max_edges_per_chunk=5)
    ref.reset(1)
    test.reset(1)
    rep = lockstep(ref, test, 400, _inputs(net.n_neurons, 400, 4))
    assert rep.status == "identical"
    assert rep.margin_mV == DEFAULT_MARGIN_MV["float32"]


def test_threshold_margin_is_the_phase2_candidate_potential() -> None:
    net = _net(n=5)
    e = NumpyLIFEngine(net, DISC)
    v = np.array([-52.0, -45.5, -45.0, -40.0, -60.0])
    g = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    before = EngineState(0, v, g, np.zeros(5, np.int64), tuple(np.zeros(0, np.int32)
                         for _ in range(DISC.delay_steps)))  # fmt: skip
    drive = StepDrive(np.array([1, 1], np.int32), np.array([2.0, 3.0]))
    who = np.array([0, 1, 4], dtype=np.int64)
    got = threshold_margins(before, drive, e, who)
    e_mv = np.array([0.0, 5.0, 0.0])
    cand = DISC.v_rest + (v[who] - DISC.v_rest) * DISC.a + g[who] * DISC.b + e_mv * DISC.c
    np.testing.assert_allclose(got, np.abs(cand - DISC.v_threshold), rtol=0, atol=1e-12)


def _ramp(steps: int) -> ListSchedule:
    """One neuron, constant 20 mV drive: it crosses threshold at step 86 (closed form)."""
    d = StepDrive(np.array([0], np.int32), np.array([20.0]))
    return ListSchedule([(d, None)] * steps)


def test_shifted_threshold_is_a_justified_flip_only_within_the_margin() -> None:
    g, _ = random_graph(1, 0, seed=0)
    net = build_network(g, UnknownSignPolicy.ERROR, 1.0)
    shifted = dataclasses.replace(DISC, v_threshold=DISC.v_threshold + 0.5)
    for margin, expect in ((1.0, "near_threshold_divergence"), (1e-6, "mismatch")):
        ref, test = _pair(net, disc=shifted)
        rep = lockstep(ref, test, 200, _ramp(200), margin_mV=margin)
        assert rep.status == expect
        assert rep.divergence is not None
        assert rep.divergence.step == 86
        assert rep.divergence.kind == "spikes"
        assert rep.divergence.neurons == (0,)
        worst = rep.divergence.max_threshold_margin_mV
        assert worst is not None
        assert 0 < worst < 0.5
        assert rep.steps_compared == 200
        if rep.ok:
            assert rep.resyncs == rep.justified_flips >= 2
            assert rep.steps_identical + rep.justified_flips == 200
            np.testing.assert_array_equal(ref.state().v, test.state().v)
        else:
            assert rep.resyncs == 0
        # Totals describe actual spikes, including the test's resynchronized trajectory.
        assert rep.n_spikes_reference > rep.n_spikes_test


class _FlagFlip(NumpyLIFEngine):
    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        out = super()._advance(n, current_idx, current_mV, forced_idx, block_idx, clamp_idx)
        if out.spike_idx.size and n >= 250:
            return dataclasses.replace(out, propagated=~out.propagated)
        return out


class _Nudge(NumpyLIFEngine):
    """Adds 1 mV to an unconnected neuron's v once: invisible in spikes, visible in state."""

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        out = super()._advance(n, current_idx, current_mV, forced_idx, block_idx, clamp_idx)
        if n == 10:
            self._v[-1] += 1.0
        return out


def test_flag_differences_are_mismatches() -> None:
    net = _net()
    ref, test = _pair(net, test_cls=_FlagFlip)
    rep = lockstep(ref, test, 400, _inputs(net.n_neurons, 400, 3))
    assert rep.status == "mismatch"
    assert not rep.ok
    assert rep.divergence is not None
    assert rep.divergence.kind == "flags"
    assert rep.divergence.step >= 250


def test_state_drift_beyond_the_margin_fails_even_with_equal_spikes() -> None:
    net = _net()
    ref, test = _pair(net, test_cls=_Nudge)
    rep = lockstep(ref, test, 40, ListSchedule([(None, None)] * 40), state_every=20)
    assert rep.divergence is None  # no input, no spikes
    assert rep.status == "state_mismatch"
    assert not rep.ok
    assert rep.max_abs_v_diff_mV is not None
    assert rep.max_abs_v_diff_mV > 0.5


def test_engines_must_be_comparable() -> None:
    a = NumpyLIFEngine(_net(seed=1), DISC)
    b = NumpyLIFEngine(_net(seed=2), DISC)
    a.reset(0)
    b.reset(0)
    with pytest.raises(ValueError, match="different networks"):
        lockstep(a, b, 1)
    c = NumpyLIFEngine(_net(seed=1), DISC, "float32")
    c.reset(0)
    with pytest.raises(ValueError, match="dtypes"):
        lockstep(a, c, 1)


@pytest.mark.parametrize("margin", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0])
def test_margin_must_be_finite_and_positive(margin: float) -> None:
    ref, test = _pair(_net(n=5))
    with pytest.raises(ValueError, match="finite and positive"):
        lockstep(ref, test, 1, margin_mV=margin)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["v", "g"])
@pytest.mark.parametrize("side", ["reference", "test"])
def test_non_finite_initial_state_is_never_identical(value: float, field: str, side: str) -> None:
    ref, test = _pair(_net(n=5))
    engine = ref if side == "reference" else test
    state = engine.state()
    getattr(state, field)[0] = value
    engine._load_state(state)  # inject a backend defect, bypassing public validation
    rep = lockstep(ref, test, 0)
    assert rep.status == "non_finite"
    assert not rep.ok
    assert rep.to_dict()[f"max_abs_{field}_diff_mV"] is None


@pytest.mark.parametrize("field", ["last_spike", "in_flight"])
def test_discrete_initial_state_must_match(field: str) -> None:
    ref, test = _pair(_net(n=5))
    state = test.state()
    if field == "last_spike":
        state.last_spike[0] = -1
    else:
        state = dataclasses.replace(
            state, in_flight=(np.array([0], np.int32), *state.in_flight[1:])
        )
    test.set_state(state)
    rep = lockstep(ref, test, 1)
    assert rep.status == "state_mismatch"


def test_state_import_sets_step_and_copies_all_dynamic_state() -> None:
    ref, test = _pair(_net(n=5))
    ref.simulate(7, schedule=ListSchedule([(StepDrive(forced_idx=np.array([0], np.int32)),
                                          None)] * 7))  # fmt: skip
    state = ref.state()
    test.set_state(state)
    assert test.current_step == 7
    assert test.seed == 1
    np.testing.assert_array_equal(test.state().last_spike, state.last_spike)
    state.v[:] = 0
    assert np.all(test.state().v < 0)  # imported arrays are owned by the backend
    assert lockstep(ref, test, 3).ok


def test_candidate_diagnostic_uses_actual_state_dtype_arithmetic() -> None:
    ref = NumpyLIFEngine(_net(n=5), DISC, "float32")
    ref.reset(1)
    who = np.array([0, 2], np.int64)
    with pytest.raises(RuntimeError, match="completed step"):
        ref.candidate_potentials(who)
    drive = StepDrive(np.array([0, 2], np.int32), np.array([3.0, 4.0]))
    ref.step(drive)
    np.testing.assert_array_equal(ref.candidate_potentials(who), ref.state().v[who])
    ref.set_state(ref.state())
    with pytest.raises(RuntimeError, match="completed step"):
        ref.candidate_potentials(who)


def test_non_finite_candidates_cannot_be_hidden_by_identical_resets() -> None:
    ref, test = _pair(_net(n=1))
    huge = np.finfo(np.float64).max
    for engine in (ref, test):
        state = engine.state()
        state.v[:] = huge
        state.g[:] = huge
        engine.set_state(state)
    drive = StepDrive(np.array([0], np.int32), np.array([huge]))
    with np.errstate(over="ignore"):
        rep = lockstep(ref, test, 1, ListSchedule([(drive, None)]))
    assert rep.steps_identical == 1  # both overflow, spike, then reset to finite v/g
    assert np.isfinite(ref.state().v).all()
    assert rep.status == "non_finite"
    assert not rep.ok
