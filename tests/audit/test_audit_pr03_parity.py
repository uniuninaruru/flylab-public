"""PR-03 review: the lockstep parity policy (flylab.neural.parity) can pass a wrong engine.

Findings 1, 2, 3, 5 and 6 of docs/audit/PR-03-review.md. Every test runs two NumPy engines:
the reference is the real `NumpyLIFEngine`, the "test" engine is a subclass with one deliberate
bug, standing in for a CUDA engine that is wrong. These regressions reject those bugs;
finding numbers remain next to their original scenarios.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import NEVER, EngineState, StepDrive, StepInterventions, _Advanced
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import SHIU2024_CODE_V1, Discretization, discretize
from flylab.neural.parity import lockstep

DISC = discretize(SHIU2024_CODE_V1, 0.1)
Inputs = tuple[StepDrive | None, StepInterventions | None]


class ListSchedule:
    def __init__(self, inputs: list[Inputs]) -> None:
        self._inputs = inputs

    def inputs(self, step: int) -> Inputs:
        return self._inputs[step]


def _net(n: int, out_degree: float, seed: int = 1, w: float = 0.6) -> SynapticNetwork:
    g, _ = random_graph(n, out_degree, seed=seed)
    return build_network(g, UnknownSignPolicy.ERROR, w)


def _start_from(engine: NumpyLIFEngine, v: NDArray[np.float64]) -> None:
    """reset(1), then set the membrane potentials (g = 0, nothing refractory or in flight)."""
    engine.reset(1)
    n = engine.n
    engine._load_state(
        EngineState(
            step=0,
            v=v.copy(),
            g=np.zeros(n),
            last_spike=np.full(n, NEVER, dtype=np.int64),
            in_flight=tuple(np.zeros(0, dtype=np.int32) for _ in range(DISC.delay_steps)),
        )
    )


def _v_with_candidate(
    n: int, neuron: int, offset_mV: float, drive_mV: float = 0.0
) -> NDArray[np.float64]:
    """Rest everywhere; `neuron` placed so that its phase-2 candidate potential at step 0
    (MODEL_ASSUMPTIONS 2.1) is V_th + offset_mV under a constant drive of drive_mV."""
    d = DISC
    v = np.full(n, d.v_rest)
    v[neuron] = d.v_rest + (d.v_threshold + offset_mV - d.v_rest - drive_mV * d.c) / d.a
    return v


def _noisy_drive(n: int, steps: int, seed: int, skip: int = 0) -> list[Inputs]:
    """Noisy injected drive to every neuron except the first `skip` (as in tests/unit)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(skip, n, dtype=np.int32)
    base = rng.uniform(4, 14, idx.size)
    return [(StepDrive(idx, base + rng.normal(0, 2, idx.size)), None) for _ in range(steps)]


# --- Finding 1 ---------------------------------------------------------------------------------


class _LosesArrivalsAfterStep0(NumpyLIFEngine):
    """Bug: from step 1 on, spikes are reported as propagated but never delivered."""

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
        if n >= 1:
            self._ring[n % self.disc.delay_steps] = np.zeros(0, dtype=np.int32)
        return out


def test_finding_1_nothing_is_checked_after_a_justified_flip() -> None:
    net = _net(300, 30)
    pops = {"all": np.arange(net.n_neurons)}
    # The test engine's threshold is 5e-7 mV higher (below the float64 margin 1e-6 mV), and
    # neuron 0 starts with a candidate potential 2e-7 mV above V_th: the reference fires at
    # step 0 and the test engine does not. Per MODEL_ASSUMPTIONS 2.3 that flip is justified.
    shifted = dataclasses.replace(DISC, v_threshold=DISC.v_threshold + 5e-7)
    ref = NumpyLIFEngine(net, DISC, populations=pops)
    test = _LosesArrivalsAfterStep0(net, shifted, populations=pops)
    v0 = _v_with_candidate(net.n_neurons, 0, +2e-7)
    _start_from(ref, v0)
    _start_from(test, v0)
    rep = lockstep(ref, test, 400, ListSchedule(_noisy_drive(net.n_neurons, 400, 3, skip=1)))
    # Today: status "near_threshold_divergence" at step 0, ok=True, although the test engine
    # has no synaptic transmission after step 0 (population "all": 776 spikes in the
    # reference, 259 in the test engine; see the report).
    assert not rep.ok, rep.to_dict()
    assert rep.resyncs >= 1  # finding 1: continue comparing after accepting the first flip
    assert rep.steps_compared == 400


# --- Finding 2 ---------------------------------------------------------------------------------


class _DoublesDrive(NumpyLIFEngine):
    """Bug: the injected drive is applied twice."""

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        return super()._advance(n, current_idx, 2.0 * current_mV, forced_idx, block_idx, clamp_idx)


def test_finding_2_flip_justified_although_the_test_engine_is_far_off() -> None:
    net = _net(1, 0, seed=0, w=1.0)
    drive = StepDrive(np.array([0], np.int32), np.array([10.0]))
    ref = NumpyLIFEngine(net, DISC)
    test = _DoublesDrive(net, DISC)
    # Reference candidate at step 0: V_th - 1e-7 mV (no spike). The doubled drive adds
    # 10 mV * c = 0.05 mV, i.e. 50,000 times the float64 margin, and the test engine fires.
    v0 = _v_with_candidate(1, 0, -1e-7, drive_mV=10.0)
    _start_from(ref, v0)
    _start_from(test, v0)
    rep = lockstep(ref, test, 200, ListSchedule([(drive, None)] * 200))
    # Today: "near_threshold_divergence" at step 0 with state_checks = 0, ok=True.
    assert not rep.ok, rep.to_dict()
    assert rep.divergence is not None
    assert rep.divergence.max_threshold_margin_mV is not None
    assert rep.divergence.max_threshold_margin_mV > 0.04  # actual doubled-drive candidate


class _IgnoresBlock(NumpyLIFEngine):
    """Bug: output blocking is ignored (every spike is propagated)."""

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        empty = np.zeros(0, dtype=np.int32)
        return super()._advance(n, current_idx, current_mV, forced_idx, empty, clamp_idx)


def test_finding_2_flag_difference_hidden_by_a_justified_flip() -> None:
    net = _net(2, 0, seed=0, w=1.0)
    shifted = dataclasses.replace(DISC, v_threshold=DISC.v_threshold + 5e-7)
    ref = NumpyLIFEngine(net, DISC)
    test = _IgnoresBlock(net, shifted)
    v0 = _v_with_candidate(2, 0, +2e-7)  # neuron 0: justified flip, as in finding 1
    _start_from(ref, v0)
    _start_from(test, v0)
    # Neuron 1 is forced to spike with its output blocked: propagated=False in the reference,
    # True in the test engine.
    step0 = (
        StepDrive(forced_idx=np.array([1], np.int32)),
        StepInterventions(block_output_idx=np.array([1], np.int32)),
    )
    rep = lockstep(ref, test, 1, ListSchedule([step0]))
    assert not rep.ok, rep.to_dict()  # today: "near_threshold_divergence", ok=True
    assert rep.divergence is not None
    assert rep.divergence.kind == "flags"


# --- Finding 3 ---------------------------------------------------------------------------------


class _NaNAtStep5(NumpyLIFEngine):
    """Bug: the last neuron's membrane potential becomes NaN at step 5."""

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
        if n == 5:
            self._v[-1] = np.nan
        return out


def test_finding_3_nan_state_passes_as_identical() -> None:
    net = _net(300, 30)
    ref = NumpyLIFEngine(net, DISC)
    test = _NaNAtStep5(net, DISC)
    ref.reset(1)
    test.reset(1)
    rep = lockstep(ref, test, 40, ListSchedule([(None, None)] * 40), state_every=10)
    assert np.isnan(test.state().v).sum() == 1
    assert not rep.ok, rep.to_dict()  # today: status "identical", max_abs_v_diff_mV 0.0


class _Nudge(NumpyLIFEngine):
    """Bug: adds 1 mV to the last neuron's v once (as in tests/unit/test_parity.py)."""

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


def test_finding_3_nan_margin_disables_the_state_check() -> None:
    net = _net(300, 30)
    ref = NumpyLIFEngine(net, DISC)
    test = _Nudge(net, DISC)
    ref.reset(1)
    test.reset(1)
    # With the default margin this is a state_mismatch (tests/unit/test_parity.py).
    try:
        rep = lockstep(
            ref, test, 40, ListSchedule([(None, None)] * 40), margin_mV=float("nan"),
            state_every=20,
        )  # fmt: skip
    except ValueError:
        return  # refusing the margin is a fix
    assert not rep.ok, rep.to_dict()


# --- Finding 5 ---------------------------------------------------------------------------------


class _ClampKeepsG(NumpyLIFEngine):
    """Bug: clamped neurons keep their synaptic drive g instead of g <- 0 (phase 5)."""

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        g_before = self._g.copy()
        out = super()._advance(n, current_idx, current_mV, forced_idx, block_idx, clamp_idx)
        self._g[clamp_idx] = g_before[clamp_idx]
        return out


def test_finding_5_g_only_difference_passes() -> None:
    net = _net(300, 30)
    n = net.n_neurons
    clamp = StepInterventions(clamp_idx=np.arange(20, dtype=np.int32))
    sched: list[Inputs] = [(StepDrive(forced_idx=np.arange(100, n, dtype=np.int32)), None)]
    sched += [(None, None)] * 24  # the step-0 spikes arrive at step 18 (delay)
    sched += [(None, clamp)] * 35  # neurons 0..19 clamped from step 25 to the end
    ref = NumpyLIFEngine(net, DISC)
    test = _ClampKeepsG(net, DISC)
    ref.reset(1)
    test.reset(1)
    rep = lockstep(ref, test, len(sched), ListSchedule(sched), state_every=10)
    # The reference zeroed g of the clamped neurons, the test engine kept it (up to 53.7 mV).
    assert np.abs(ref.state().g - test.state().g).max() > 0.1
    assert np.array_equal(ref.state().v, test.state().v)
    assert not rep.ok, rep.to_dict()  # today: status "identical"


# --- Finding 6 ---------------------------------------------------------------------------------


def _documented_float64_bound(disc: Discretization) -> float:
    """MODEL_ASSUMPTIONS 2.3's own float64 estimate, for its own example of m = 1e4 arrivals of
    sum |w| = 1e4 mV per step, times 2 because both engines round their sums (|S_cpu - S_gpu|
    <= |S_cpu - S| + |S - S_gpu|)."""
    m, sum_w = 1e4, 1e4
    per_step_g = 2 * (m - 1) * 2.0**-53 * sum_w
    return per_step_g / (1 - disc.d) * disc.b / (1 - disc.a)


@pytest.mark.parametrize("dt_ms", [0.1, 0.01])
def test_finding_6_default_margin_covers_the_documented_bound(dt_ms: float) -> None:
    disc = discretize(SHIU2024_CODE_V1, dt_ms)
    net = _net(1, 0, seed=0, w=1.0)
    ref = NumpyLIFEngine(net, disc)
    test = NumpyLIFEngine(net, disc)
    ref.reset(0)
    test.reset(0)
    rep = lockstep(ref, test, 1)  # today: DEFAULT_MARGIN_MV["float64"] = 1e-6 for every dt
    bound = _documented_float64_bound(disc)
    assert rep.margin_mV >= bound, f"margin {rep.margin_mV:g} mV < documented bound {bound:g} mV"
