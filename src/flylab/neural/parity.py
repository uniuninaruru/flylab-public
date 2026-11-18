"""Run two LIF engines in lockstep on identical inputs and explain any difference (spec 7.4).

Policy (docs/MODEL_ASSUMPTIONS.md section 2.3):

- Both engines receive the same `StepDrive` / `StepInterventions` objects at every step.
- Initial, periodic and pre-flip states must agree in v, g and the discrete spike history.
  Non-finite states fail at every step. Margins are engineering tolerances, not universal bounds.
- Every spike set and the common spikes' flags are compared. A flip is justified only when
  both engines' actual phase-2 candidates lie within the margin of their thresholds.
- After each justified flip the tested engine receives the reference's full state. Every
  subsequent step is still compared; population counts are supplementary descriptions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from flylab.neural.engine import EngineState, LIFEngineBase, NeuralStep, Schedule, StepDrive
from flylab.neural.params import SHIU2024_CODE_V1, Discretization, discretize

#: Engineering tolerances at the Shiu preset's dt=0.1 ms (MODEL_ASSUMPTIONS 2.3).
DEFAULT_MARGIN_MV = {"float64": 2e-6, "float32": 1e-2}
_MAX_LISTED = 20


def _roundoff_gain(disc: Discretization) -> float:
    """Steady-state gains for repeated g and v perturbations in the linear recurrence."""
    g_gain = 1.0 / (1.0 - disc.d)
    v_gain = 1.0 / (1.0 - disc.a)
    return max(g_gain, v_gain, disc.b * g_gain * v_gain)


_BASE_GAIN = _roundoff_gain(discretize(SHIU2024_CODE_V1, 0.1))


def default_margin_mV(dtype: str, disc: Discretization) -> float:
    """Scale the baseline tolerance upward for more persistent numerical perturbations.

    This accounts for dt/time-constant dependence, but does not assert a bound for arbitrary
    graph weights, drives or reset histories. Those require empirical parity checks.
    """
    return DEFAULT_MARGIN_MV[dtype] * max(1.0, _roundoff_gain(disc) / _BASE_GAIN)


def validate_margin(margin_mV: float) -> float:
    delta = float(margin_mV)
    if not isfinite(delta) or delta <= 0:
        raise ValueError("margin_mV must be finite and positive")
    return delta


def threshold_margins(
    before: EngineState,
    drive: StepDrive | None,
    engine: LIFEngineBase,
    neurons: NDArray[np.int64],
) -> NDArray[np.float64]:
    """|v_candidate - v_threshold| (mV) of `neurons` for the step that starts in `before`.

    v_candidate is the phase-2 subthreshold update (before spike detection and reset),
    reconstructed in float64 from the supplied state. This is an analytic diagnostic;
    lockstep uses each backend's actual `candidate_potentials` to justify a flip.
    """
    d = engine.disc
    v = before.v[neurons].astype(np.float64)
    g = before.g[neurons].astype(np.float64)
    cand = d.v_rest + (v - d.v_rest) * d.a + g * d.b
    if drive is not None and drive.current_idx.size:
        e = np.zeros(engine.n, dtype=np.float64)
        np.add.at(e, drive.current_idx, drive.current_mV)
        cand = cand + e[neurons] * d.c
    return np.abs(cand - d.v_threshold)


@dataclass(frozen=True)
class Divergence:
    step: int
    kind: str  # "spikes" (different sets) or "flags" (common spikes' forced/propagated differ)
    n_neurons: int
    neurons: tuple[int, ...]  # first few neuron indices involved
    max_threshold_margin_mV: float | None
    near_threshold: bool

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass
class ParityReport:
    reference: str
    test: str
    dtype: str
    steps: int
    margin_mV: float
    state_every: int
    steps_identical: int = 0
    steps_compared: int = 0
    justified_flips: int = 0
    resyncs: int = 0
    state_checks: int = 0
    max_abs_v_diff_mV: float | None = 0.0
    max_abs_g_diff_mV: float | None = 0.0
    state_within_margin: bool = True
    state_finite: bool = True
    divergence: Divergence | None = None
    n_spikes_reference: int = 0
    n_spikes_test: int = 0
    population_spikes_reference: dict[str, int] = field(default_factory=dict)
    population_spikes_test: dict[str, int] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.state_finite:
            return "non_finite"
        if self.divergence is not None and not self.divergence.near_threshold:
            return "mismatch"
        if not self.state_within_margin:
            return "state_mismatch"
        if self.divergence is None:
            return "identical"
        return "near_threshold_divergence"

    @property
    def ok(self) -> bool:
        return self.status in ("identical", "near_threshold_divergence")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ok": self.ok,
            "reference": self.reference,
            "test": self.test,
            "dtype": self.dtype,
            "steps": self.steps,
            "steps_identical": self.steps_identical,
            "steps_compared": self.steps_compared,
            "justified_flips": self.justified_flips,
            "resyncs": self.resyncs,
            "margin_mV": self.margin_mV,
            "state_every": self.state_every,
            "state_checks": self.state_checks,
            "max_abs_v_diff_mV": self.max_abs_v_diff_mV,
            "max_abs_g_diff_mV": self.max_abs_g_diff_mV,
            "state_within_margin": self.state_within_margin,
            "state_finite": self.state_finite,
            "divergence": None if self.divergence is None else self.divergence.to_dict(),
            "n_spikes_reference": self.n_spikes_reference,
            "n_spikes_test": self.n_spikes_test,
            "population_spikes_reference": dict(self.population_spikes_reference),
            "population_spikes_test": dict(self.population_spikes_test),
        }


def _state_diff(a: EngineState, b: EngineState) -> tuple[float, float]:
    with np.errstate(invalid="ignore", over="ignore"):
        dv = float(np.max(np.abs(a.v.astype(np.float64) - b.v.astype(np.float64)), initial=0.0))
        dg = float(np.max(np.abs(a.g.astype(np.float64) - b.g.astype(np.float64)), initial=0.0))
    return dv, dg


def _discrete_equal(a: EngineState, b: EngineState) -> bool:
    return (
        a.step == b.step
        and np.array_equal(a.last_spike, b.last_spike)
        and len(a.in_flight) == len(b.in_flight)
        and all(np.array_equal(x, y) for x, y in zip(a.in_flight, b.in_flight, strict=True))
    )


def lockstep(
    reference: LIFEngineBase,
    test: LIFEngineBase,
    steps: int,
    schedule: Schedule | None = None,
    margin_mV: float | None = None,
    state_every: int = 100,
    progress: Callable[[int], None] | None = None,
) -> ParityReport:
    """Compare every step; resynchronize after each individually justified threshold flip.

    Thresholds may differ for diagnostics, but both candidates must lie within the margin of
    their respective thresholds. All other dynamics, network, dtype and populations must match.
    """
    if reference.current_step != test.current_step:
        raise ValueError("engines must start at the same step")
    if reference.network.fingerprint() != test.network.fingerprint():
        raise ValueError("engines run different networks")
    if reference.dtype != test.dtype:
        raise ValueError("engines use different dtypes")
    if replace(reference.disc, v_threshold=test.disc.v_threshold) != test.disc:
        raise ValueError("engines use different discretizations")
    ref_pops, test_pops = reference.population_indices(), test.population_indices()
    if ref_pops.keys() != test_pops.keys() or any(
        not np.array_equal(idx, test_pops[name]) for name, idx in ref_pops.items()
    ):
        raise ValueError("engines use different populations")
    if state_every < 1:
        raise ValueError("state_every must be >= 1")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    delta = validate_margin(
        default_margin_mV(reference.dtype, reference.disc) if margin_mV is None else margin_mV
    )
    rep = ParityReport(
        reference=reference.backend,
        test=test.backend,
        dtype=reference.dtype,
        steps=steps,
        margin_mV=delta,
        state_every=state_every,
    )

    def check_finite(a: EngineState, b: EngineState) -> bool:
        finite = (
            all(np.isfinite(x).all() for x in (a.v, a.g, b.v, b.g))
            and reference.state_finite()
            and test.state_finite()
        )
        if not finite:
            rep.state_finite = False
            rep.state_within_margin = False
            # null means undefined, never a misleading zero or non-standard JSON NaN.
            if not np.isfinite(a.v).all() or not np.isfinite(b.v).all():
                rep.max_abs_v_diff_mV = None
            if not np.isfinite(a.g).all() or not np.isfinite(b.g).all():
                rep.max_abs_g_diff_mV = None
        return finite

    def check_states(a: EngineState, b: EngineState) -> bool:
        rep.state_checks += 1
        if not check_finite(a, b):
            return False
        dv, dg = _state_diff(a, b)
        if rep.max_abs_v_diff_mV is not None:
            rep.max_abs_v_diff_mV = max(rep.max_abs_v_diff_mV, dv)
        if rep.max_abs_g_diff_mV is not None:
            rep.max_abs_g_diff_mV = max(rep.max_abs_g_diff_mV, dg)
        within = dv <= delta and dg <= delta and _discrete_equal(a, b)
        if not within:
            rep.state_within_margin = False
        return within

    before_ref, before_test = reference.state(), test.state()
    check_states(before_ref, before_test)
    for k in range(steps):
        drive, iv = schedule.inputs(reference.current_step) if schedule else (None, None)
        a = reference.step(drive, iv)
        b = test.step(drive, iv)
        after_ref, after_test = reference.state(), test.state()
        finite = check_finite(after_ref, after_test)
        rep.n_spikes_reference += int(a.spike_idx.size)
        rep.n_spikes_test += int(b.spike_idx.size)
        for name, c in a.population_spike_counts.items():
            rep.population_spikes_reference[name] = rep.population_spikes_reference.get(name, 0) + c
        for name, c in b.population_spike_counts.items():
            rep.population_spikes_test[name] = rep.population_spikes_test.get(name, 0) + c
        rep.steps_compared += 1
        divergence = _compare_step(a, b, reference, test, delta)
        if divergence is None:
            rep.steps_identical += 1
        else:
            pre_agrees = check_states(before_ref, before_test)
            who = np.setxor1d(a.spike_idx, b.spike_idx).astype(np.int64)
            # A threshold flip cannot explain a refractory/clamped or forced-only spike.
            active = all(
                np.all(before.step - before.last_spike[who] >= engine.disc.refractory_steps)
                for before, engine in ((before_ref, reference), (before_test, test))
            )
            if iv is not None:
                active = active and not np.intersect1d(who, iv.clamp_idx).size
            justified = divergence.near_threshold and pre_agrees and finite and active
            divergence = replace(divergence, near_threshold=bool(justified))
            if rep.divergence is None or (rep.divergence.near_threshold and not justified):
                rep.divergence = divergence
            if justified and rep.ok:
                rep.justified_flips += 1
                test.set_state(after_ref)
                rep.resyncs += 1
                after_test = test.state()
        # After an accepted flip this measures the resynchronized state. After a rejected
        # flip it also records the reset discontinuity; the mismatch verdict already persists.
        if (k + 1) % state_every == 0 or k + 1 == steps:
            check_states(after_ref, after_test)
        before_ref, before_test = after_ref, after_test
        if progress is not None:
            progress(k + 1)
    return rep


def _compare_step(
    a: NeuralStep,
    b: NeuralStep,
    reference: LIFEngineBase,
    test: LIFEngineBase,
    delta: float,
) -> Divergence | None:
    """Check common flags before considering each engine's actual candidate potentials."""
    _, ia, ib = np.intersect1d(a.spike_idx, b.spike_idx, return_indices=True)
    flags = (a.forced[ia] != b.forced[ib]) | (a.propagated[ia] != b.propagated[ib])
    if flags.any():
        who = a.spike_idx[ia[flags]]
        return Divergence(
            step=a.step, kind="flags", n_neurons=int(who.size),
            neurons=tuple(int(x) for x in who[:_MAX_LISTED]),
            max_threshold_margin_mV=None, near_threshold=False,
        )  # fmt: skip
    who = np.setxor1d(a.spike_idx, b.spike_idx).astype(np.int64)
    if not who.size:
        return None
    margins = np.concatenate(
        [
            np.abs(
                engine.candidate_potentials(who)
                - np.dtype(engine.dtype).type(engine.disc.v_threshold)
            )
            for engine in (reference, test)
        ]
    )
    worst = float(margins.max())
    forced = any(np.any(out.forced[np.isin(out.spike_idx, who)]) for out in (a, b))
    near = bool(np.isfinite(margins).all() and worst <= delta and not forced)
    return Divergence(
        step=a.step, kind="spikes", n_neurons=int(who.size),
        neurons=tuple(int(x) for x in who[:_MAX_LISTED]),
        max_threshold_margin_mV=worst if isfinite(worst) else None, near_threshold=near,
    )  # fmt: skip
