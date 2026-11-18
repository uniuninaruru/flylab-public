"""LIF parameters as labelled values with units (docs/MODEL_ASSUMPTIONS.md section 3).

Every parameter is a `Labeled[float]`. A preset can only be changed by supplying a complete new
labelled value (`LIFParameters.with_overrides`); there is no way to pass a bare number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace

from flylab.provenance.labels import Evidence, Labeled

#: Units every parameter must carry. Times are ms, voltages mV.
PARAMETER_UNITS: dict[str, str] = {
    "v_rest": "mV",
    "v_reset": "mV",
    "v_threshold": "mV",
    "tau_m": "ms",
    "tau_syn": "ms",
    "t_ref": "ms",
    "delay": "ms",
    "w_syn": "mV",
}

#: Tolerance for "dt divides this duration exactly" (relative to the step count).
STEP_RATIO_TOLERANCE = 1e-9

#: |dt*(1/tau_m - 1/tau_syn)| below which the cancellation-free propagator form is used.
NEAR_EQUAL_TAU_X = 1e-3

SHIU_CITATION = "Shiu et al. 2024, Nature, doi:10.1038/s41586-024-07763-9"
SHIU_CODE = (
    "github.com/philshiu/Drosophila_brain_model@91bdd1e7dcf193f3e7ca5a8933497fcef63b7960 model.py"
)


class ParameterError(ValueError):
    """A parameter set is inconsistent, mislabelled or has the wrong units."""


def _shiu(value: float, units: str, methods: str, line: int, quote: str) -> Labeled[float]:
    """Value stated in the paper's Methods and identical in its reference code."""
    return Labeled(
        value=value,
        evidence=Evidence.LITERATURE_DERIVED,
        source=(
            f"{SHIU_CITATION}; Methods 'Computational model': {methods}; "
            f"reference code {SHIU_CODE}:{line} {quote}"
        ),
        units=units,
    )


@dataclass(frozen=True)
class LIFParameters:
    """Uniform LIF parameters (same value for every neuron)."""

    v_rest: Labeled[float]
    v_reset: Labeled[float]
    v_threshold: Labeled[float]
    tau_m: Labeled[float]
    tau_syn: Labeled[float]
    t_ref: Labeled[float]
    delay: Labeled[float]
    w_syn: Labeled[float]

    def __post_init__(self) -> None:
        for f in fields(self):
            p = getattr(self, f.name)
            if not isinstance(p, Labeled):
                raise ParameterError(f"{f.name} must be a Labeled value, got {type(p).__name__}")
            want = PARAMETER_UNITS[f.name]
            if p.units != want:
                raise ParameterError(f"{f.name} must be in {want}, got {p.units!r}")
            if isinstance(p.value, bool) or not isinstance(p.value, int | float):
                raise ParameterError(f"{f.name} must be a number, got {p.value!r}")
            if not math.isfinite(p.value):
                raise ParameterError(f"{f.name} must be finite, got {p.value!r}")
        v = self.values()
        if v["tau_m"] <= 0 or v["tau_syn"] <= 0:
            raise ParameterError("tau_m and tau_syn must be > 0")
        if v["t_ref"] < 0:
            raise ParameterError("t_ref must be >= 0")
        if v["delay"] <= 0:
            raise ParameterError("delay must be > 0 (instantaneous transmission is not supported)")
        if v["w_syn"] < 0:
            raise ParameterError("w_syn must be >= 0; the sign comes from the synapse sign")
        if not v["v_reset"] < v["v_threshold"]:
            raise ParameterError("v_reset must be below v_threshold")

    def values(self) -> dict[str, float]:
        """Plain float values keyed by parameter name."""
        return {f.name: float(getattr(self, f.name).value) for f in fields(self)}

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {f.name: getattr(self, f.name).to_dict() for f in fields(self)}

    def with_overrides(self, overrides: Mapping[str, Labeled[float]]) -> LIFParameters:
        """Return a copy with some parameters replaced by new labelled values."""
        unknown = sorted(set(overrides) - set(PARAMETER_UNITS))
        if unknown:
            raise ParameterError(
                f"unknown parameter(s) {unknown}; known: {sorted(PARAMETER_UNITS)}"
            )
        return replace(self, **dict(overrides))


#: Values stated in the Shiu et al. (2024) Methods and identical in their reference code; see
#: docs/MODEL_ASSUMPTIONS.md section 3 (quotes with typography normalised to ASCII).
SHIU2024_CODE_V1 = LIFParameters(
    v_rest=_shiu(-52.0, "mV", "'Vresting = -52 mV (resting potential)'", 22,
                 "'v_0' : -52 * mV # resting potential"),
    v_reset=_shiu(-52.0, "mV", "'Vreset = -52 mV (reset potential after spike)'", 23,
                  "'v_rst' : -52 * mV # reset potential after spike"),
    v_threshold=_shiu(-45.0, "mV", "'Vthreshold = -45 mV (threshold for spiking)'", 24,
                      "'v_th' : -45 * mV # threshold for spiking"),
    tau_m=_shiu(20.0, "ms", "'Tmbr = Cmb x Rmbr' with 'Rmbr = 10 Kohm cm2', 'Cmbr = 2 uF cm-2'",
                25, "'t_mbr' : 20 * ms # membrane time scale"),
    tau_syn=_shiu(5.0, "ms", "'tau = 5 ms (synapse decay timescale)'", 28,
                  "'tau' : 5 * ms # time constant"),
    t_ref=_shiu(2.2, "ms", "'Trefractory = 2.2 ms (refractory period)'", 31,
                "'t_rfc' : 2.2 * ms # refractory period"),
    delay=_shiu(1.8, "ms", "'Tdly = 1.8 ms (time delay from spike to change in membrane "
                "potential)'", 34, "'t_dly' : 1.8*ms # delay for changes in post-synaptic neuron"),
    w_syn=Labeled(
        value=0.275,
        evidence=Evidence.ASSUMED,
        rationale=(
            "Methods: 'Wsyn = 0.275 mV (free parameter ...)' and 'We chose Wsyn such that "
            "activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing' "
            f"on FlyWire v630 (female); same value in {SHIU_CODE}:37. Not recalibrated for "
            "MaleCNS synapse counts (OPEN_QUESTIONS Q16)"
        ),
        units="mV",
    ),
)  # fmt: skip

PARAMETER_SETS: dict[str, LIFParameters] = {"shiu2024_code_v1": SHIU2024_CODE_V1}


def labeled_dt(dt_ms: float) -> Labeled[float]:
    """The run's time step as a labelled value (it is a numerical choice, not biology)."""
    return Labeled(
        value=float(dt_ms),
        evidence=Evidence.ENGINEERED,
        rationale=(
            "numerical time step set in the run config; t_ref/dt and delay/dt must be integers"
        ),
        units="ms",
    )


def _exact_steps(duration_ms: float, dt_ms: float, name: str) -> int:
    ratio = duration_ms / dt_ms
    steps = round(ratio)
    if abs(ratio - steps) > STEP_RATIO_TOLERANCE * max(1.0, abs(ratio)):
        raise ParameterError(
            f"{name} = {duration_ms} ms is not a whole number of steps of dt = {dt_ms} ms "
            f"({ratio!r}); choose a dt that divides it (no silent rounding)"
        )
    return int(steps)


@dataclass(frozen=True)
class Discretization:
    """Exact one-step propagator of the subthreshold ODEs (docs/MODEL_ASSUMPTIONS.md 2.1).

    v <- v_rest + (v - v_rest)*a + g*b + e*c ;  g <- g*d. Constants are float64.
    """

    dt_ms: float
    refractory_steps: int
    delay_steps: int
    a: float
    b: float
    c: float
    d: float
    v_rest: float
    v_reset: float
    v_threshold: float

    def to_dict(self) -> dict[str, object]:
        return {
            "dt_ms": self.dt_ms,
            "refractory_steps": self.refractory_steps,
            "delay_steps": self.delay_steps,
            "propagator": {"a": self.a, "b": self.b, "c": self.c, "d": self.d},
            "update": "v <- v_rest + (v - v_rest)*a + g*b + e*c; g <- g*d (active neurons only)",
        }


def discretize(params: LIFParameters, dt_ms: float) -> Discretization:
    """Propagator constants and integer step counts for `dt_ms`. Rejects inexact ratios."""
    if not (isinstance(dt_ms, int | float) and math.isfinite(dt_ms) and dt_ms > 0):
        raise ParameterError(f"dt_ms must be a positive number, got {dt_ms!r}")
    v = params.values()
    tau_m, tau_s = v["tau_m"], v["tau_syn"]
    a = math.exp(-dt_ms / tau_m)
    d = math.exp(-dt_ms / tau_s)
    # b = tau_s/(tau_s - tau_m) * (d - a). Near tau_m == tau_s that difference cancels, so use the
    # algebraically equal b = a*(dt/tau_m)*expm1(x)/x, x = dt*(1/tau_m - 1/tau_s), which is exact
    # across the limit (x -> 0 gives (dt/tau_m)*a). The general formula is kept elsewhere so that
    # results for ordinary parameters (e.g. the Shiu preset) are bit-for-bit unchanged.
    x = dt_ms * (1.0 / tau_m - 1.0 / tau_s)
    if abs(x) < NEAR_EQUAL_TAU_X:
        b = a * (dt_ms / tau_m) * (math.expm1(x) / x if x != 0.0 else 1.0)
    else:
        b = tau_s / (tau_s - tau_m) * (d - a)
    delay_steps = _exact_steps(v["delay"], dt_ms, "delay")
    if delay_steps < 1:
        raise ParameterError("delay must be at least one time step")
    return Discretization(
        dt_ms=float(dt_ms),
        refractory_steps=_exact_steps(v["t_ref"], dt_ms, "t_ref"),
        delay_steps=delay_steps,
        a=a,
        b=b,
        c=1.0 - a,
        d=d,
        v_rest=v["v_rest"],
        v_reset=v["v_reset"],
        v_threshold=v["v_threshold"],
    )


def steps_for(duration_ms: float, dt_ms: float, name: str) -> int:
    """Whole number of steps in `duration_ms`; raises ParameterError if inexact."""
    return _exact_steps(duration_ms, dt_ms, name)
