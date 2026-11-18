"""Body and motor-decoder contracts (spec sections 9.1 and 9.2).

Pure NumPy. This module never imports MuJoCo, dm_control or flybody, so the contracts can be
tested in the main FlyLab environment and implemented by any body backend.

Conventions for every payload here:

- Arrays are ``float64``, owned copies, and read-only after construction.
- Times are in milliseconds (``*_ms``) of **simulation** time, never wall-clock time.
- ``schema_version`` is ``flylab.SCHEMA_VERSION`` at construction.
- Units are strings, or ``None`` when FlyLab has not established them (never guessed).

The action vector of a body backend is a **software interface** of the upstream body model
(evidence label ENGINEERED). It is not a biological motor map: action element ``i`` drives one
MuJoCo actuator, not an identified motor neuron or muscle.
"""

from __future__ import annotations

import enum
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from flylab import SCHEMA_VERSION
from flylab.provenance.labels import Evidence, Labeled

FloatArray = npt.NDArray[np.float64]

#: Evidence labels a motor decoder may carry (spec 9.2): a hand-made mapping is ENGINEERED,
#: a fitted one is LEARNED. Anything else (EMPIRICAL, ...) would overstate what a decoder is.
DECODER_EVIDENCE = frozenset({Evidence.ENGINEERED, Evidence.LEARNED})

#: Attached to every action spec so the claim boundary travels with the data.
ACTION_SPEC_NOTE = (
    "Action elements are MuJoCo actuator controls of the upstream body model (a software "
    "interface). They are not a biological motor map."
)

# dtype kinds accepted as actions: float, signed int, unsigned int. Not bool/complex/object.
_NUMERIC_KINDS = frozenset("fiu")


class BodyError(RuntimeError):
    """A body backend was used incorrectly or failed."""


class EpisodeEndedError(BodyError):
    """``step`` was called after the episode ended; call ``reset(seed)`` explicitly."""


class BodyPhysicsError(BodyError):
    """The physics engine reported an invalid state (for example a MuJoCo warning)."""


class InvalidActionError(ValueError):
    """An action does not match the backend's verified action spec."""


class ActuatorKind(enum.Enum):
    """What one action element controls, classified from the compiled MuJoCo model."""

    #: Position servo on a hinge joint; control = target joint angle.
    POSITION_HINGE = "position_hinge"
    #: Position servo on a fixed tendon whose path is hinge joints only;
    #: control = target value of the tendon, a weighted sum of hinge angles.
    POSITION_FIXED_TENDON = "position_fixed_tendon"
    #: MuJoCo adhesion actuator on a body; control = dimensionless adhesion command.
    ADHESION = "adhesion"
    #: Anything else. Units are not established.
    OTHER = "other"


#: Control units per kind. Angles are radians because the flybody model compiles with
#: ``<compiler angle="radian"/>``. ``None`` = not established.
ACTUATOR_UNITS: Mapping[ActuatorKind, str | None] = types.MappingProxyType(
    {
        ActuatorKind.POSITION_HINGE: "rad",
        ActuatorKind.POSITION_FIXED_TENDON: "rad",
        ActuatorKind.ADHESION: "1",
        ActuatorKind.OTHER: None,
    }
)


def _readonly(values: npt.ArrayLike, *, name: str) -> FloatArray:
    arr = np.array(values, dtype=np.float64)  # always a copy (NumPy 1.26 and 2.x alike)
    if arr.dtype.hasobject:  # pragma: no cover - dtype=float64 already guarantees this
        raise TypeError(f"{name} must be numeric")
    arr.setflags(write=False)
    return arr


def _check_time_ms(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number of ms, got {value!r}")


@dataclass(frozen=True)
class ActuatorInfo:
    """One action element.

    Attributes:
        name: Upstream actuator name, as listed in the backend's action spec.
        kind: What the element controls (see `ActuatorKind`).
        minimum: Lower bound of the control, in ``units``.
        maximum: Upper bound of the control, in ``units``.
        units: Control units, or None if not established.
        filter_tau_ms: Time constant of a first-order activation filter between the control and
            the actuator, in ms of simulation time; None if the actuator has no filter.
    """

    name: str
    kind: ActuatorKind
    minimum: float
    maximum: float
    units: str | None
    filter_tau_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("actuator name must be non-empty")
        if math.isnan(self.minimum) or math.isnan(self.maximum) or self.minimum > self.maximum:
            raise ValueError(f"{self.name}: invalid bounds [{self.minimum}, {self.maximum}]")
        if self.filter_tau_ms is not None and not (self.filter_tau_ms > 0):
            raise ValueError(f"{self.name}: filter_tau_ms must be > 0 or None")


@dataclass(frozen=True)
class ActionSpec:
    """The verified action interface of a body backend, read from the backend at runtime.

    Attributes:
        actuators: One entry per action element, **in action order** (which need not equal the
            MuJoCo model's actuator order).
        control_timestep_ms: Simulation time advanced by one ``step`` call (the action rate).
        physics_timestep_ms: Physics integrator step; ``step`` runs several of these.
        source: Where the spec was read from (package, revision, API call).
        dtype: dtype of validated actions. Always ``float64``.
    """

    actuators: tuple[ActuatorInfo, ...]
    control_timestep_ms: float
    physics_timestep_ms: float
    source: str
    dtype: str = "float64"
    schema_version: int = SCHEMA_VERSION
    _minimum: FloatArray = field(init=False, repr=False, compare=False)
    _maximum: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.actuators:
            raise ValueError("an action spec needs at least one actuator")
        names = [a.name for a in self.actuators]
        if len(set(names)) != len(names):
            raise ValueError("actuator names must be unique")
        for label, value in (
            ("control_timestep_ms", self.control_timestep_ms),
            ("physics_timestep_ms", self.physics_timestep_ms),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{label} must be finite and > 0, got {value!r}")
        if self.physics_timestep_ms > self.control_timestep_ms:
            raise ValueError("physics_timestep_ms must not exceed control_timestep_ms")
        if self.dtype != "float64":
            raise ValueError("only float64 actions are supported")
        if not self.source.strip():
            raise ValueError("source must say where the spec was read from")
        object.__setattr__(
            self, "_minimum", _readonly([a.minimum for a in self.actuators], name="minimum")
        )
        object.__setattr__(
            self, "_maximum", _readonly([a.maximum for a in self.actuators], name="maximum")
        )

    @property
    def evidence(self) -> Evidence:
        """Always ENGINEERED: a software interface, not a measured biological quantity."""
        return Evidence.ENGINEERED

    @property
    def dim(self) -> int:
        return len(self.actuators)

    @property
    def shape(self) -> tuple[int]:
        return (self.dim,)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.actuators)

    @property
    def minimum(self) -> FloatArray:
        return self._minimum

    @property
    def maximum(self) -> FloatArray:
        return self._maximum

    def zeros(self) -> FloatArray:
        """The all-zero action. Raises if zero lies outside any element's bounds."""
        z = np.zeros(self.shape, dtype=np.float64)
        bad = [a.name for a in self.actuators if not (a.minimum <= 0.0 <= a.maximum)]
        if bad:
            raise InvalidActionError(f"zero is outside the bounds of: {', '.join(bad[:5])}")
        return z

    def validate(self, action: npt.ArrayLike) -> FloatArray:
        """Return a float64 copy of ``action`` after checking it against this spec.

        Checks shape, numeric dtype, finiteness and bounds. Never clips: an out-of-range action
        is an error, so a decoder bug cannot be hidden by silent saturation.
        """
        arr = np.asarray(action)
        if arr.dtype.kind not in _NUMERIC_KINDS:
            raise InvalidActionError(f"action dtype must be real numeric, got {arr.dtype}")
        if arr.shape != self.shape:
            raise InvalidActionError(f"action shape {arr.shape} != spec shape {self.shape}")
        out = np.array(arr, dtype=np.float64)
        if not np.all(np.isfinite(out)):
            idx = np.flatnonzero(~np.isfinite(out))
            raise InvalidActionError(
                f"action has non-finite values at {[self.names[i] for i in idx[:5]]}"
            )
        outside = np.flatnonzero((out < self._minimum) | (out > self._maximum))
        if outside.size:
            first = int(outside[0])
            raise InvalidActionError(
                f"{outside.size} action value(s) outside bounds, first {self.names[first]}="
                f"{out[first]!r} not in [{self._minimum[first]}, {self._maximum[first]}]"
            )
        return out

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly description, including the claim boundary note."""
        return {
            "schema_version": self.schema_version,
            "evidence": self.evidence.value,
            "note": ACTION_SPEC_NOTE,
            "source": self.source,
            "dim": self.dim,
            "dtype": self.dtype,
            "control_timestep_ms": self.control_timestep_ms,
            "physics_timestep_ms": self.physics_timestep_ms,
            "actuators": [
                {
                    "name": a.name,
                    "kind": a.kind.value,
                    "units": a.units,
                    "minimum": a.minimum,
                    "maximum": a.maximum,
                    "filter_tau_ms": a.filter_tau_ms,
                }
                for a in self.actuators
            ],
        }


@dataclass(frozen=True, eq=False)  # holds arrays: identity equality
class BodyObservation:
    """Observations of the body at one control step.

    Attributes:
        time_ms: Simulation time since the last reset.
        step_index: Number of ``step`` calls since the last reset (0 right after reset).
        values: Observation name -> read-only float64 array (owned copy).
        units: Observation name -> units, or None if not established by FlyLab.
    """

    time_ms: float
    step_index: int
    values: Mapping[str, FloatArray]
    units: Mapping[str, str | None]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_time_ms(self.time_ms, "time_ms")
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        if set(self.units) != set(self.values):
            raise ValueError("units must have exactly one entry per observation")
        frozen = {k: _readonly(v, name=k) for k, v in self.values.items()}
        object.__setattr__(self, "values", types.MappingProxyType(frozen))
        object.__setattr__(self, "units", types.MappingProxyType(dict(self.units)))

    def all_finite(self) -> bool:
        return all(bool(np.all(np.isfinite(v))) for v in self.values.values())

    def non_finite(self) -> tuple[str, ...]:
        """Names of observations that contain NaN or inf."""
        return tuple(k for k, v in self.values.items() if not np.all(np.isfinite(v)))


@dataclass(frozen=True, eq=False)  # holds arrays: identity equality
class BodyStep:
    """Result of one ``BodyBackend.step`` call.

    Attributes:
        observation: Observation after the step.
        action: The validated action that was applied (float64 copy).
        reward: Upstream task reward, if the task defines one. Not a FlyLab metric.
        discount: Upstream discount (0.0 marks a failure termination in dm_env tasks).
        episode_ended: True if the upstream episode ended at this step. The backend then
            refuses further steps until ``reset(seed)``; it never auto-resets.
        physics_warnings: MuJoCo warning name -> number of new occurrences in this step.
            Empty means no warnings.
    """

    observation: BodyObservation
    action: FloatArray
    reward: float | None
    discount: float | None
    episode_ended: bool
    physics_warnings: Mapping[str, int]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _readonly(self.action, name="action"))
        object.__setattr__(
            self, "physics_warnings", types.MappingProxyType(dict(self.physics_warnings))
        )


@dataclass(frozen=True, eq=False)  # holds arrays: identity equality
class NeuralFeatures:
    """Population-level neural activity handed to a motor decoder.

    This is the decoder's input contract only; the neural engine produces it. How neurons are
    grouped into populations is decided outside the body module and must be documented there.

    Attributes:
        time_ms: Simulation time at the end of the estimation window.
        population_names: One name per population.
        rates_hz: Firing rate per population, in Hz (spikes per second of simulation time).
        window_ms: Width of the window over which rates were estimated.
    """

    time_ms: float
    population_names: tuple[str, ...]
    rates_hz: FloatArray
    window_ms: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_time_ms(self.time_ms, "time_ms")
        if not (math.isfinite(self.window_ms) and self.window_ms > 0):
            raise ValueError("window_ms must be finite and > 0")
        if len(set(self.population_names)) != len(self.population_names):
            raise ValueError("population names must be unique")
        rates = _readonly(self.rates_hz, name="rates_hz")
        if rates.shape != (len(self.population_names),):
            raise ValueError(
                f"rates_hz shape {rates.shape} != ({len(self.population_names)},) populations"
            )
        if not np.all(np.isfinite(rates)) or np.any(rates < 0):
            raise ValueError("rates_hz must be finite and >= 0")
        object.__setattr__(self, "rates_hz", rates)


@dataclass(frozen=True, eq=False)  # holds arrays: identity equality
class Action:
    """A decoder output, validated against the spec the decoder was built for.

    Attributes:
        values: Action vector (float64, read-only) in the spec's action order.
        time_ms: Simulation time the action is meant for.
        decoder: Name of the decoder that produced it.
        evidence: ENGINEERED or LEARNED (the decoder's label).
    """

    values: FloatArray
    time_ms: float
    decoder: str
    evidence: Evidence
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _check_time_ms(self.time_ms, "time_ms")
        if self.evidence not in DECODER_EVIDENCE:
            raise ValueError(f"decoder actions must be ENGINEERED or LEARNED, not {self.evidence}")
        object.__setattr__(self, "values", _readonly(self.values, name="values"))


def check_decoder_provenance(provenance: Labeled[str]) -> Labeled[str]:
    """Raise unless ``provenance`` names a decoder labelled ENGINEERED or LEARNED."""
    if not isinstance(provenance, Labeled):
        raise TypeError("decoder provenance must be a flylab.provenance.labels.Labeled")
    if provenance.evidence not in DECODER_EVIDENCE:
        raise ValueError(
            f"a motor decoder must be ENGINEERED or LEARNED, got {provenance.evidence.value}"
        )
    if not (isinstance(provenance.value, str) and provenance.value.strip()):
        raise ValueError("decoder provenance value must be the decoder's non-empty name")
    return provenance


@runtime_checkable
class BodyBackend(Protocol):
    """Spec 9.1. A body backend never auto-resets and never clips actions."""

    def reset(self, seed: int) -> None: ...

    def action_spec(self) -> ActionSpec: ...

    def observe(self) -> BodyObservation: ...

    def step(self, action: npt.ArrayLike) -> BodyStep: ...

    def close(self) -> None: ...


@runtime_checkable
class MotorDecoder(Protocol):
    """Spec 9.2. ``provenance.evidence`` must be ENGINEERED or LEARNED.

    See `check_decoder_provenance`.
    """

    @property
    def provenance(self) -> Labeled[str]: ...

    @property
    def action_spec(self) -> ActionSpec: ...

    def decode(
        self, neural_features: NeuralFeatures, observation: BodyObservation, time_ms: float
    ) -> Action: ...


def summarize_kinds(spec: ActionSpec) -> dict[str, int]:
    """Count of action elements per `ActuatorKind`, for reports."""
    counts: dict[str, int] = {}
    for a in spec.actuators:
        counts[a.kind.value] = counts.get(a.kind.value, 0) + 1
    return counts
