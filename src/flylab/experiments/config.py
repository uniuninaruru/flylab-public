"""Experiment config: YAML schema v1 and structural validation (spec 8.1).

Validation happens in two stages and fails loudly at each:
1. `parse_config` (this module): types, ranges, unknown keys, references between sections,
   time grid. Needs no data.
2. `flylab.experiments.populations.resolve_populations`: every population must exist in the
   loaded dataset (unknown body ids, columns or values are errors, never silently empty).

Paths in the config are resolved against the current working directory, like every other
`flylab` command.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from flylab.data.graph import DuplicateRule, UnknownSignPolicy
from flylab.data.null_models import NULL_METHODS
from flylab.neural.engine import SUPPORTED_DTYPES
from flylab.neural.params import (
    PARAMETER_SETS,
    PARAMETER_UNITS,
    LIFParameters,
    ParameterError,
    discretize,
)
from flylab.neural.signs import SIGN_MAPPINGS, SignMappingError, SignSource, SignSpec
from flylab.provenance.labels import Evidence, Labeled

CONFIG_SCHEMA_VERSION = 1
BACKENDS = ("numpy", "torch-cpu", "torch-cuda")
#: `torch-cuda` implementations: `eager` (one torch op per phase) or `fused` (Triton kernels,
#: batched host transfers); `auto` picks `fused` when it can run, else `eager`.
ENGINE_IMPLS = ("auto", "eager", "fused")
DATASET_KINDS = ("processed", "csv_fixture", "synthetic_random")
STIMULUS_KINDS = ("injected_current", "forced_spikes")
INTERVENTION_KINDS = ("silence_neurons", "remove_synapses")
SILENCE_MODES = ("block_output", "clamp_to_rest")
ALL_POPULATION = "__all__"
MAX_STEPS = 100_000_000

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
_VARIANT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}$")
_TOL = 1e-9


class ConfigError(ValueError):
    """The experiment config is invalid. The message names the offending field."""


# --- small typed readers ---------------------------------------------------------------------


def _mapping(obj: object, path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(obj).__name__}")
    for k in obj:
        if not isinstance(k, str):
            raise ConfigError(f"{path}: keys must be strings, got {k!r}")
    return obj


def _keys(
    d: Mapping[str, Any], path: str, required: Sequence[str], optional: Sequence[str]
) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ConfigError(f"{path}: missing required key(s) {missing}")
    unknown = sorted(set(d) - set(required) - set(optional))
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s) {unknown}; allowed: {sorted([*required, *optional])}"
        )


def _str(obj: object, path: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise ConfigError(f"{path}: expected a non-empty string, got {obj!r}")
    if pattern is not None and not pattern.match(obj):
        raise ConfigError(f"{path}: {obj!r} does not match {pattern.pattern}")
    return obj


def _choice(obj: object, path: str, choices: Sequence[str]) -> str:
    s = _str(obj, path)
    if s not in choices:
        raise ConfigError(f"{path}: {s!r} is not one of {list(choices)}")
    return s


def _int(obj: object, path: str, lo: int | None = None, hi: int | None = None) -> int:
    if isinstance(obj, bool) or not isinstance(obj, int):
        raise ConfigError(f"{path}: expected an integer, got {obj!r}")
    if lo is not None and obj < lo:
        raise ConfigError(f"{path}: must be >= {lo}, got {obj}")
    if hi is not None and obj > hi:
        raise ConfigError(f"{path}: must be <= {hi}, got {obj}")
    return obj


def _num(obj: object, path: str, lo: float | None = None, lo_open: bool = False) -> float:
    if isinstance(obj, bool) or not isinstance(obj, int | float) or not math.isfinite(obj):
        raise ConfigError(f"{path}: expected a finite number, got {obj!r}")
    x = float(obj)
    if lo is not None and (x < lo or (lo_open and x == lo)):
        raise ConfigError(f"{path}: must be {'>' if lo_open else '>='} {lo}, got {x}")
    return x


def _bool(obj: object, path: str) -> bool:
    if not isinstance(obj, bool):
        raise ConfigError(f"{path}: expected true or false, got {obj!r}")
    return obj


def _list(obj: object, path: str) -> list[Any]:
    if not isinstance(obj, list):
        raise ConfigError(f"{path}: expected a list, got {type(obj).__name__}")
    return obj


def _on_grid(t_ms: float, dt_ms: float, path: str) -> int:
    ratio = t_ms / dt_ms
    n = round(ratio)
    if abs(ratio - n) > _TOL * max(1.0, abs(ratio)):
        raise ConfigError(f"{path}: {t_ms} ms is not a multiple of dt_ms = {dt_ms}")
    return int(n)


# --- config sections ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    kind: str
    path: str | None = None  # processed
    expected_dataset_id: str | None = None  # processed (optional check)
    neurons: str | None = None  # csv_fixture
    edges: str | None = None  # csv_fixture
    duplicate_rule: str | None = None  # csv_fixture
    dataset_id: str | None = None  # csv_fixture
    n_neurons: int | None = None  # synthetic_random
    mean_out_degree: float | None = None
    graph_seed: int | None = None
    inhibitory_fraction: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class PopulationConfig:
    name: str
    body_ids: tuple[int, ...] | None = None
    column: str | None = None
    values: tuple[str, ...] | None = None

    def selector(self) -> dict[str, object]:
        if self.body_ids is not None:
            return {"body_ids": list(self.body_ids)}
        return {"column": self.column, "values": list(self.values or ())}


@dataclass(frozen=True)
class StimulusConfig:
    kind: str
    population: str
    onset_ms: float
    offset_ms: float
    amplitude_mV: float | None = None
    rate_hz: float | None = None
    times_ms: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if v is not None}
        if self.times_ms is not None:
            d["times_ms"] = list(self.times_ms)
        return d


@dataclass(frozen=True)
class InterventionConfig:
    kind: str
    mode: str | None = None
    population: str | None = None
    onset_ms: float | None = None
    offset_ms: float | None = None
    pre_population: str | None = None
    post_population: str | None = None
    fraction: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class RecordingConfig:
    spike_events: bool = True
    population_rate_bin_ms: float = 10.0
    populations: tuple[str, ...] | None = None  # None: every defined population + __all__

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "spike_events": self.spike_events,
            "population_rate_bin_ms": self.population_rate_bin_ms,
        }
        if self.populations is not None:
            d["populations"] = list(self.populations)
        return d


@dataclass(frozen=True)
class ModelConfig:
    parameter_set: str
    overrides: Mapping[str, Labeled[float]] = field(default_factory=dict)

    def parameters(self) -> LIFParameters:
        return PARAMETER_SETS[self.parameter_set].with_overrides(self.overrides)

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter_set": self.parameter_set,
            "overrides": {k: v.to_dict() for k, v in self.overrides.items()},
        }


@dataclass(frozen=True)
class GraphNullConfig:
    """Replace the connectome by a null-model graph before the run (a control, never data).

    `flylab.data.null_models`; the label of the resulting graph is ENGINEERED."""

    method: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "seed": self.seed}


@dataclass(frozen=True)
class SensitivityVariant:
    name: str
    synapse_sign: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "synapse_sign": dict(self.synapse_sign)}


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    dataset: DatasetConfig
    backend: str
    dtype: str
    dt_ms: float
    steps: int
    seed: int
    deterministic: bool
    model: ModelConfig
    synapse_sign: SignSpec
    populations: tuple[PopulationConfig, ...]
    stimuli: tuple[StimulusConfig, ...]
    interventions: tuple[InterventionConfig, ...]
    recording: RecordingConfig
    sensitivity: tuple[SensitivityVariant, ...] = ()
    description: str | None = None
    graph_null: GraphNullConfig | None = None
    engine_impl: str = "auto"

    @property
    def duration_ms(self) -> float:
        return self.steps * self.dt_ms

    def population_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.populations)

    def recorded_populations(self) -> tuple[str, ...]:
        if self.recording.populations is not None:
            return self.recording.populations
        return (*self.population_names(), ALL_POPULATION)

    def without_interventions(self) -> ExperimentConfig:
        """Matched baseline: same seed, initial conditions and stimuli (spec 8.2 `none`)."""
        return replace(self, interventions=(), sensitivity=())

    def with_sign_variant(self, variant: SensitivityVariant) -> ExperimentConfig:
        return replace(
            self, synapse_sign=apply_sign_variant(self.synapse_sign, variant), sensitivity=()
        )

    def to_dict(self) -> dict[str, object]:
        """Canonical form written to the run's frozen `config.yaml`."""
        d: dict[str, object] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
        }
        if self.description is not None:
            d["description"] = self.description
        d["dataset"] = self.dataset.to_dict()
        if self.graph_null is not None:
            d["graph_null"] = self.graph_null.to_dict()
        d |= {
            "backend": self.backend,
            "dtype": self.dtype,
            "dt_ms": self.dt_ms,
            "steps": self.steps,
            "seed": self.seed,
            "deterministic": self.deterministic,
            # only when set: configs written before the key existed keep their canonical form
            **({"engine_impl": self.engine_impl} if self.engine_impl != "auto" else {}),
            "model": self.model.to_dict(),
            "synapse_sign": self.synapse_sign.to_dict(),
            "populations": {p.name: p.selector() for p in self.populations},
            "stimuli": [s.to_dict() for s in self.stimuli],
            "interventions": [i.to_dict() for i in self.interventions],
            "recording": self.recording.to_dict(),
        }
        if self.sensitivity:
            d["sensitivity"] = [v.to_dict() for v in self.sensitivity]
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


# --- section parsers ---------------------------------------------------------------------------


def _parse_dataset(obj: object) -> DatasetConfig:
    p = "dataset"
    d = _mapping(obj, p)
    kind = _choice(d.get("kind"), f"{p}.kind", DATASET_KINDS)
    if kind == "processed":
        _keys(d, p, ("kind", "path"), ("expected_dataset_id",))
        exp = d.get("expected_dataset_id")
        return DatasetConfig(
            kind=kind,
            path=_str(d["path"], f"{p}.path"),
            expected_dataset_id=None if exp is None else _str(exp, f"{p}.expected_dataset_id"),
        )
    if kind == "csv_fixture":
        _keys(d, p, ("kind", "neurons", "edges", "duplicate_rule", "dataset_id"), ())
        return DatasetConfig(
            kind=kind,
            neurons=_str(d["neurons"], f"{p}.neurons"),
            edges=_str(d["edges"], f"{p}.edges"),
            duplicate_rule=_choice(
                d["duplicate_rule"], f"{p}.duplicate_rule", [r.value for r in DuplicateRule]
            ),
            dataset_id=_str(d["dataset_id"], f"{p}.dataset_id"),
        )
    _keys(d, p, ("kind", "n_neurons", "mean_out_degree", "graph_seed"), ("inhibitory_fraction",))
    frac = d.get("inhibitory_fraction", 0.2)
    frac_v = _num(frac, f"{p}.inhibitory_fraction", lo=0.0)
    if frac_v > 1.0:
        raise ConfigError(f"{p}.inhibitory_fraction: must be <= 1")
    return DatasetConfig(
        kind=kind,
        n_neurons=_int(d["n_neurons"], f"{p}.n_neurons", lo=1, hi=2_000_000),
        mean_out_degree=_num(d["mean_out_degree"], f"{p}.mean_out_degree", lo=0.0),
        graph_seed=_int(d["graph_seed"], f"{p}.graph_seed", lo=0),
        inhibitory_fraction=frac_v,
    )


def _parse_model(obj: object) -> ModelConfig:
    p = "model"
    d = _mapping(obj, p)
    _keys(d, p, ("parameter_set",), ("overrides",))
    pset = _choice(d["parameter_set"], f"{p}.parameter_set", sorted(PARAMETER_SETS))
    raw = _mapping(d.get("overrides", {}) or {}, f"{p}.overrides")
    overrides: dict[str, Labeled[float]] = {}
    for name, spec in raw.items():
        q = f"{p}.overrides.{name}"
        if name not in PARAMETER_UNITS:
            raise ConfigError(f"{q}: unknown parameter; known: {sorted(PARAMETER_UNITS)}")
        s = _mapping(spec, q)
        _keys(s, q, ("value", "units", "evidence"), ("source", "rationale"))
        try:
            evidence = Evidence(_str(s["evidence"], f"{q}.evidence"))
        except ValueError as exc:
            raise ConfigError(
                f"{q}.evidence: must be one of {[e.value for e in Evidence]}"
            ) from exc
        units = _str(s["units"], f"{q}.units")
        if units != PARAMETER_UNITS[name]:
            raise ConfigError(f"{q}.units: {name} must be in {PARAMETER_UNITS[name]}, got {units}")
        try:
            overrides[name] = Labeled(
                value=_num(s["value"], f"{q}.value"),
                evidence=evidence,
                source=s.get("source"),
                rationale=s.get("rationale"),
                units=units,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{q}: {exc}") from exc
    try:
        ModelConfig(pset, overrides).parameters()
    except ParameterError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    return ModelConfig(parameter_set=pset, overrides=overrides)


_SIGN_KEYS = ("source", "unknown_sign_policy", "transmitter_column", "mapping", "overrides")


def _sign_overrides(obj: object, path: str) -> dict[str, int]:
    raw = _mapping(obj or {}, path)
    out: dict[str, int] = {}
    for k, v in raw.items():
        out[k] = _int(v, f"{path}.{k}", lo=-1, hi=1)
    return out


def _parse_sign(obj: object, path: str = "synapse_sign") -> SignSpec:
    d = _mapping(obj, path)
    _keys(d, path, ("source", "unknown_sign_policy"), _SIGN_KEYS[2:])
    source = SignSource(_choice(d["source"], f"{path}.source", [s.value for s in SignSource]))
    policy = UnknownSignPolicy(
        _choice(
            d["unknown_sign_policy"],
            f"{path}.unknown_sign_policy",
            [p.value for p in UnknownSignPolicy],
        )
    )
    column = d.get("transmitter_column")
    mapping = d.get("mapping")
    try:
        return SignSpec(
            source=source,
            unknown_sign_policy=policy,
            transmitter_column=None
            if column is None
            else _str(column, f"{path}.transmitter_column"),
            mapping=None
            if mapping is None
            else _choice(mapping, f"{path}.mapping", sorted(SIGN_MAPPINGS)),
            overrides=_sign_overrides(d.get("overrides"), f"{path}.overrides"),
        )
    except SignMappingError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def apply_sign_variant(base: SignSpec, variant: SensitivityVariant) -> SignSpec:
    """Variant fields replace the base fields; `overrides` are merged (variant wins)."""
    merged: dict[str, object] = base.to_dict()
    for k, v in variant.synapse_sign.items():
        if k == "overrides":
            assert isinstance(v, dict)
            merged["overrides"] = {**base.overrides, **v}
        else:
            merged[k] = v
    return _parse_sign(merged, f"sensitivity.{variant.name}.synapse_sign")


def _parse_populations(obj: object) -> tuple[PopulationConfig, ...]:
    d = _mapping(obj, "populations")
    out = []
    for name, spec in d.items():
        q = f"populations.{name}"
        if name == ALL_POPULATION:
            raise ConfigError(f"{q}: {ALL_POPULATION!r} is reserved (all neurons)")
        _str(name, q, _NAME)
        s = _mapping(spec, q)
        if "body_ids" in s:
            _keys(s, q, ("body_ids",), ())
            ids = [
                _int(x, f"{q}.body_ids[{i}]", lo=0)
                for i, x in enumerate(_list(s["body_ids"], f"{q}.body_ids"))
            ]
            if not ids:
                raise ConfigError(f"{q}.body_ids: must not be empty")
            if len(set(ids)) != len(ids):
                raise ConfigError(f"{q}.body_ids: contains duplicates")
            out.append(PopulationConfig(name=name, body_ids=tuple(ids)))
        else:
            _keys(s, q, ("column", "values"), ())
            vals = [
                _str(x, f"{q}.values[{i}]") for i, x in enumerate(_list(s["values"], f"{q}.values"))
            ]
            if not vals:
                raise ConfigError(f"{q}.values: must not be empty")
            if len(set(vals)) != len(vals):
                raise ConfigError(f"{q}.values: contains duplicates")
            out.append(
                PopulationConfig(
                    name=name, column=_str(s["column"], f"{q}.column"), values=tuple(vals)
                )
            )
    return tuple(out)


def _window(d: Mapping[str, Any], q: str, dt: float, duration: float) -> tuple[float, float]:
    """[onset_ms, offset_ms) on the dt grid and inside the run. Omitted = the whole run."""
    if "onset_ms" not in d and "offset_ms" not in d:
        return 0.0, duration
    if "onset_ms" not in d or "offset_ms" not in d:
        raise ConfigError(f"{q}: give both onset_ms and offset_ms")
    on = _num(d["onset_ms"], f"{q}.onset_ms", lo=0.0)
    off = _num(d["offset_ms"], f"{q}.offset_ms", lo=0.0)
    if not on < off:
        raise ConfigError(f"{q}: onset_ms must be < offset_ms")
    if off > duration * (1 + _TOL):
        raise ConfigError(f"{q}.offset_ms: {off} ms is after the end of the run ({duration} ms)")
    _on_grid(on, dt, f"{q}.onset_ms")
    _on_grid(off, dt, f"{q}.offset_ms")
    return on, off


def _parse_stimuli(
    obj: object, pops: set[str], dt: float, duration: float
) -> tuple[StimulusConfig, ...]:
    out = []
    for i, spec in enumerate(_list(obj, "stimuli")):
        q = f"stimuli[{i}]"
        s = _mapping(spec, q)
        kind = _choice(s.get("kind"), f"{q}.kind", STIMULUS_KINDS)
        common = ("kind", "population", "onset_ms", "offset_ms")
        if kind == "injected_current":
            _keys(s, q, (*common, "amplitude_mV"), ())
        else:
            _keys(s, q, common, ("rate_hz", "times_ms"))
            if ("rate_hz" in s) == ("times_ms" in s):
                raise ConfigError(f"{q}: forced_spikes needs exactly one of rate_hz or times_ms")
        pop = _str(s["population"], f"{q}.population")
        if pop not in pops:
            raise ConfigError(f"{q}.population: {pop!r} is not defined under `populations`")
        if "onset_ms" not in s or "offset_ms" not in s:
            raise ConfigError(f"{q}: stimuli need onset_ms and offset_ms")
        on, off = _window(s, q, dt, duration)
        times: tuple[float, ...] | None = None
        if "times_ms" in s:
            raw = _list(s["times_ms"], f"{q}.times_ms")
            if not raw:
                raise ConfigError(f"{q}.times_ms: must not be empty")
            ts = [_num(t, f"{q}.times_ms[{j}]", lo=0.0) for j, t in enumerate(raw)]
            for j, t in enumerate(ts):
                _on_grid(t, dt, f"{q}.times_ms[{j}]")
                if not on <= t < off:
                    raise ConfigError(f"{q}.times_ms[{j}]: {t} ms is outside [onset_ms, offset_ms)")
            if len(set(ts)) != len(ts):
                raise ConfigError(f"{q}.times_ms: contains duplicates")
            times = tuple(sorted(ts))
        out.append(
            StimulusConfig(
                kind=kind,
                population=pop,
                onset_ms=on,
                offset_ms=off,
                amplitude_mV=_num(s["amplitude_mV"], f"{q}.amplitude_mV")
                if "amplitude_mV" in s
                else None,
                rate_hz=_num(s["rate_hz"], f"{q}.rate_hz", lo=0.0, lo_open=True)
                if "rate_hz" in s
                else None,
                times_ms=times,
            )
        )
    return tuple(out)


def _parse_interventions(
    obj: object, pops: set[str], dt: float, duration: float
) -> tuple[InterventionConfig, ...]:
    out = []
    for i, spec in enumerate(_list(obj, "interventions")):
        q = f"interventions[{i}]"
        s = _mapping(spec, q)
        kind = s.get("kind")
        if kind == "none":
            raise ConfigError(f"{q}.kind: write `interventions: []` for the matched baseline")
        kind = _choice(kind, f"{q}.kind", INTERVENTION_KINDS)

        def pop_ref(key: str, q: str = q, s: dict[str, Any] = s) -> str | None:
            if key not in s:
                return None
            name = _str(s[key], f"{q}.{key}")
            if name not in pops:
                raise ConfigError(f"{q}.{key}: {name!r} is not defined under `populations`")
            return name

        if kind == "silence_neurons":
            _keys(s, q, ("kind", "mode", "population"), ("onset_ms", "offset_ms"))
            mode = _choice(s["mode"], f"{q}.mode", SILENCE_MODES)
            on, off = _window(s, q, dt, duration)
            out.append(
                InterventionConfig(
                    kind=kind,
                    mode=mode,
                    population=pop_ref("population"),
                    onset_ms=on,
                    offset_ms=off,
                )
            )
        else:
            _keys(s, q, ("kind",), ("pre_population", "post_population", "fraction", "seed"))
            pre, post = pop_ref("pre_population"), pop_ref("post_population")
            if pre is None and post is None:
                raise ConfigError(
                    f"{q}: remove_synapses needs pre_population and/or post_population"
                )
            frac = _num(s.get("fraction", 1.0), f"{q}.fraction", lo=0.0, lo_open=True)
            if frac > 1.0:
                raise ConfigError(f"{q}.fraction: must be <= 1")
            seed = None if "seed" not in s else _int(s["seed"], f"{q}.seed", lo=0)
            if frac < 1.0 and seed is None:
                raise ConfigError(f"{q}.seed: required when fraction < 1 (reproducible subset)")
            out.append(
                InterventionConfig(
                    kind=kind, pre_population=pre, post_population=post, fraction=frac, seed=seed
                )
            )
    return tuple(out)


def _parse_recording(obj: object, pops: set[str], dt: float, steps: int) -> RecordingConfig:
    p = "recording"
    d = _mapping(obj, p)
    _keys(d, p, (), ("spike_events", "population_rate_bin_ms", "populations"))
    bin_ms = _num(
        d.get("population_rate_bin_ms", 10.0), f"{p}.population_rate_bin_ms", lo=0.0, lo_open=True
    )
    bin_steps = _on_grid(bin_ms, dt, f"{p}.population_rate_bin_ms")
    if bin_steps < 1 or steps % bin_steps:
        raise ConfigError(
            f"{p}.population_rate_bin_ms: {bin_ms} ms must be a whole number of steps that "
            f"divides the run ({steps} steps)"
        )
    names: tuple[str, ...] | None = None
    if "populations" in d:
        raw = [
            _str(x, f"{p}.populations[{i}]")
            for i, x in enumerate(_list(d["populations"], f"{p}.populations"))
        ]
        for i, name in enumerate(raw):
            if name != ALL_POPULATION and name not in pops:
                raise ConfigError(
                    f"{p}.populations[{i}]: {name!r} is not defined under `populations`"
                )
        if len(set(raw)) != len(raw):
            raise ConfigError(f"{p}.populations: contains duplicates")
        names = tuple(raw)
    return RecordingConfig(
        spike_events=_bool(d.get("spike_events", True), f"{p}.spike_events"),
        population_rate_bin_ms=bin_ms,
        populations=names,
    )


def _parse_sensitivity(obj: object, base: SignSpec) -> tuple[SensitivityVariant, ...]:
    out = []
    seen: set[str] = set()
    for i, spec in enumerate(_list(obj, "sensitivity")):
        q = f"sensitivity[{i}]"
        s = _mapping(spec, q)
        _keys(s, q, ("name", "synapse_sign"), ())
        name = _str(s["name"], f"{q}.name", _VARIANT)
        if name in seen or name == "baseline":
            raise ConfigError(f"{q}.name: {name!r} is reserved or used twice")
        seen.add(name)
        sign = _mapping(s["synapse_sign"], f"{q}.synapse_sign")
        _keys(sign, f"{q}.synapse_sign", (), _SIGN_KEYS[1:])
        if not sign:
            raise ConfigError(f"{q}.synapse_sign: a variant must change something")
        if "overrides" in sign:
            sign = {
                **sign,
                "overrides": _sign_overrides(sign["overrides"], f"{q}.synapse_sign.overrides"),
            }
        variant = SensitivityVariant(name=name, synapse_sign=sign)
        apply_sign_variant(base, variant)  # validates the merged spec
        out.append(variant)
    return tuple(out)


_TOP_REQUIRED = (
    "schema_version", "experiment_id", "dataset", "backend", "dt_ms", "steps", "seed",
    "model", "synapse_sign", "populations", "stimuli", "interventions", "recording",
)  # fmt: skip
_TOP_OPTIONAL = (
    "description",
    "dtype",
    "deterministic",
    "sensitivity",
    "graph_null",
    "engine_impl",
)


def _parse_graph_null(obj: object) -> GraphNullConfig | None:
    if obj is None:
        return None
    p = "graph_null"
    d = _mapping(obj, p)
    _keys(d, p, ("method", "seed"), ())
    return GraphNullConfig(
        method=_choice(d["method"], f"{p}.method", NULL_METHODS),
        seed=_int(d["seed"], f"{p}.seed", lo=0, hi=2**63 - 1),
    )


def parse_config(obj: object) -> ExperimentConfig:
    """Validate a decoded YAML document. Raises ConfigError naming the first bad field."""
    d = _mapping(obj, "config")
    _keys(d, "config", _TOP_REQUIRED, _TOP_OPTIONAL)
    version = _int(d["schema_version"], "schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version: this FlyLab reads version {CONFIG_SCHEMA_VERSION}, got {version}"
        )
    backend = _choice(_str(d["backend"], "backend"), "backend", BACKENDS)
    engine_impl = _choice(d.get("engine_impl", "auto"), "engine_impl", ENGINE_IMPLS)
    if engine_impl != "auto" and backend != "torch-cuda":
        raise ConfigError(f"engine_impl: {engine_impl!r} applies to backend 'torch-cuda' only")
    dt = _num(d["dt_ms"], "dt_ms", lo=0.0, lo_open=True)
    steps = _int(d["steps"], "steps", lo=1, hi=MAX_STEPS)
    model = _parse_model(d["model"])
    try:
        discretize(model.parameters(), dt)
    except ParameterError as exc:
        raise ConfigError(f"dt_ms: {exc}") from exc
    duration = steps * dt
    populations = _parse_populations(d["populations"])
    names = {p.name for p in populations}
    sign = _parse_sign(d["synapse_sign"])
    description = d.get("description")
    return ExperimentConfig(
        schema_version=version,
        experiment_id=_str(d["experiment_id"], "experiment_id", _NAME),
        description=None if description is None else _str(description, "description"),
        dataset=_parse_dataset(d["dataset"]),
        backend=backend,
        dtype=_choice(d.get("dtype", "float64"), "dtype", SUPPORTED_DTYPES),
        dt_ms=dt,
        steps=steps,
        seed=_int(d["seed"], "seed", lo=0, hi=2**63 - 1),
        deterministic=_bool(d.get("deterministic", True), "deterministic"),
        engine_impl=engine_impl,
        model=model,
        synapse_sign=sign,
        populations=populations,
        stimuli=_parse_stimuli(d["stimuli"], names, dt, duration),
        interventions=_parse_interventions(d["interventions"], names, dt, duration),
        recording=_parse_recording(d["recording"], names, dt, steps),
        sensitivity=_parse_sensitivity(d.get("sensitivity", []), sign),
        graph_null=_parse_graph_null(d.get("graph_null")),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate a YAML experiment config (yaml.safe_load only)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {p}: {exc}") from exc
    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{p}: not valid YAML: {exc}") from exc
    return parse_config(obj)
