"""Experiment config schema validation (spec 8.1): bad configs fail loudly, naming the field."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from flylab.experiments.config import ConfigError, load_config, parse_config
from flylab.experiments.dataset import load_dataset
from flylab.experiments.populations import resolve_populations


def base(tiny_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "cfg_test",
        "dataset": {
            "kind": "csv_fixture",
            "neurons": str(tiny_dir / "neurons.csv"),
            "edges": str(tiny_dir / "edges.csv"),
            "duplicate_rule": "sum",
            "dataset_id": "synthetic:tiny:v1",
        },
        "backend": "numpy",
        "dt_ms": 0.1,
        "steps": 1000,
        "seed": 1,
        "model": {"parameter_set": "shiu2024_code_v1"},
        "synapse_sign": {"source": "graph", "unknown_sign_policy": "exclude"},
        "populations": {
            "a": {"body_ids": [900001]},
            "bc": {"column": "label", "values": ["syn_b", "syn_c"]},
        },
        "stimuli": [
            {
                "kind": "injected_current",
                "population": "a",
                "onset_ms": 10.0,
                "offset_ms": 50.0,
                "amplitude_mV": 9.0,
            },
            {
                "kind": "forced_spikes",
                "population": "bc",
                "onset_ms": 0.0,
                "offset_ms": 100.0,
                "rate_hz": 20.0,
            },
        ],
        "interventions": [
            {
                "kind": "silence_neurons",
                "mode": "clamp_to_rest",
                "population": "bc",
                "onset_ms": 20.0,
                "offset_ms": 40.0,
            },
            {"kind": "remove_synapses", "pre_population": "a", "fraction": 0.5, "seed": 2},
        ],
        "recording": {"population_rate_bin_ms": 10.0},
    }


@pytest.fixture
def cfg(tiny_dir: Path) -> dict[str, Any]:
    return base(tiny_dir)


@pytest.mark.parametrize("backend", ["numpy", "torch-cpu", "torch-cuda"])
def test_every_backend_parses_without_torch(cfg: dict[str, Any], backend: str) -> None:
    """Parsing needs no torch or GPU; availability is checked when a run starts."""
    c = copy.deepcopy(cfg)
    c["backend"] = backend
    assert parse_config(c).backend == backend


def test_valid_config_parses_and_roundtrips(cfg: dict[str, Any]) -> None:
    c = parse_config(cfg)
    assert c.duration_ms == pytest.approx(100.0)
    assert c.dtype == "float64"
    assert c.deterministic is True
    assert c.recorded_populations() == ("a", "bc", "__all__")
    again = parse_config(yaml.safe_load(c.to_yaml()))
    assert again.to_dict() == c.to_dict()


def _set(d: dict[str, Any], path: str, value: object) -> dict[str, Any]:
    out = copy.deepcopy(d)
    keys = path.split(".")
    node: Any = out
    for k in keys[:-1]:
        node = node[int(k)] if k.isdigit() else node[k]
    last = keys[-1]
    if value is _DELETE:
        del node[int(last) if last.isdigit() else last]
    elif last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value
    return out


_DELETE = object()

BAD: list[tuple[str, object, str]] = [
    ("surprise", 1, r"config: unknown key\(s\) \['surprise'\]"),
    ("schema_version", 2, "schema_version: this FlyLab reads version 1"),
    ("backend", "jax", "backend: 'jax' is not one of"),
    ("dt_ms", 1.0, "dt_ms: .*not a whole number of steps"),
    ("dt_ms", -0.1, "dt_ms: must be > 0"),
    ("steps", 0, "steps: must be >= 1"),
    ("steps", 10.5, "steps: expected an integer"),
    ("seed", True, "seed: expected an integer"),
    ("dtype", "float16", "dtype: 'float16' is not one of"),
    (
        "synapse_sign.unknown_sign_policy",
        _DELETE,
        r"synapse_sign: missing required key\(s\) \['unknown_sign_policy'\]",
    ),
    ("synapse_sign.unknown_sign_policy", "guess", "unknown_sign_policy: 'guess' is not one of"),
    ("synapse_sign.source", "transmitter", "transmitter_column"),
    ("synapse_sign.mapping", "fast_ionotropic_v1", "only apply to source 'transmitter'"),
    ("model.parameter_set", "made_up", "model.parameter_set"),
    (
        "model.overrides",
        {"w_syn": {"value": 1.0, "units": "V", "evidence": "ASSUMED", "rationale": "x"}},
        "units: w_syn must be in mV",
    ),
    (
        "model.overrides",
        {"w_syn": {"value": 1.0, "units": "mV", "evidence": "ASSUMED"}},
        "ASSUMED values need a non-empty `rationale`",
    ),
    (
        "model.overrides",
        {"w_syn": {"value": 1.0, "units": "mV", "evidence": "GUESSED", "rationale": "x"}},
        "evidence: must be one of",
    ),
    (
        "model.overrides",
        {"v_reset": {"value": -40.0, "units": "mV", "evidence": "ASSUMED", "rationale": "x"}},
        "v_reset must be below",
    ),
    (
        "model.overrides",
        {"tau": {"value": 1.0, "units": "ms", "evidence": "ASSUMED", "rationale": "x"}},
        "unknown parameter",
    ),
    ("populations.__all__", {"body_ids": [900001]}, "reserved"),
    ("populations.a", {"body_ids": []}, "must not be empty"),
    ("populations.a", {"body_ids": [900001, 900001]}, "duplicates"),
    ("populations.a", {"column": "label"}, r"missing required key\(s\) \['values'\]"),
    ("populations.a", {"body_ids": [1], "column": "x"}, "unknown key"),
    ("stimuli.0.population", "nobody", "stimuli\\[0\\].population: 'nobody' is not defined"),
    ("stimuli.0.onset_ms", 10.05, "not a multiple of dt_ms"),
    ("stimuli.0.offset_ms", 200.0, "after the end of the run"),
    ("stimuli.0.onset_ms", 60.0, "onset_ms must be < offset_ms"),
    ("stimuli.0.amplitude_mV", _DELETE, "amplitude_mV"),
    ("stimuli.0.kind", "optogenetics", "is not one of"),
    ("stimuli.1.times_ms", [5.0], "exactly one of rate_hz or times_ms"),
    ("stimuli.1.rate_hz", 0.0, "rate_hz: must be > 0"),
    (
        "stimuli.1",
        {
            "kind": "forced_spikes",
            "population": "a",
            "onset_ms": 0.0,
            "offset_ms": 10.0,
            "times_ms": [12.0],
        },
        "outside \\[onset_ms, offset_ms\\)",
    ),
    ("interventions.0.kind", "none", "write `interventions: \\[\\]`"),
    ("interventions.0.mode", "kill", "mode: 'kill' is not one of"),
    ("interventions.0.mode", _DELETE, r"missing required key\(s\) \['mode'\]"),
    ("interventions.0.population", "ghost", "'ghost' is not defined"),
    ("interventions.1.seed", _DELETE, "seed: required when fraction < 1"),
    ("interventions.1.fraction", 1.5, "fraction: must be <= 1"),
    ("interventions.1", {"kind": "remove_synapses"}, "needs pre_population and/or post_population"),
    ("recording.population_rate_bin_ms", 3.0, "divides the run"),
    ("recording.populations", ["a", "zzz"], "'zzz' is not defined"),
    ("recording.spike_events", "yes", "expected true or false"),
    ("experiment_id", "bad id!", "does not match"),
]


@pytest.mark.parametrize(
    ("path", "value", "message"), BAD, ids=[f"{p}={v!r}"[:60] for p, v, _ in BAD]
)
def test_invalid_configs_are_rejected(
    cfg: dict[str, Any], path: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_config(_set(cfg, path, value))


def test_sensitivity_variants_are_validated(cfg: dict[str, Any]) -> None:
    ok = copy.deepcopy(cfg)
    ok["sensitivity"] = [
        {"name": "as-exc", "synapse_sign": {"unknown_sign_policy": "as_excitatory"}}
    ]
    assert parse_config(ok).sensitivity[0].name == "as-exc"
    for variant, msg in [
        ({"name": "x", "synapse_sign": {"source": "transmitter"}}, "unknown key"),
        ({"name": "x", "synapse_sign": {}}, "must change something"),
        ({"name": "baseline", "synapse_sign": {"unknown_sign_policy": "error"}}, "reserved"),
        ({"name": "x", "synapse_sign": {"overrides": {"glutamate": 1}}}, "only apply"),
    ]:
        bad = copy.deepcopy(cfg)
        bad["sensitivity"] = [variant]
        with pytest.raises(ConfigError, match=msg):
            parse_config(bad)


def test_not_a_mapping_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        parse_config(["not", "a", "mapping"])
    p = tmp_path / "bad.yaml"
    p.write_text("a: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(p)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


# --- second stage: the data must contain every population -------------------------------


def _resolve(cfg: dict[str, Any]) -> dict[str, Any]:
    c = parse_config(cfg)
    return resolve_populations(c.populations, load_dataset(c.dataset))


def test_populations_resolve_against_the_fixture(cfg: dict[str, Any]) -> None:
    pops = _resolve(cfg)
    assert pops["a"].tolist() == [0]
    assert pops["bc"].tolist() == [1, 2]
    assert pops["__all__"].size == 7


@pytest.mark.parametrize(
    ("population", "message"),
    [
        ({"body_ids": [123]}, "body_ids: .*not in neuron table"),
        ({"column": "cell_type", "values": ["x"]}, "'cell_type' is not a neuron-table column"),
        ({"column": "label", "values": ["PLACEHOLDER_VERIFY_WITH_DATA"]}, "matches no neuron"),
        ({"column": "label", "values": ["syn_a", "syn_typo"]}, "'syn_typo' matches no neuron"),
    ],
)
def test_unknown_populations_are_rejected(
    cfg: dict[str, Any], population: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        _resolve(_set(cfg, "populations.a", population))
