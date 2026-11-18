"""Body/decoder contracts (spec 9.1, 9.2) without MuJoCo: runs in the main FlyLab environment."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flylab import SCHEMA_VERSION
from flylab.embodiment import (
    Action,
    ActionSpec,
    ActuatorInfo,
    ActuatorKind,
    BodyBackend,
    BodyError,
    BodyObservation,
    BodyStep,
    FlybodyBackend,
    FlybodyConfig,
    InvalidActionError,
    MotorDecoder,
    NeuralFeatures,
    ZeroDecoder,
    check_decoder_provenance,
)
from flylab.embodiment.interface import ACTUATOR_UNITS, DECODER_EVIDENCE
from flylab.provenance.labels import Evidence, Labeled

EMBODIMENT = Path(__file__).resolve().parents[2] / "src" / "flylab" / "embodiment"
HEAVY = ("mujoco", "dm_control", "flybody", "torch")


def _spec(bounds: list[tuple[float, float]] | None = None) -> ActionSpec:
    bounds = bounds or [(0.0, 1.0), (-0.5, 0.5), (-3.0, 3.0)]
    kinds = [ActuatorKind.ADHESION, ActuatorKind.POSITION_HINGE, ActuatorKind.POSITION_HINGE]
    return ActionSpec(
        actuators=tuple(
            ActuatorInfo(f"a{i}", kinds[i % 3], lo, hi, ACTUATOR_UNITS[kinds[i % 3]])
            for i, (lo, hi) in enumerate(bounds)
        ),
        control_timestep_ms=2.0,
        physics_timestep_ms=0.2,
        source="SYNTHETIC test spec",
    )


def _obs() -> BodyObservation:
    return BodyObservation(
        time_ms=0.0, step_index=0, values={"x": np.array([1.0, 2.0])}, units={"x": None}
    )


def _features() -> NeuralFeatures:
    return NeuralFeatures(
        time_ms=10.0, population_names=("p0", "p1"), rates_hz=np.array([1.0, 0.0]), window_ms=10.0
    )


# -- ActionSpec -------------------------------------------------------------------------------


def test_spec_shape_bounds_and_label() -> None:
    spec = _spec()
    assert spec.dim == 3
    assert spec.shape == (3,)
    assert spec.names == ("a0", "a1", "a2")
    np.testing.assert_array_equal(spec.minimum, [0.0, -0.5, -3.0])
    np.testing.assert_array_equal(spec.maximum, [1.0, 0.5, 3.0])
    assert spec.evidence is Evidence.ENGINEERED
    assert spec.schema_version == SCHEMA_VERSION
    assert not spec.minimum.flags.writeable
    d = spec.to_dict()
    assert d["dim"] == 3
    assert d["evidence"] == "ENGINEERED"
    assert "not a biological motor map" in str(d["note"])


def test_validate_accepts_in_bounds_and_returns_float64_copy() -> None:
    spec = _spec()
    src = np.array([1, 0, -3])  # int input is converted, bounds are inclusive
    out = spec.validate(src)
    assert out.dtype == np.float64
    out[0] = 0.5
    assert src[0] == 1  # caller's array untouched


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (np.zeros(2), "shape"),
        (np.zeros((3, 1)), "shape"),
        (np.array([0.0, np.nan, 0.0]), "non-finite"),
        (np.array([0.0, 0.0, np.inf]), "non-finite"),
        (np.array([1.5, 0.0, 0.0]), "outside bounds, first a0"),
        (np.array([0.0, -0.6, 0.0]), "outside bounds, first a1"),
        (np.array([True, False, False]), "dtype"),
        (np.array(["0", "0", "0"]), "dtype"),
    ],
)
def test_validate_rejects_without_clipping(action: np.ndarray, message: str) -> None:
    with pytest.raises(InvalidActionError, match=message):
        _spec().validate(action)


def test_zeros_inside_bounds_or_error() -> None:
    np.testing.assert_array_equal(_spec().zeros(), np.zeros(3))
    with pytest.raises(InvalidActionError, match="zero is outside"):
        _spec([(0.1, 1.0), (-1.0, 1.0)]).zeros()


def test_spec_invariants() -> None:
    a = ActuatorInfo("a", ActuatorKind.OTHER, -1.0, 1.0, None)
    with pytest.raises(ValueError, match="unique"):
        ActionSpec((a, a), 2.0, 0.2, "x")
    with pytest.raises(ValueError, match="at least one"):
        ActionSpec((), 2.0, 0.2, "x")
    with pytest.raises(ValueError, match="control_timestep_ms"):
        ActionSpec((a,), 0.0, 0.2, "x")
    with pytest.raises(ValueError, match="must not exceed"):
        ActionSpec((a,), 0.1, 0.2, "x")
    with pytest.raises(ValueError, match="source"):
        ActionSpec((a,), 2.0, 0.2, " ")
    with pytest.raises(ValueError, match="invalid bounds"):
        ActuatorInfo("b", ActuatorKind.OTHER, 1.0, -1.0, None)
    with pytest.raises(ValueError, match="filter_tau_ms"):
        ActuatorInfo("b", ActuatorKind.OTHER, -1.0, 1.0, None, filter_tau_ms=0.0)


def test_actuator_units_cover_every_kind() -> None:
    assert set(ACTUATOR_UNITS) == set(ActuatorKind)
    assert ACTUATOR_UNITS[ActuatorKind.OTHER] is None  # unknown stays unknown


# -- payloads ---------------------------------------------------------------------------------


def test_observation_is_read_only_copy() -> None:
    raw = np.array([1.0, np.nan])
    obs = BodyObservation(time_ms=2.0, step_index=1, values={"x": raw}, units={"x": None})
    raw[0] = 99.0
    assert obs.values["x"][0] == 1.0
    assert not obs.values["x"].flags.writeable
    assert not obs.all_finite()
    assert obs.non_finite() == ("x",)
    with pytest.raises(TypeError):
        obs.values["y"] = np.zeros(1)  # type: ignore[index]


def test_observation_rejects_bad_metadata() -> None:
    with pytest.raises(ValueError, match="units"):
        BodyObservation(time_ms=0.0, step_index=0, values={"x": np.zeros(1)}, units={})
    with pytest.raises(ValueError, match="time_ms"):
        BodyObservation(time_ms=-1.0, step_index=0, values={}, units={})


def test_body_step_freezes_action_and_warnings() -> None:
    step = BodyStep(
        observation=_obs(),
        action=np.zeros(3),
        reward=None,
        discount=None,
        episode_ended=False,
        physics_warnings={},
    )
    assert not step.action.flags.writeable
    assert dict(step.physics_warnings) == {}


def test_neural_features_validation() -> None:
    f = _features()
    assert f.rates_hz.shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        NeuralFeatures(time_ms=1.0, population_names=("p",), rates_hz=np.zeros(2), window_ms=1.0)
    with pytest.raises(ValueError, match=">= 0"):
        NeuralFeatures(
            time_ms=1.0, population_names=("p",), rates_hz=np.array([-1.0]), window_ms=1.0
        )
    with pytest.raises(ValueError, match="window_ms"):
        NeuralFeatures(time_ms=1.0, population_names=(), rates_hz=np.zeros(0), window_ms=0.0)


# -- decoders and provenance ------------------------------------------------------------------


def test_decoder_evidence_is_engineered_or_learned_only() -> None:
    assert frozenset({Evidence.ENGINEERED, Evidence.LEARNED}) == DECODER_EVIDENCE
    ok = Labeled("ridge-v0", Evidence.LEARNED, source="fit run 123")
    assert check_decoder_provenance(ok) is ok
    for ev in set(Evidence) - DECODER_EVIDENCE:
        label = Labeled("x", ev, source="s", rationale="r")
        with pytest.raises(ValueError, match="ENGINEERED or LEARNED"):
            check_decoder_provenance(label)
    with pytest.raises(ValueError, match="EMPIRICAL"):
        Action(values=np.zeros(1), time_ms=0.0, decoder="x", evidence=Evidence.EMPIRICAL)


def test_zero_decoder_is_engineered_null_control() -> None:
    spec = _spec()
    dec = ZeroDecoder(spec)
    assert isinstance(dec, MotorDecoder)
    assert dec.provenance.evidence is Evidence.ENGINEERED
    assert "not a biological motor mapping" in (dec.provenance.rationale or "")
    act = dec.decode(_features(), _obs(), time_ms=10.0)
    assert act.evidence is Evidence.ENGINEERED
    assert act.decoder == dec.provenance.value
    np.testing.assert_array_equal(spec.validate(act.values), np.zeros(3))
    louder = NeuralFeatures(
        time_ms=10.0, population_names=("p0", "p1"), rates_hz=np.array([500.0, 9.0]), window_ms=1
    )
    np.testing.assert_array_equal(dec.decode(louder, _obs(), 10.0).values, act.values)
    with pytest.raises(ValueError, match="finite"):
        dec.decode(_features(), _obs(), time_ms=float("nan"))


def test_zero_decoder_refuses_spec_without_zero() -> None:
    with pytest.raises(InvalidActionError):
        ZeroDecoder(_spec([(0.1, 1.0)]))


# -- FlybodyBackend without flybody -----------------------------------------------------------


def test_flybody_backend_satisfies_protocol_without_building() -> None:
    backend = FlybodyBackend()
    assert isinstance(backend, BodyBackend)
    for call in (backend.action_spec, backend.observe):
        with pytest.raises(BodyError, match="reset"):
            call()
    with pytest.raises(BodyError, match="reset"):
        backend.step(np.zeros(3))
    backend.close()  # no-op before reset


@pytest.mark.parametrize("seed", [-1, 2**32, 1.0, True])
def test_flybody_backend_rejects_bad_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        FlybodyBackend().reset(seed)  # type: ignore[arg-type]


def test_flybody_config_validation() -> None:
    assert FlybodyConfig().terminal_com_dist == 0.3  # upstream default, not silently changed
    assert FlybodyConfig(terminal_com_dist=float("inf")).to_dict()["terminal_com_dist"] == "inf"
    with pytest.raises(ValueError, match="env_name"):
        FlybodyConfig(env_name="flight_imitation")
    with pytest.raises(ValueError, match="terminal_com_dist"):
        FlybodyConfig(terminal_com_dist=0.0)
    with pytest.raises(ValueError, match="joint_filter_s"):
        FlybodyConfig(joint_filter_s=-1.0)


# -- isolation from heavy stacks --------------------------------------------------------------


def test_importing_embodiment_loads_no_physics_or_torch() -> None:
    code = (
        "import sys, flylab.embodiment, flylab.embodiment.gl; "
        f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_embodiment_never_imports_neural_data_or_torch() -> None:
    forbidden = ("torch", "flylab.neural", "flylab.data", "flylab.api", "fastapi", "paramiko")
    for py in EMBODIMENT.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "import_module":
                names = [str(a.value) for a in node.args if isinstance(a, ast.Constant)]
            for name in names:
                assert not any(name == f or name.startswith(f + ".") for f in forbidden), (
                    f"{py.name} imports {name}"
                )
