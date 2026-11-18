"""LIF parameter labels, units and discretization (docs/MODEL_ASSUMPTIONS.md section 3)."""

from __future__ import annotations

import math

import pytest

from flylab.neural.params import (
    PARAMETER_UNITS,
    SHIU2024_CODE_V1,
    LIFParameters,
    ParameterError,
    discretize,
    labeled_dt,
)
from flylab.provenance.labels import Evidence, Labeled

COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"


def test_preset_values_labels_and_sources() -> None:
    p = SHIU2024_CODE_V1
    assert p.values() == {
        "v_rest": -52.0,
        "v_reset": -52.0,
        "v_threshold": -45.0,
        "tau_m": 20.0,
        "tau_syn": 5.0,
        "t_ref": 2.2,
        "delay": 1.8,
        "w_syn": 0.275,
    }
    for name, lab in p.to_dict().items():
        assert lab["units"] == PARAMETER_UNITS[name]
        if name == "w_syn":
            # A free parameter of a female-brain model: not literature-derived for MaleCNS.
            assert lab["evidence"] == Evidence.ASSUMED.value
            assert "free parameter" in str(lab["rationale"])
        else:
            assert lab["evidence"] == Evidence.LITERATURE_DERIVED.value
            assert COMMIT in str(lab["source"])
            assert "model.py:" in str(lab["source"])
            assert "Methods 'Computational model'" in str(lab["source"])


def test_discretization_at_shiu_dt() -> None:
    d = discretize(SHIU2024_CODE_V1, 0.1)
    assert (d.refractory_steps, d.delay_steps) == (22, 18)
    assert d.a == pytest.approx(math.exp(-0.1 / 20), abs=0)
    assert d.d == pytest.approx(math.exp(-0.1 / 5), abs=0)
    assert d.b == pytest.approx(5 / (5 - 20) * (math.exp(-0.1 / 5) - math.exp(-0.1 / 20)), abs=0)
    assert d.c == pytest.approx(1 - math.exp(-0.1 / 20), abs=0)


def test_dt_must_divide_refractory_and_delay() -> None:
    with pytest.raises(ParameterError, match="not a whole number of steps"):
        discretize(SHIU2024_CODE_V1, 1.0)  # 1.8 ms and 2.2 ms are not whole 1 ms steps
    d = discretize(SHIU2024_CODE_V1, 0.2)
    assert (d.refractory_steps, d.delay_steps) == (11, 9)
    with pytest.raises(ParameterError, match="dt_ms"):
        discretize(SHIU2024_CODE_V1, 0.0)


def _lab(value: float, units: str) -> Labeled[float]:
    return Labeled(value, Evidence.ASSUMED, rationale="test", units=units)


def test_equal_time_constants_use_the_limit() -> None:
    p = SHIU2024_CODE_V1.with_overrides({"tau_syn": _lab(20.0, "ms")})
    d = discretize(p, 0.1)
    a = math.exp(-0.1 / 20)
    assert d.b == pytest.approx(0.1 / 20 * a, rel=1e-15)
    # The general formula approaches the same value as tau_syn -> tau_m.
    near = discretize(SHIU2024_CODE_V1.with_overrides({"tau_syn": _lab(20.0 + 1e-6, "ms")}), 0.1)
    assert near.b == pytest.approx(d.b, rel=1e-6)


def test_overrides_need_labels_units_and_consistency() -> None:
    p = SHIU2024_CODE_V1.with_overrides({"w_syn": _lab(1.0, "mV")})
    assert p.w_syn.value == 1.0
    assert p.w_syn.evidence is Evidence.ASSUMED
    with pytest.raises(ParameterError, match="unknown parameter"):
        SHIU2024_CODE_V1.with_overrides({"tau": _lab(1.0, "ms")})
    with pytest.raises(ParameterError, match="must be in mV"):
        SHIU2024_CODE_V1.with_overrides({"v_rest": _lab(-52.0, "V")})
    with pytest.raises(ParameterError, match="v_reset must be below"):
        SHIU2024_CODE_V1.with_overrides({"v_reset": _lab(-40.0, "mV")})
    with pytest.raises(ParameterError, match="w_syn must be >= 0"):
        SHIU2024_CODE_V1.with_overrides({"w_syn": _lab(-1.0, "mV")})
    with pytest.raises(ParameterError, match="delay must be > 0"):
        SHIU2024_CODE_V1.with_overrides({"delay": _lab(0.0, "ms")})


def test_parameters_cannot_be_bare_numbers() -> None:
    kwargs: dict[str, object] = dict(SHIU2024_CODE_V1.__dict__)
    kwargs["tau_m"] = 20.0
    with pytest.raises(ParameterError, match="Labeled"):
        LIFParameters(**kwargs)  # type: ignore[arg-type]


def test_dt_label() -> None:
    lab = labeled_dt(0.1)
    assert lab.evidence is Evidence.ENGINEERED
    assert lab.units == "ms"
