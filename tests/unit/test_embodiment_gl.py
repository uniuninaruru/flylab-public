"""MUJOCO_GL handling: never set an untested renderer, never switch after dm_control import."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from flylab.embodiment import gl


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gl.ENV_VAR, raising=False)
    for marker in gl._FROZEN_MARKERS:
        monkeypatch.delitem(sys.modules, marker, raising=False)


def test_default_is_physics_only(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gl.configure_mujoco_gl() == "disable"
    assert os.environ[gl.ENV_VAR] == "disable"


def test_existing_value_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gl.ENV_VAR, "osmesa")
    assert gl.configure_mujoco_gl() == "osmesa"
    assert gl.configure_mujoco_gl("egl") == "egl"  # explicit request wins before import


@pytest.mark.parametrize("value", ["", "1", "off", "EGL", "cuda"])
def test_unknown_values_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        gl.configure_mujoco_gl(value)
    monkeypatch.setenv(gl.ENV_VAR, value)
    if value:  # an empty variable counts as unset
        with pytest.raises(ValueError, match="must be one of"):
            gl.configure_mujoco_gl()


def test_cannot_switch_after_dm_control_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gl.ENV_VAR, "disable")
    monkeypatch.setitem(sys.modules, "dm_control._render", types.ModuleType("dm_control._render"))
    assert gl.configure_mujoco_gl("disable") == "disable"  # same value: fine
    with pytest.raises(RuntimeError, match="already imported"):
        gl.configure_mujoco_gl("egl")


def test_probe_reports_failure_instead_of_raising() -> None:
    result = gl.probe_mujoco_gl("egl", timeout_s=60)
    assert result.backend == "egl"
    if importlib.util.find_spec("dm_control") is None:  # main FlyLab env: no dm_control
        assert not result.physics_ok
        assert not result.render_ok
        assert result.detail is not None
        assert "ModuleNotFoundError" in result.detail
    assert set(result.to_dict()) == {"backend", "physics_ok", "render_ok", "detail"}


def test_probe_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        gl.probe_mujoco_gl("vulkan")


def test_select_falls_back_to_physics_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(name: str, *, timeout_s: float = 0.0) -> gl.GLProbe:
        calls.append(name)
        return gl.GLProbe(name, physics_ok=False, render_ok=False, detail="no lib")

    monkeypatch.setattr(gl, "probe_mujoco_gl", fake)
    chosen, probes = gl.select_headless_backend()
    assert chosen == "disable"
    assert calls == ["egl", "osmesa"]
    assert [p.backend for p in probes] == ["egl", "osmesa"]


def test_select_stops_at_first_working(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(name: str, *, timeout_s: float = 0.0) -> gl.GLProbe:
        return gl.GLProbe(name, physics_ok=True, render_ok=name == "egl", detail=None)

    monkeypatch.setattr(gl, "probe_mujoco_gl", fake)
    chosen, probes = gl.select_headless_backend()
    assert chosen == "egl"
    assert len(probes) == 1


def test_scrub_removes_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path("/home/someone")))
    assert gl._scrub("File /home/someone/x.py") == "File ~/x.py"
