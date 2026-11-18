"""Headless OpenGL selection for MuJoCo/dm_control (``MUJOCO_GL``).

dm_control fixes its OpenGL backend when ``dm_control.mujoco`` is first imported, from the
``MUJOCO_GL`` environment variable. Measured on 2026-11-04 in the cloud container (Ubuntu 24.04,
no GPU, see docs/BODY_BACKEND.md): without the EGL/OSMesa system libraries,
``MUJOCO_GL=egl`` and ``MUJOCO_GL=osmesa`` make ``from dm_control import mujoco`` itself fail,
so physics would not run either. Therefore:

1. Physics-only runs use ``MUJOCO_GL=disable`` (dm_control's explicit "no renderer").
2. A rendering backend is used only after `probe_mujoco_gl` has rendered a frame with it in a
   separate process (the probe cannot run in-process: the choice is frozen at import).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "MUJOCO_GL"
#: dm_control's spelling for "no renderer"; physics still works.
PHYSICS_ONLY = "disable"
#: Values FlyLab accepts. dm_control also accepts synonyms; FlyLab insists on these spellings.
VALID_BACKENDS = ("egl", "osmesa", "glfw", PHYSICS_ONLY)
#: Headless rendering candidates in order of preference (glfw needs a display).
HEADLESS_CANDIDATES = ("egl", "osmesa")

# Imported by dm_control once the backend is frozen.
_FROZEN_MARKERS = ("dm_control._render", "dm_control.mujoco")

_PROBE_CODE = r"""
import json, sys
out = {"physics_ok": False, "render_ok": False, "detail": None}
try:
    from dm_control import mujoco
    xml = ('<mujoco><worldbody><light pos="0 0 1"/>'
           '<geom type="sphere" size=".1"/></worldbody></mujoco>')
    physics = mujoco.Physics.from_xml_string(xml)
    physics.step()
    out["physics_ok"] = True
    pixels = physics.render(height=16, width=16)
    out["render_ok"] = bool(pixels.shape == (16, 16, 3) and pixels.any())
    if not out["render_ok"]:
        out["detail"] = "rendered frame was empty"
except Exception as exc:  # report, never raise: this is a probe
    out["detail"] = f"{type(exc).__name__}: {exc}"[:300]
print("FLYLAB_GL_PROBE " + json.dumps(out))
"""


@dataclass(frozen=True)
class GLProbe:
    """Outcome of trying one ``MUJOCO_GL`` value in a fresh interpreter."""

    backend: str
    physics_ok: bool
    render_ok: bool
    detail: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "physics_ok": self.physics_ok,
            "render_ok": self.render_ok,
            "detail": self.detail,
        }


def _check_backend(value: str) -> str:
    if value not in VALID_BACKENDS:
        raise ValueError(f"{ENV_VAR} must be one of {VALID_BACKENDS}, got {value!r}")
    return value


def _scrub(text: str) -> str:
    """Drop the home directory from error text so probe output cannot carry a user name."""
    home = str(Path.home())
    return text.replace(home, "~") if home not in ("", "/") else text


def configure_mujoco_gl(backend: str | None = None) -> str:
    """Set ``MUJOCO_GL`` before dm_control is imported and return the effective value.

    Args:
        backend: Value to use. None keeps an existing ``MUJOCO_GL``, else selects
            ``"disable"`` (physics only; never an untested renderer).

    Raises:
        ValueError: unknown value.
        RuntimeError: dm_control is already imported with a different value; switching
            in-process is impossible, so the caller must start a new process.
    """
    current = os.environ.get(ENV_VAR)
    wanted = _check_backend(backend if backend is not None else (current or PHYSICS_ONLY))
    if any(m in sys.modules for m in _FROZEN_MARKERS) and current != wanted:
        raise RuntimeError(
            f"dm_control is already imported with {ENV_VAR}={current!r}; cannot switch to "
            f"{wanted!r} in this process"
        )
    os.environ[ENV_VAR] = wanted
    return wanted


def probe_mujoco_gl(
    backend: str, *, timeout_s: float = 120.0, python: str | None = None
) -> GLProbe:
    """Import dm_control, step a tiny model and render one frame with ``backend``.

    Runs in a child interpreter (``python``, default: this one) so the parent's frozen backend
    is untouched. Never raises for a failing backend; the failure is in the result.
    """
    _check_backend(backend)
    env = dict(os.environ)
    env[ENV_VAR] = backend
    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", _PROBE_CODE],
            env=env, capture_output=True, text=True, timeout=timeout_s, check=False,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        return GLProbe(backend, False, False, f"timed out after {timeout_s:g} s")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("FLYLAB_GL_PROBE "):
            data = json.loads(line.removeprefix("FLYLAB_GL_PROBE "))
            detail = data.get("detail")
            return GLProbe(
                backend=backend,
                physics_ok=bool(data.get("physics_ok")),
                render_ok=bool(data.get("render_ok")),
                detail=_scrub(str(detail)) if detail is not None else None,
            )
    tail = proc.stderr.strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
    return GLProbe(backend, False, False, _scrub(tail[0])[:300])


def select_headless_backend(
    candidates: Sequence[str] = HEADLESS_CANDIDATES, *, timeout_s: float = 120.0
) -> tuple[str, tuple[GLProbe, ...]]:
    """First candidate that renders a frame, else ``"disable"``; plus every probe result."""
    probes: list[GLProbe] = []
    for name in candidates:
        result = probe_mujoco_gl(name, timeout_s=timeout_s)
        probes.append(result)
        if result.render_ok:
            return name, tuple(probes)
    return PHYSICS_ONLY, tuple(probes)
