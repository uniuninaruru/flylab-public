"""Headless flybody smoke test (spec 9.1; milestone M5). CPU only, no closed loop.

Run in the isolated body environment from the repository root:

    PYTHONPATH=src uv run --locked --project envs/body python scripts/body_smoke.py

Steps: probe headless OpenGL backends (each in a child process); pick EGL/OSMesa only if it
rendered a frame, else MUJOCO_GL=disable; reset with a seed; read and check the action spec from
``env.action_spec()``; step the spec-valid all-zero action N >= 100 times; check finite
observations, no MuJoCo warnings, simulated time; repeat with the same seed and compare.
A fly that falls or does not walk under zero action is not a failure.

Writes artifacts/body/smoke-<UTC>.json, prints ``key=value`` lines; exit 0 only if every check
passes. Only measured values are written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from flylab.embodiment.flybody_backend import FlybodyBackend, FlybodyConfig, body_stack_versions
from flylab.embodiment.gl import PHYSICS_ONLY, configure_mujoco_gl, select_headless_backend
from flylab.embodiment.interface import (
    ActuatorKind,
    BodyObservation,
    BodyPhysicsError,
    InvalidActionError,
)
from flylab.embodiment.interface import summarize_kinds as kind_counts
from flylab.provenance.environment import capture_environment

MIN_STEPS = 100  # spec 9.1 / 9.3: at least 100 finite physics steps


def _digest(obs: BodyObservation) -> str:
    h = hashlib.sha256()
    for name in sorted(obs.values):
        h.update(name.encode())
        h.update(np.ascontiguousarray(obs.values[name]).tobytes())
    return h.hexdigest()


def _rollout(backend: FlybodyBackend, seed: int, steps: int) -> dict[str, Any]:
    backend.reset(seed)
    spec = backend.action_spec()
    zero = spec.zeros()
    digests = [_digest(backend.observe())]
    non_finite: list[str] = []
    warning_steps: dict[str, int] = {}
    physics_error: str | None = None
    ended_at: int | None = None
    completed = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(steps):
            try:
                result = backend.step(zero)
            except BodyPhysicsError as exc:
                physics_error = str(exc)[:300]
                break
            completed += 1
            digests.append(_digest(result.observation))
            non_finite.extend(f"{i}:{n}" for n in result.observation.non_finite())
            for name, count in result.physics_warnings.items():
                warning_steps[name] = warning_steps.get(name, 0) + count
            if result.episode_ended:
                ended_at = i + 1
                break
    final = backend.observe()
    return {
        "seed": seed,
        "steps_requested": steps,
        "steps_completed": completed,
        "episode_ended_at_step": ended_at,
        "final_time_ms": final.time_ms,
        "non_finite": non_finite[:20],
        "mujoco_warnings": warning_steps,
        "physics_error": physics_error,
        "python_warnings": sorted({f"{w.category.__name__}: {w.message}"[:200] for w in caught}),
        "observation_shapes": {k: list(v.shape) for k, v in final.values.items()},
        "digests": digests,
    }


def run(seed: int, steps: int, probe: bool) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "flylab_body_smoke",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": capture_environment(),
        "body_stack": body_stack_versions(),
        "checks": checks,
        "status": "FAIL",
    }
    probes = select_headless_backend() if probe else (PHYSICS_ONLY, ())
    gl, probe_results = probes
    result["gl"] = {"selected": gl, "probes": [p.to_dict() for p in probe_results]}
    configure_mujoco_gl(gl)

    # ENGINEERED for the smoke test only: inf disables upstream's ghost-distance termination
    # (as upstream tests/test_walking_env.py does), so the zero-action body can be stepped.
    config = FlybodyConfig(terminal_com_dist=math.inf, mujoco_gl=gl)
    result["config"] = config.to_dict()
    t0 = time.perf_counter()
    with FlybodyBackend(config) as backend:
        backend.reset(seed)
        build_s = time.perf_counter() - t0
        spec = backend.action_spec()
        upstream_shape = tuple(int(s) for s in backend.env.action_spec().shape)
        result["action_spec"] = spec.to_dict()
        result["action_dim"] = spec.dim
        result["action_kinds"] = kind_counts(spec)
        checks["action_dim_read_from_env"] = spec.shape == upstream_shape and spec.dim > 0
        checks["all_actuators_classified"] = all(
            a.kind is not ActuatorKind.OTHER and a.units is not None for a in spec.actuators
        )
        checks["bounds_finite_and_ordered"] = bool(
            np.all(np.isfinite(spec.minimum))
            and np.all(np.isfinite(spec.maximum))
            and np.all(spec.minimum < spec.maximum)
        )
        try:
            spec.validate(spec.zeros())
            checks["zero_action_valid"] = True
        except InvalidActionError as exc:
            checks["zero_action_valid"] = False
            result["zero_action_error"] = str(exc)
            return result

        t1 = time.perf_counter()
        first = _rollout(backend, seed, steps)
        rollout_s = time.perf_counter() - t1
        second = _rollout(backend, seed, steps)

        checks["steps_completed"] = first["steps_completed"] == steps >= MIN_STEPS
        checks["observations_finite"] = not first["non_finite"]
        checks["no_mujoco_warnings"] = not first["mujoco_warnings"]
        checks["no_physics_error"] = first["physics_error"] is None
        expected_ms = first["steps_completed"] * spec.control_timestep_ms
        checks["sim_time_matches_steps"] = math.isclose(
            first["final_time_ms"], expected_ms, rel_tol=1e-9, abs_tol=1e-9
        )
        checks["same_seed_reproducible"] = first["digests"] == second["digests"]

        if gl != PHYSICS_ONLY:
            try:
                pixels = np.asarray(backend.render(height=120, width=160))
                checks["render_frame"] = pixels.shape == (120, 160, 3) and bool(pixels.any())
            except Exception as exc:  # report the renderer failure, keep the physics results
                checks["render_frame"] = False
                result["render_error"] = f"{type(exc).__name__}: {exc}"[:300]

    result["rollout"] = {k: v for k, v in first.items() if k != "digests"}
    result["rollout"]["final_observation_sha256"] = first["digests"][-1]
    result["timing_wall_s"] = {
        "note": "informational wall-clock times on this machine, not a benchmark",
        "build_and_first_reset": round(build_s, 3),
        "rollout": round(rollout_s, 3),
    }
    result["status"] = "PASS" if all(checks.values()) else "FAIL"
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200, help=f">= {MIN_STEPS}")
    p.add_argument("--no-render-probe", action="store_true", help="physics only, skip GL probes")
    p.add_argument("--out", type=Path, default=Path("artifacts/body"))
    args = p.parse_args(argv)
    if args.steps < MIN_STEPS:
        p.error(f"--steps must be >= {MIN_STEPS}")

    result = run(args.seed, args.steps, probe=not args.no_render_probe)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"smoke-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    stack = result["body_stack"]
    for key in ("python", "numpy", "mujoco", "dm-control", "flybody", "flybody_commit"):
        print(f"{key}={stack.get(key)}")
    print(f"mujoco_gl={result['gl']['selected']}")
    for probe in result["gl"]["probes"]:
        print(f"gl_probe.{probe['backend']}=render_ok:{probe['render_ok']} ({probe['detail']})")
    print(f"action_dim={result.get('action_dim')}")
    print(f"action_kinds={result.get('action_kinds')}")
    rollout = result.get("rollout", {})
    print(f"steps_completed={rollout.get('steps_completed')}")
    print(f"final_time_ms={rollout.get('final_time_ms')}")
    for name, value in result["checks"].items():
        print(f"check.{name}={value}")
    print(f"smoke_test={result['status']}")
    print(f"log={path}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
