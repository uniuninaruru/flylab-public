"""Headless flybody smoke tests (spec 9.1, gate M5). Run only in the isolated body environment:

    PYTHONPATH=src MUJOCO_GL=disable uv run --locked --project envs/body pytest tests/body

Skipped in the main FlyLab environment, where flybody is deliberately not installed.
A zero-action fly may fall or fail to walk; that alone is not a failure here.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from collections.abc import Iterator

import numpy as np
import pytest

from flylab.embodiment import (
    ActuatorKind,
    BodyBackend,
    BodyError,
    EpisodeEndedError,
    FlybodyBackend,
    FlybodyConfig,
    InvalidActionError,
    ZeroDecoder,
    body_stack_versions,
)
from flylab.embodiment.interface import NeuralFeatures

if importlib.util.find_spec("flybody") is None:  # find_spec does not import dm_control
    pytest.skip(
        "flybody is installed only in the isolated envs/body environment",
        allow_module_level=True,
    )

STEPS = 150  # >= 100 (spec 9.1); the built-in reference trajectory allows 235 steps

# ENGINEERED for testing only: disable upstream's ghost-distance termination, as upstream's
# tests/test_walking_env.py does, so the zero-action body can be stepped.
SMOKE_CONFIG = FlybodyConfig(terminal_com_dist=math.inf)


@pytest.fixture(scope="module")
def backend() -> Iterator[FlybodyBackend]:
    b = FlybodyBackend(SMOKE_CONFIG)
    b.reset(0)
    yield b
    b.close()


def _sweep_action(b: FlybodyBackend, i: int) -> np.ndarray:
    """Deterministic, spec-valid, non-zero action (stays strictly inside the bounds)."""
    spec = b.action_spec()
    mid = (spec.minimum + spec.maximum) / 2
    half = (spec.maximum - spec.minimum) / 2
    return np.asarray(mid + 0.25 * half * np.sin(0.05 * i + np.arange(spec.dim)))


def _rollout(b: FlybodyBackend, seed: int, steps: int, sweep: bool) -> list[dict[str, np.ndarray]]:
    b.reset(seed)
    spec = b.action_spec()
    out = [dict(b.observe().values)]
    for i in range(steps):
        result = b.step(_sweep_action(b, i) if sweep else spec.zeros())
        out.append(dict(result.observation.values))
    return out


def test_versions_recorded() -> None:
    v = body_stack_versions()
    for key in ("mujoco", "dm-control", "flybody", "numpy", "python"):
        assert v[key], key
    assert v["flybody_commit"], "flybody should be installed from a pinned git commit"


def test_action_spec_read_from_env(backend: FlybodyBackend) -> None:
    spec = backend.action_spec()
    upstream = backend.env.action_spec()
    assert spec.shape == tuple(upstream.shape)
    assert spec.names == tuple(str(upstream.name).split("\t"))
    np.testing.assert_array_equal(spec.minimum, np.broadcast_to(upstream.minimum, spec.shape))
    np.testing.assert_array_equal(spec.maximum, np.broadcast_to(upstream.maximum, spec.shape))
    assert np.all(spec.minimum < spec.maximum)
    assert all(a.kind is not ActuatorKind.OTHER and a.units for a in spec.actuators)
    constants = importlib.import_module("flybody.tasks.constants")
    assert math.isclose(spec.control_timestep_ms, constants._WALK_CONTROL_TIMESTEP * 1000)
    assert math.isclose(spec.physics_timestep_ms, constants._WALK_PHYSICS_TIMESTEP * 1000)
    assert "flybody@" in spec.source
    print(f"observed action_dim={spec.dim}")


def test_action_order_is_not_model_actuator_order(backend: FlybodyBackend) -> None:
    """Upstream action i != MuJoCo actuator i, so FlyLab maps by name.

    Pinned for flybody d015e9b; if it changes, re-verify docs/BODY_BACKEND.md.
    """
    mujoco = importlib.import_module("mujoco")
    model = backend.env.physics.model.ptr
    model_names = [
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").split("/")[-1]
        for i in range(model.nu)
    ]
    spec = backend.action_spec()
    assert sorted(model_names) == sorted(spec.names)
    assert list(spec.names) != model_names


def test_zero_action_steps_finite_without_warnings(backend: FlybodyBackend) -> None:
    backend.reset(0)
    spec = backend.action_spec()
    decoder = ZeroDecoder(spec)  # ENGINEERED null control, no neural input
    features = NeuralFeatures(0.0, (), np.zeros(0), window_ms=spec.control_timestep_ms)
    assert backend.observe().all_finite()
    for i in range(STEPS):
        action = decoder.decode(features, backend.observe(), backend.observe().time_ms)
        result = backend.step(action.values)
        assert result.observation.all_finite(), result.observation.non_finite()
        assert dict(result.physics_warnings) == {}
        assert result.observation.step_index == i + 1
        assert not result.episode_ended
    assert math.isclose(backend.observe().time_ms, STEPS * spec.control_timestep_ms, rel_tol=1e-9)


def test_same_seed_reproduces_rollout(backend: FlybodyBackend) -> None:
    a = _rollout(backend, seed=3, steps=STEPS, sweep=True)
    b = _rollout(backend, seed=3, steps=STEPS, sweep=True)
    with FlybodyBackend(SMOKE_CONFIG) as fresh:
        c = _rollout(fresh, seed=3, steps=STEPS, sweep=True)
    with pytest.raises(BodyError, match="reset"):
        fresh.action_spec()  # closed: nothing stale survives
    for x, y, z in zip(a, b, c, strict=True):
        for key in x:
            np.testing.assert_array_equal(x[key], y[key], err_msg=key)
            np.testing.assert_array_equal(x[key], z[key], err_msg=key)


def test_invalid_actions_rejected(backend: FlybodyBackend) -> None:
    backend.reset(0)
    spec = backend.action_spec()
    bad = spec.zeros()
    bad[0] = np.nan
    over = spec.maximum + 0.1
    for action in (np.zeros(spec.dim + 1), bad, over):
        with pytest.raises(InvalidActionError):
            backend.step(action)
    assert backend.observe().step_index == 0  # nothing was applied


def test_no_silent_auto_reset_after_episode_end(backend: FlybodyBackend) -> None:
    backend.reset(0)
    zero = backend.action_spec().zeros()
    for _ in range(10_000):
        if backend.step(zero).episode_ended:
            break
    assert backend.episode_ended
    with pytest.raises(EpisodeEndedError):
        backend.step(zero)
    backend.reset(0)
    assert backend.observe().step_index == 0
    assert backend.step(zero).observation.step_index == 1


def test_backend_protocol(backend: FlybodyBackend) -> None:
    assert isinstance(backend, BodyBackend)
