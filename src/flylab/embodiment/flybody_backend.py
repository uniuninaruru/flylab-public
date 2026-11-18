"""`FlybodyBackend`: a `BodyBackend` over TuragaLab/flybody's walking environment (spec 9.1).

Runs only in the isolated body environment (``envs/body``); flybody, dm_control and MuJoCo are
imported lazily, on the first ``reset``, so importing this module needs NumPy only.

Upstream API used (flybody commit d015e9b, inspected 2026-11-04; see docs/BODY_BACKEND.md):

- ``flybody.fly_envs.walk_imitation(ref_path=None, force_actuators, disable_wings,
  random_state, terminal_com_dist, joint_filter)`` -> ``dm_control.composer.Environment``.
  With ``ref_path=None`` the task runs in inference mode on a built-in synthetic trajectory
  (straight walking at 2 cm/s, 300 control steps); its reward is the constant 1.
- ``env.action_spec()`` -> ``dm_env.specs.BoundedArray``; ``.name`` is the tab-joined list of
  actuator names **in action order**, which differs from the MuJoCo model's actuator order.
- ``env.reset()`` / ``env.step(action)`` -> ``dm_env.TimeStep``; ``env.control_timestep()``;
  ``env.physics`` (dm_control ``Physics``; ``.model.ptr`` is the ``mujoco.MjModel``).

Nothing here hardcodes the action dimension: it is read from ``env.action_spec()`` at runtime.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

from flylab.embodiment.gl import configure_mujoco_gl
from flylab.embodiment.interface import (
    ACTUATOR_UNITS,
    ActionSpec,
    ActuatorInfo,
    ActuatorKind,
    BodyError,
    BodyObservation,
    BodyPhysicsError,
    BodyStep,
    EpisodeEndedError,
)

#: Upstream environment factories this adapter has been verified against.
SUPPORTED_ENVS = ("walk_imitation",)

#: Distributions whose versions describe the body stack (recorded in reports).
BODY_DISTRIBUTIONS = ("numpy", "mujoco", "dm-control", "dm-env", "flybody", "scipy", "h5py")

_MAX_SEED = 2**32 - 1  # numpy.random.RandomState accepts [0, 2**32 - 1]


@dataclass(frozen=True)
class FlybodyConfig:
    """Arguments for the upstream environment factory.

    Defaults equal upstream ``walk_imitation`` defaults, except where noted.

    Attributes:
        env_name: Upstream factory in ``flybody.fly_envs``; only ``walk_imitation`` is verified.
        terminal_com_dist: Upstream episode ends when the fly's centre of mass is farther than
            this from the reference "ghost" (model length units; the model is cm-g-s). Upstream
            default 0.3. ``math.inf`` disables this termination, as upstream's own tests do.
        joint_filter_s: Time constant of the joint actuator activation filter, in seconds
            (upstream unit). 0 disables the filter. Upstream default 0.01.
        force_actuators: Upstream option; False = position actuators (upstream default).
        disable_wings: Upstream option; True = wings retracted and without actuators (default).
        mujoco_gl: ``MUJOCO_GL`` value; None keeps the environment's, else ``"disable"``.
    """

    env_name: str = "walk_imitation"
    terminal_com_dist: float = 0.3
    joint_filter_s: float = 0.01
    force_actuators: bool = False
    disable_wings: bool = True
    mujoco_gl: str | None = None

    def __post_init__(self) -> None:
        if self.env_name not in SUPPORTED_ENVS:
            raise ValueError(f"env_name must be one of {SUPPORTED_ENVS}, got {self.env_name!r}")
        if math.isnan(self.terminal_com_dist) or self.terminal_com_dist <= 0:
            raise ValueError("terminal_com_dist must be > 0 (math.inf disables it)")
        if not (math.isfinite(self.joint_filter_s) and self.joint_filter_s >= 0):
            raise ValueError("joint_filter_s must be finite and >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "env_name": self.env_name,
            "terminal_com_dist": (
                "inf" if math.isinf(self.terminal_com_dist) else self.terminal_com_dist
            ),
            "joint_filter_s": self.joint_filter_s,
            "force_actuators": self.force_actuators,
            "disable_wings": self.disable_wings,
            "mujoco_gl": self.mujoco_gl,
        }


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def flybody_revision() -> str | None:
    """Git commit flybody was installed from (PEP 610 ``direct_url.json``), if recorded."""
    try:
        raw = metadata.distribution("flybody").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not raw:
        return None
    vcs = json.loads(raw).get("vcs_info") or {}
    commit = vcs.get("commit_id")
    return str(commit) if commit else None


def body_stack_versions() -> dict[str, object]:
    """Installed versions of the body stack. Never imports dm_control or flybody."""
    versions: dict[str, object] = {name: _dist_version(name) for name in BODY_DISTRIBUTIONS}
    versions["python"] = platform.python_version()
    versions["flybody_commit"] = flybody_revision()
    return versions


def _classify(mujoco: Any, model: Any, aid: int) -> tuple[ActuatorKind, float | None]:
    """Kind and activation-filter time constant (ms) of actuator ``aid`` of ``model``."""
    trn = int(model.actuator_trntype[aid])
    gain = int(model.actuator_gaintype[aid])
    bias = int(model.actuator_biastype[aid])
    dyn = int(model.actuator_dyntype[aid])
    gainprm = model.actuator_gainprm[aid]
    biasprm = model.actuator_biasprm[aid]
    target = int(model.actuator_trnid[aid, 0])

    filter_tau_ms: float | None = None
    if dyn in (int(mujoco.mjtDyn.mjDYN_FILTER), int(mujoco.mjtDyn.mjDYN_FILTEREXACT)):
        filter_tau_ms = float(model.actuator_dynprm[aid, 0]) * 1000.0

    fixed_gain = gain == int(mujoco.mjtGain.mjGAIN_FIXED)
    # MuJoCo <position>: force = kp * ctrl - kp * length (gainprm[0] = kp, biasprm[1] = -kp).
    servo = (
        fixed_gain
        and bias == int(mujoco.mjtBias.mjBIAS_AFFINE)
        and float(biasprm[0]) == 0.0
        and float(gainprm[0]) > 0.0
        and float(biasprm[1]) == -float(gainprm[0])
    )
    hinge = int(mujoco.mjtJoint.mjJNT_HINGE)
    if trn == int(mujoco.mjtTrn.mjTRN_BODY) and fixed_gain:
        if bias == int(mujoco.mjtBias.mjBIAS_NONE):
            return ActuatorKind.ADHESION, filter_tau_ms
    elif servo and trn == int(mujoco.mjtTrn.mjTRN_JOINT):
        if int(model.jnt_type[target]) == hinge:
            return ActuatorKind.POSITION_HINGE, filter_tau_ms
    elif servo and trn == int(mujoco.mjtTrn.mjTRN_TENDON):
        adr, num = int(model.tendon_adr[target]), int(model.tendon_num[target])
        wraps = range(adr, adr + num)
        if num > 0 and all(
            int(model.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_JOINT)
            and int(model.jnt_type[int(model.wrap_objid[w])]) == hinge
            for w in wraps
        ):
            return ActuatorKind.POSITION_FIXED_TENDON, filter_tau_ms
    return ActuatorKind.OTHER, filter_tau_ms


class FlybodyBackend:
    """`BodyBackend` over ``flybody.fly_envs.walk_imitation``.

    Guarantees beyond upstream behaviour:

    - actions are validated against the runtime `ActionSpec` and never clipped;
    - the upstream environment gets a private copy of each action (upstream mutates it);
    - after an episode ends, ``step`` raises `EpisodeEndedError` instead of dm_control's
      silent auto-reset;
    - new MuJoCo warnings are reported per step, and dm_control's ``PhysicsError`` becomes
      `BodyPhysicsError` (the episode is then over).

    Seeding: the first ``reset(seed)`` builds the environment with
    ``numpy.random.RandomState(seed)``; later resets re-seed that generator in place.
    """

    def __init__(self, config: FlybodyConfig | None = None) -> None:
        self.config = config or FlybodyConfig()
        self.mujoco_gl: str | None = None
        self._env: Any = None
        self._mujoco: Any = None
        self._physics_error: type[BaseException] = BodyPhysicsError
        self._spec: ActionSpec | None = None
        self._obs: BodyObservation | None = None
        self._ended = False
        self._step_index = 0
        self._warning_names: tuple[str, ...] = ()

    # -- context manager ---------------------------------------------------------------
    def __enter__(self) -> FlybodyBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- BodyBackend ---------------------------------------------------------------------
    def reset(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SEED:
            raise ValueError(f"seed must be an int in [0, {_MAX_SEED}], got {seed!r}")
        if self._env is None:
            self._env = self._build(seed)
        else:
            self._env.random_state.seed(seed)
        timestep = self._env.reset()
        self._step_index = 0
        self._ended = False
        if self._spec is None:
            self._spec = self._read_action_spec()
        self._obs = self._observation(timestep.observation)

    def action_spec(self) -> ActionSpec:
        if self._spec is None:
            raise BodyError("call reset(seed) before action_spec()")
        return self._spec

    def observe(self) -> BodyObservation:
        if self._obs is None:
            raise BodyError("call reset(seed) before observe()")
        return self._obs

    def step(self, action: npt.ArrayLike) -> BodyStep:
        if self._env is None or self._spec is None:
            raise BodyError("call reset(seed) before step()")
        if self._ended:
            raise EpisodeEndedError("episode ended; call reset(seed) explicitly")
        applied = self._spec.validate(action)
        before = self._warning_counts()
        try:
            timestep = self._env.step(applied.copy())
        except self._physics_error as exc:
            self._ended = True
            raise BodyPhysicsError(str(exc)) from exc
        after = self._warning_counts()
        new = {n: int(a - b) for n, a, b in zip(self._warning_names, after, before, strict=True)}
        self._step_index += 1
        self._ended = bool(timestep.last())
        self._obs = self._observation(timestep.observation)
        return BodyStep(
            observation=self._obs,
            action=applied,
            reward=None if timestep.reward is None else float(np.asarray(timestep.reward)),
            discount=None if timestep.discount is None else float(np.asarray(timestep.discount)),
            episode_ended=self._ended,
            physics_warnings={n: c for n, c in new.items() if c > 0},
        )

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._spec = None  # re-read from the rebuilt env on the next reset
        self._obs = None
        self._ended = False

    # -- extras (not part of the BodyBackend protocol) -----------------------------------
    @property
    def env(self) -> Any:
        """The upstream dm_control environment, for diagnostics only (None before reset)."""
        return self._env

    @property
    def episode_ended(self) -> bool:
        return self._ended

    def render(self, height: int = 240, width: int = 320, camera_id: int = 0) -> Any:
        """Render one RGB frame. Needs a working ``MUJOCO_GL`` (see `flylab.embodiment.gl`)."""
        if self._env is None:
            raise BodyError("call reset(seed) before render()")
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)

    # -- internals -------------------------------------------------------------------------
    def _build(self, seed: int) -> Any:
        self.mujoco_gl = configure_mujoco_gl(self.config.mujoco_gl)  # before dm_control import
        fly_envs = importlib.import_module("flybody.fly_envs")
        self._mujoco = importlib.import_module("mujoco")
        control = importlib.import_module("dm_control.rl.control")
        self._physics_error = control.PhysicsError
        self._warning_names = tuple(
            n for n in self._mujoco.mjtWarning.__members__ if n != "mjNWARNING"
        )
        factory = getattr(fly_envs, self.config.env_name)
        env = factory(
            random_state=np.random.RandomState(seed),
            terminal_com_dist=self.config.terminal_com_dist,
            joint_filter=self.config.joint_filter_s,
            force_actuators=self.config.force_actuators,
            disable_wings=self.config.disable_wings,
        )
        n_warn = len(env.physics.data.warning.number)
        if n_warn != len(self._warning_names):
            env.close()
            raise BodyError(
                f"MuJoCo reports {n_warn} warning counters but names {len(self._warning_names)}"
            )
        return env

    def _warning_counts(self) -> npt.NDArray[np.int64]:
        return np.array(self._env.physics.data.warning.number, dtype=np.int64)

    def _read_action_spec(self) -> ActionSpec:
        upstream = self._env.action_spec()
        names = str(upstream.name).split("\t")
        shape = tuple(int(s) for s in upstream.shape)
        if shape != (len(names),):
            raise BodyError(f"action spec shape {shape} does not match {len(names)} names")
        if np.dtype(upstream.dtype) != np.float64:
            raise BodyError(f"unexpected upstream action dtype {upstream.dtype}")
        minimum = np.broadcast_to(np.asarray(upstream.minimum, dtype=np.float64), shape)
        maximum = np.broadcast_to(np.asarray(upstream.maximum, dtype=np.float64), shape)

        mj = self._mujoco
        model = self._env.physics.model.ptr
        all_names = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)
        ]
        actuators: list[ActuatorInfo] = []
        for i, name in enumerate(names):
            matches = [aid for aid, full in enumerate(all_names) if full.split("/")[-1] == name]
            if len(matches) != 1:
                raise BodyError(f"action {name!r} matches {len(matches)} model actuators")
            aid = matches[0]
            if bool(model.actuator_ctrllimited[aid]):
                lo, hi = (float(v) for v in model.actuator_ctrlrange[aid])
                if (lo, hi) != (float(minimum[i]), float(maximum[i])):
                    raise BodyError(f"{name}: spec bounds differ from the model's ctrlrange")
            kind, tau_ms = _classify(mj, model, aid)
            actuators.append(
                ActuatorInfo(
                    name=name,
                    kind=kind,
                    minimum=float(minimum[i]),
                    maximum=float(maximum[i]),
                    units=ACTUATOR_UNITS[kind],
                    filter_tau_ms=tau_ms,
                )
            )
        commit = flybody_revision()
        return ActionSpec(
            actuators=tuple(actuators),
            control_timestep_ms=float(self._env.control_timestep()) * 1000.0,
            physics_timestep_ms=float(self._env.physics.timestep()) * 1000.0,
            source=(
                f"flybody@{commit or 'unknown-revision'} fly_envs.{self.config.env_name}: "
                "env.action_spec() + compiled MjModel actuator fields, read at runtime"
            ),
        )

    def _observation(self, raw: Mapping[str, Any]) -> BodyObservation:
        values = {str(k): np.array(v, dtype=np.float64) for k, v in raw.items()}
        return BodyObservation(
            time_ms=float(self._env.physics.time()) * 1000.0,
            step_index=self._step_index,
            values=values,
            units=dict.fromkeys(values),  # observation units not established yet (OPEN_QUESTIONS)
        )
