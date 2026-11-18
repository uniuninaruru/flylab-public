# Body backend (flybody) — version matrix, install, compatibility

Status: **PR-05 / milestone M5**, CPU only, **no closed loop** (spec 9.3 forbids one until the
neural and body gates pass separately). Audit date **2026-11-04**. Everything below was run in
the cloud container unless marked UNVERIFIED. Evidence: `docs/acceptance/M5.md`.

## Version matrix

| Component | Version tested | How it is pinned | Status |
|---|---|---|---|
| flybody | 0.1.0 at commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26` (= `main` HEAD on 2026-11-04; commit date 2025-07-30) | git rev in `envs/body/pyproject.toml` + `envs/body/uv.lock` | VERIFIED (cloud container) |
| MuJoCo | 3.17.0 (`mujoco.mj_versionString()` = 3.17.0) | exact pin | VERIFIED |
| dm_control | 1.0.50 (requires `mujoco>=3.17.0`) | exact pin | VERIFIED |
| dm-env | 1.6 | lock | VERIFIED |
| NumPy | **1.26.4** (flybody's `pyproject.toml` pins `numpy==1.26.4`) | lock | VERIFIED |
| Python | 3.12.3 (`envs/body` requires `==3.12.*`) | `envs/body/.python-version` | VERIFIED |
| Other | scipy 1.17.1, h5py 3.16.0, labmaze 1.0.6, PyOpenGL 3.1.10 (60 packages in lock) | lock | VERIFIED |

| Platform | Status |
|---|---|
| Cloud container: Ubuntu 24.04.5 LTS, Linux x86_64, no GPU | **VERIFIED** 2026-11-04 (smoke PASS, 49 tests pass) |
| GitHub Actions `ubuntu-24.04` (`.github/workflows/body.yml`) | **VERIFIED** 2026-11-04 on PR head `73f16ed` ([run](https://github.com/uniuninaruru/flylab/actions/runs/41780013208)): smoke PASS, action_dim 59, 200 finite steps, 0 warnings; 49 tests pass. The runner has no EGL/OSMesa libraries either, so `MUJOCO_GL=disable` was selected. |
| macOS arm64 (owner's Mac) | UNVERIFIED. The lock contains macOS arm64 cp312 wheels for every compiled dependency; nothing was run. |
| RTX host (Windows 11 + WSL2) | UNVERIFIED and not needed: MuJoCo physics runs on the CPU (spec 2.4). |

## Why the body stack has its own environment

- **NumPy conflict.** flybody pins `numpy==1.26.4`; the main FlyLab project requires
  `numpy>=2.0` (and torch 2.15.0 for the GPU stack). One resolution cannot satisfy both, so
  `envs/body` has its own `pyproject.toml` and `uv.lock`. The root `pyproject.toml`/`uv.lock` are
  untouched.
- **Python 3.12 only.** NumPy 1.26.4 and labmaze 1.0.6 publish no cp313 wheels (PyPI file lists,
  checked 2026-11-04).
- **Not on PyPI.** `https://pypi.org/pypi/flybody/json` returns 404, so flybody is installed from
  git at the pinned commit; the commit is recorded at runtime from PEP 610 `direct_url.json`.
- **No CUDA 11.** flybody's `tf`/`ray` extras pin `nvidia-cudnn-cu11==8.9.*`,
  `tensorflow==2.8.0`, and its README creates a conda env with `cudatoolkit=11.8.0`. FlyLab
  installs the **core** package only. CUDA 11 never enters FlyLab's neural stack (spec 2.4).
- **FlyLab itself is not installed into `envs/body`** (its `numpy>=2` would conflict). The body
  env imports FlyLab from the checkout with `PYTHONPATH=src`. `flylab.embodiment` therefore has to
  work on NumPy 1.26 and 2.x; its unit tests run under both.

Consequence for M6: neural engine (NumPy 2 / torch) and body (NumPy 1.26) cannot share one
interpreter today. See `OPEN_QUESTIONS.md` B1.

## Install and run

```bash
uv sync --locked --project envs/body
```

```bash
PYTHONPATH=src uv run --locked --project envs/body python scripts/body_smoke.py
```

```bash
PYTHONPATH=src MUJOCO_GL=disable uv run --locked --project envs/body pytest tests/body tests/unit/test_embodiment.py tests/unit/test_embodiment_gl.py
```

`scripts/body_smoke.py` writes `artifacts/body/smoke-<UTC>.json` (git-ignored) and exits 0 only if
every check passes. In the main environment `tests/body` is skipped (flybody not installed).

Upgrading: change the pins in `envs/body/pyproject.toml`, run `uv lock --project envs/body`, rerun
the smoke script and the tests, then update this file and `SOURCES_AND_LICENSES.md`.

## Headless rendering (measured)

dm_control fixes its OpenGL backend from `MUJOCO_GL` at the first `import dm_control.mujoco`.
Each cell is the result of importing dm_control, stepping a one-sphere model and rendering a
16×16 frame in a fresh interpreter.

| Container state | `egl` | `osmesa` | `glfw` | `disable` |
|---|---|---|---|---|
| As provisioned (no `libEGL.so.1`, no `libOSMesa`; `libGL`, `libGLX_mesa`, `libgbm` present) | **import of dm_control fails**: `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | **import fails**: `... no attribute 'glGetError'` | physics OK, render fails (`DISPLAY` missing) | physics OK; render raises `RuntimeError` by design |
| After `apt-get install libegl1 libegl-mesa0 libosmesa6` (Mesa 25.x) | physics OK, **render OK** (`GL_RENDERER` = llvmpipe, LLVM 20.1.2: software) | physics OK, render OK; a `RuntimeError: cannot schedule new futures after shutdown` is printed at interpreter exit | not tested | — |

So setting `MUJOCO_GL=egl` blindly would break physics too. FlyLab's policy
(`flylab.embodiment.gl`): physics-only runs use `MUJOCO_GL=disable`; the smoke script probes `egl`
then `osmesa` in child processes and uses one only if it rendered a non-empty frame. The smoke test
passes with or without rendering. EGL with the NVIDIA driver on the RTX host is UNVERIFIED.

## Upstream API and observed facts

Inspected in flybody `d015e9b` source and observed with MuJoCo 3.17.0 / dm_control 1.0.50.

| Item | Observation | Source |
|---|---|---|
| Factory | `flybody.fly_envs.walk_imitation(ref_path=None, force_actuators=False, disable_wings=True, traj_indices=None, random_state=None, terminal_com_dist=0.3, joint_filter=0.01)` returns a `dm_control.composer.Environment` | `flybody/fly_envs.py` |
| Inference mode | With `ref_path=None` the task uses `InferenceWalkingTrajectoryLoader`: a synthetic straight trajectory, 300 steps at 2 cm/s, height 0.1278. Reward is the constant 1. | `flybody/tasks/trajectory_loaders.py`, `walk_imitation.py` |
| Episode length | `300 - future_steps(64) - 1 = 235` control steps; observed: episode ends after 235 steps (discount 1.0) | code + run |
| Action spec | `dm_env.specs.BoundedArray`, shape **(59,)**, dtype float64, read at runtime | `env.action_spec()` |
| Action composition | 6 adhesion (`[0, 1]`, dimensionless, no filter), 45 hinge position servos (rad), 8 fixed-tendon position servos (rad; tendon = Σ coef·hinge angle, coef 1.0 or 0.5) | compiled `MjModel` |
| Servo parameters | `kp` ∈ {0.1, 0.4, 0.8}, `biasprm[1] = -kp`; activation filter τ = 10 ms (`joint_filter=0.01` s) | compiled `MjModel` |
| **Action order ≠ model actuator order** | action[0] = `adhere_claw_T1_left`, model actuator 0 = `walker/head_abduct`. Bounds equal the model's `ctrlrange` only when matched **by name**. FlyLab maps by name and refuses any mismatch. | run |
| Timesteps | control 2 ms, physics 0.2 ms (10 substeps), equal to `flybody.tasks.constants._WALK_CONTROL_TIMESTEP/_WALK_PHYSICS_TIMESTEP`. (`fruitfly.xml` itself declares 0.1 ms; the task overrides it.) | run + code |
| Model units | `fruitfly.xml`: `<compiler angle="radian"/>`, `<option gravity="0 0 -981" density="0.00128" …>`, i.e. cm-g-s with radians; flybody constants are commented `# cm/s` | `fruitfly.xml`, `constants.py` |
| Observations | 12 float64 arrays: `walker/accelerometer` (3), `actuator_activation` (53), `appendages_pos` (21), `force` (18), `gyro` (3), `joints_pos` (85), `joints_vel` (85), `touch` (6), `velocimeter` (3), `world_zaxis` (3), `ref_displacement` (65, 3), `ref_root_quat` (65, 4). Units **not established** by FlyLab (`None`). | run |
| Ghost termination | With the upstream default `terminal_com_dist=0.3`, a zero-action episode ended after **76** steps with discount 0 (the reference moves away at 2 cm/s). `FlybodyConfig` keeps 0.3 by default; smoke tests set `inf`, as upstream's `tests/test_walking_env.py` does (ENGINEERED test setting). | run |
| Action mutation | Upstream `before_step` does `action[np.isnan(action)] = 0.` in place. FlyLab rejects NaN first and passes a private copy. | `walk_imitation.py` |
| Auto-reset | composer `Environment.step()` after a LAST step silently calls `reset()`. FlyLab raises `EpisodeEndedError` instead. | `dm_control/composer/environment.py` |
| Physics errors | composer default `raise_exception_on_physics_error=True` turns new MuJoCo warnings into `PhysicsError`; FlyLab wraps it as `BodyPhysicsError` and also reports per-step deltas of `data.warning.number` (7 counters in MuJoCo 3.17.0). | dm_control source + run |
| Zero-action posture (one run) | Root body height 0.1278 → 0.1293 (cm), `world_zaxis` z-component 0.99986 after 200 steps (0.4 s): the body did not tip over in this run. One-off diagnostic via the private `env.task._walker`; not a claim about stability in general. | run |
| Reproducibility | Same seed → bitwise-identical observations for 150–200 steps on one machine, after in-place re-seeding and with a freshly built env. **Not bitwise-identical across machines:** the same 200-step zero-action rollout (same versions, seed 0) ended with final-observation SHA-256 `9c67b1b5…` in the cloud container and `4e24a79d…` on the GitHub runner. Cause not established; compare across machines with tolerances, never by hash. | tests + smoke logs |
| Throughput | ≈ 196–199 control steps/s in a bare upstream loop; ≈ 60–90 steps/s inside `scripts/body_smoke.py` (adds per-step copies, checks and hashing). Cloud container, wall-clock, single runs: informational, not a benchmark. | runs + smoke logs |

## FlyLab contract (`src/flylab/embodiment/interface.py`)

| Payload | Content | Units / dtype |
|---|---|---|
| `ActionSpec` | per element: name (action order), `ActuatorKind`, bounds, units, filter τ; control and physics timestep; source string | bounds in element units; timesteps in ms of simulation time; float64; label **ENGINEERED** |
| `BodyObservation` | `time_ms`, `step_index`, `values` (name → read-only float64 copy), `units` (name → str or None) | ms; float64 |
| `BodyStep` | observation, applied action, upstream reward/discount, `episode_ended`, `physics_warnings` | — |
| `NeuralFeatures` | decoder input: population names, `rates_hz` (P,), `window_ms` | Hz, ms; float64 |
| `Action` | decoder output values, `time_ms`, decoder name, evidence ENGINEERED/LEARNED | float64 |

Every payload carries `schema_version` (= `flylab.SCHEMA_VERSION`). Actions are validated (shape,
real numeric dtype, finite, within bounds) and never clipped. `ZeroDecoder` is the only decoder:
ENGINEERED, ignores its input, returns zeros (a null control, not a motor map).

## What this is not

- The 59-element action vector is the upstream model's actuator interface. It is **not** a
  biological motor map, and no connectome neuron is mapped to any element.
- `walk_imitation` in inference mode tracks a synthetic trajectory, not recorded fly walking; its
  reward is constant. Nothing here is behavioural evidence.
- flybody's controllers in the Nature (2025) paper are learned policies, not the connectome.

## Not verified / not done

- macOS; the RTX host; EGL with an NVIDIA driver; rendering on GitHub runners (no EGL/OSMesa libraries there).
- Units of the 12 observations; any decoder other than `ZeroDecoder`; FlyGym (spec 4.2, optional).
- Other flybody environments (`walk_on_ball`, `template_task`, flight); `force_actuators=True`.
