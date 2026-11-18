"""Body backends and explicit motor decoders (spec section 9; milestone M5, PR-05).

- `interface`: `BodyBackend` / `MotorDecoder` protocols and their payloads (units, shapes,
  dtypes, schema version).
- `flybody_backend`: `FlybodyBackend`, an adapter over TuragaLab/flybody's walking
  environment. flybody/dm_control/MuJoCo live in the isolated ``envs/body`` environment and are
  imported lazily, so this package imports with NumPy alone.
- `decoders`: `ZeroDecoder`, an ENGINEERED placeholder (null control), not a motor map.
- `gl`: headless ``MUJOCO_GL`` selection.

Boundary: must not parse connectome files (no flylab.data imports), must not import flylab.api
or the neural/torch stack. Decoder mappings are ENGINEERED or LEARNED, never EMPIRICAL.
There is no closed loop here: spec 9.3 forbids one until the neural and body gates pass.
"""

from flylab.embodiment.decoders import ZeroDecoder
from flylab.embodiment.flybody_backend import FlybodyBackend, FlybodyConfig, body_stack_versions
from flylab.embodiment.interface import (
    Action,
    ActionSpec,
    ActuatorInfo,
    ActuatorKind,
    BodyBackend,
    BodyError,
    BodyObservation,
    BodyPhysicsError,
    BodyStep,
    EpisodeEndedError,
    InvalidActionError,
    MotorDecoder,
    NeuralFeatures,
    check_decoder_provenance,
)

__all__ = [
    "Action",
    "ActionSpec",
    "ActuatorInfo",
    "ActuatorKind",
    "BodyBackend",
    "BodyError",
    "BodyObservation",
    "BodyPhysicsError",
    "BodyStep",
    "EpisodeEndedError",
    "FlybodyBackend",
    "FlybodyConfig",
    "InvalidActionError",
    "MotorDecoder",
    "NeuralFeatures",
    "ZeroDecoder",
    "body_stack_versions",
    "check_decoder_provenance",
]
