"""Engine contract (spec 7.2) and the backend-independent part of the LIF engines.

`LIFEngineBase` owns the step loop, input validation, population summaries and the checkpoint
format. Backends (`lif_numpy`, `lif_torch` on CPU or CUDA) implement only `_advance`, the per-step
math of docs/MODEL_ASSUMPTIONS.md section 2.1, plus state import/export.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from flylab import SCHEMA_VERSION
from flylab.neural.network import SynapticNetwork
from flylab.neural.params import Discretization

INDEX_CONVENTION = "flylab.data.graph.NeuronIndex: idx i is body_ids[i], body ids ascending"
STEP_UNITS: Mapping[str, str] = {"t_ms": "ms", "dt_ms": "ms", "v": "mV", "g": "mV"}
CHECKPOINT_KIND = "flylab_lif_checkpoint"
#: "Never spiked": far enough in the past that no refractory period reaches t = 0.
NEVER = np.iinfo(np.int64).min // 4
SUPPORTED_DTYPES = ("float32", "float64")

_EMPTY_IDX: NDArray[np.int32] = np.zeros(0, dtype=np.int32)
_EMPTY_F64: NDArray[np.float64] = np.zeros(0, dtype=np.float64)


class EngineInputError(ValueError):
    """Per-step input has the wrong shape, dtype or index range."""


def _as_index(arr: NDArray[np.integer] | None, n: int, name: str) -> NDArray[np.int32]:
    if arr is None:
        return _EMPTY_IDX
    a = np.asarray(arr)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise EngineInputError(f"{name} must be a 1-D integer array")
    if a.size and (int(a.min()) < 0 or int(a.max()) >= n):
        raise EngineInputError(f"{name} has an index outside [0, {n})")
    return a.astype(np.int32, copy=False)


@dataclass(frozen=True, eq=False)
class StepDrive:
    """External input for one step. Injected drive is in mV (R*I_ext); see MODEL_ASSUMPTIONS 5."""

    current_idx: NDArray[np.int32] = field(default_factory=lambda: _EMPTY_IDX)
    current_mV: NDArray[np.float64] = field(default_factory=lambda: _EMPTY_F64)
    forced_idx: NDArray[np.int32] = field(default_factory=lambda: _EMPTY_IDX)


@dataclass(frozen=True, eq=False)
class StepInterventions:
    """Neurons silenced during this step (MODEL_ASSUMPTIONS 6)."""

    block_output_idx: NDArray[np.int32] = field(default_factory=lambda: _EMPTY_IDX)
    clamp_idx: NDArray[np.int32] = field(default_factory=lambda: _EMPTY_IDX)


@dataclass(frozen=True, eq=False)
class NeuralStep:
    """Result of one step. Spike arrays are sparse and aligned; nothing is N-sized."""

    step: int
    t_ms: float  # left edge of the step, = step * dt_ms
    dt_ms: float
    spike_idx: NDArray[np.int32]  # ascending neuron indices
    forced: NDArray[np.bool_]  # aligned with spike_idx: True if the spike was forced
    propagated: NDArray[np.bool_]  # aligned with spike_idx: False if its output was blocked
    n_forced_dropped: int  # forced requests on refractory or clamped neurons
    population_spike_counts: Mapping[str, int]
    schema_version: int = SCHEMA_VERSION
    index_convention: str = INDEX_CONVENTION
    units: Mapping[str, str] = field(default_factory=lambda: dict(STEP_UNITS))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "step": self.step,
            "t_ms": self.t_ms,
            "dt_ms": self.dt_ms,
            "spike_idx": self.spike_idx.tolist(),
            "forced": self.forced.tolist(),
            "propagated": self.propagated.tolist(),
            "n_forced_dropped": self.n_forced_dropped,
            "population_spike_counts": dict(self.population_spike_counts),
            "index_convention": self.index_convention,
            "units": dict(self.units),
            "dtypes": {"spike_idx": "int32", "forced": "bool", "propagated": "bool"},
        }


@dataclass(frozen=True)
class RunSummary:
    steps_run: int
    first_step: int
    end_step: int  # next step to run
    t_end_ms: float
    n_spikes: int
    n_forced_spikes: int
    n_forced_dropped: int
    n_not_propagated: int
    state_finite: bool
    wall_time_s: float  # measured on whatever machine ran it; not a benchmark

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, eq=False)
class EngineState:
    """Backend-neutral copy of the full dynamic state (NumPy arrays)."""

    step: int
    v: NDArray[np.floating]
    g: NDArray[np.floating]
    last_spike: NDArray[np.int64]
    in_flight: tuple[NDArray[np.int32], ...]  # ring slots, slot k holds spikes of a step = k mod D


class Schedule(Protocol):
    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]: ...


class Recorder(Protocol):
    def record(self, step: NeuralStep) -> None: ...


@runtime_checkable
class NeuralEngine(Protocol):
    backend: str

    def reset(self, seed: int) -> None: ...
    def step(
        self, drive: StepDrive | None = None, interventions: StepInterventions | None = None
    ) -> NeuralStep: ...
    def simulate(
        self, steps: int, recorder: Recorder | None = None, schedule: Schedule | None = None
    ) -> RunSummary: ...
    def checkpoint(self, path: str | Path) -> None: ...
    def restore(self, path: str | Path) -> None: ...


@dataclass(frozen=True, eq=False)
class _Advanced:
    spike_idx: NDArray[np.int32]
    forced: NDArray[np.bool_]
    propagated: NDArray[np.bool_]
    n_forced_dropped: int


class LIFEngineBase(ABC):
    """Shared loop; subclasses implement the per-step math in `_advance`."""

    backend = "abstract"

    def __init__(
        self,
        network: SynapticNetwork,
        disc: Discretization,
        dtype: str = "float64",
        populations: Mapping[str, NDArray[np.integer]] | None = None,
    ) -> None:
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}")
        self.network = network
        self.disc = disc
        self.dtype = dtype
        self.n = network.n_neurons
        self._pop_masks: dict[str, NDArray[np.bool_]] = {}
        for name, idx in (populations or {}).items():
            mask = np.zeros(self.n, dtype=bool)
            mask[_as_index(np.asarray(idx), self.n, f"population {name!r}")] = True
            self._pop_masks[name] = mask
        self.seed: int | None = None
        self._step_index = 0

    # --- backend hooks -------------------------------------------------------------------
    @abstractmethod
    def _reset_state(self) -> None: ...

    @abstractmethod
    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced: ...

    @abstractmethod
    def state(self) -> EngineState: ...

    @abstractmethod
    def candidate_potentials(self, neurons: NDArray[np.int64]) -> NDArray[np.float64]:
        """Actual phase-2 potentials from the last step, before threshold/reset (mV).

        Only requested neurons are copied to the host. Used to diagnose threshold flips;
        unavailable before a step, after reset, or after importing state.
        """

    @abstractmethod
    def _load_state(self, state: EngineState) -> None: ...

    @abstractmethod
    def _state_finite(self) -> bool: ...

    def describe(self) -> dict[str, object]:
        return {"backend": self.backend, "dtype": self.dtype}

    def synchronize(self) -> None:
        """Wait for queued device work (CUDA). A no-op for host backends."""
        return None

    @abstractmethod
    def retained_bytes(self) -> dict[str, int]:
        """Bytes of the arrays this engine keeps alive between steps (not library overhead).

        `edges`: edge arrays the engine owns (0 when it reads the network's arrays in place);
        `state`: per-neuron state; `in_flight`: queued spike indices."""

    def device_memory_stats(self) -> dict[str, int] | None:
        """Accelerator allocator counters (bytes), or None on host backends."""
        return None

    def reset_peak_device_memory(self) -> None:
        """Start a new peak-memory window (no-op on host backends)."""
        return None

    # --- contract ------------------------------------------------------------------------
    @property
    def current_step(self) -> int:
        return self._step_index

    def population_indices(self) -> dict[str, NDArray[np.int64]]:
        """Copy the configured populations as ascending neuron indices."""
        return {name: np.flatnonzero(mask) for name, mask in self._pop_masks.items()}

    def state_finite(self) -> bool:
        """Check dynamic arrays, including last-step candidates before reset/clamp."""
        return self._state_finite()

    def reset(self, seed: int) -> None:
        """Initial conditions of MODEL_ASSUMPTIONS 2.1. The engines have no RNG; the seed is kept
        for provenance and must be used by whoever builds stochastic inputs."""
        self.seed = int(seed)
        self._step_index = 0
        self._reset_state()

    def set_state(self, state: EngineState) -> None:
        """Import a backend-neutral state after reset, keeping this engine's seed.

        Validation precedes loading; the step changes only after the backend accepts it.
        This also supports parity resynchronization without a checkpoint round trip.
        """
        if self.seed is None:
            raise RuntimeError("call reset(seed) before set_state()")
        if state.step < 0:
            raise ValueError("state step must be >= 0")
        if any(x.shape != (self.n,) for x in (state.v, state.g, state.last_spike)):
            raise ValueError("state vectors do not match the network size")
        if not np.isfinite(state.v).all() or not np.isfinite(state.g).all():
            raise ValueError("state v and g must be finite")
        if not np.issubdtype(state.last_spike.dtype, np.integer):
            raise ValueError("state last_spike must be an integer array")
        if len(state.in_flight) != self.disc.delay_steps:
            raise ValueError("state has a different number of in-flight slots")
        for slot in state.in_flight:
            _as_index(slot, self.n, "in_flight")
        self._load_state(state)
        self._step_index = int(state.step)

    def step(
        self, drive: StepDrive | None = None, interventions: StepInterventions | None = None
    ) -> NeuralStep:
        if self.seed is None:
            raise RuntimeError("call reset(seed) before step()")
        n = self._step_index
        adv = self._advance(n, *self._validated_inputs(drive, interventions))
        self._step_index = n + 1
        return self._neural_step(n, adv)

    def _validated_inputs(
        self, drive: StepDrive | None, interventions: StepInterventions | None
    ) -> tuple[
        NDArray[np.int32], NDArray[np.float64], NDArray[np.int32], NDArray[np.int32],
        NDArray[np.int32],
    ]:  # fmt: skip
        """Checked per-step inputs in `_advance` argument order (current_idx, current_mV,
        forced_idx, block_idx, clamp_idx)."""
        d = drive or StepDrive()
        iv = interventions or StepInterventions()
        cur_idx = _as_index(d.current_idx, self.n, "current_idx")
        cur_mv = np.asarray(d.current_mV, dtype=np.float64)
        if cur_mv.shape != cur_idx.shape:
            raise EngineInputError("current_mV must align with current_idx")
        if cur_mv.size and not np.isfinite(cur_mv).all():
            raise EngineInputError("current_mV must be finite")
        return (
            cur_idx,
            cur_mv,
            _as_index(d.forced_idx, self.n, "forced_idx"),
            _as_index(iv.block_output_idx, self.n, "block_output_idx"),
            _as_index(iv.clamp_idx, self.n, "clamp_idx"),
        )

    def _neural_step(self, n: int, adv: _Advanced) -> NeuralStep:
        counts = {name: int(m[adv.spike_idx].sum()) for name, m in self._pop_masks.items()}
        return NeuralStep(
            step=n,
            t_ms=n * self.disc.dt_ms,
            dt_ms=self.disc.dt_ms,
            spike_idx=adv.spike_idx,
            forced=adv.forced,
            propagated=adv.propagated,
            n_forced_dropped=adv.n_forced_dropped,
            population_spike_counts=counts,
        )

    def _run_steps(self, steps: int, schedule: Schedule | None) -> Iterator[NeuralStep]:
        """Results of the next `steps` steps, in order. Backends may compute several steps
        before yielding (inputs never depend on outputs inside `simulate`)."""
        if self.seed is None:
            raise RuntimeError("call reset(seed) before simulate()")
        for _ in range(steps):
            drive, iv = schedule.inputs(self._step_index) if schedule else (None, None)
            yield self.step(drive, iv)

    def simulate(
        self, steps: int, recorder: Recorder | None = None, schedule: Schedule | None = None
    ) -> RunSummary:
        if steps < 0:
            raise ValueError("steps must be >= 0")
        first = self._step_index
        totals = {"spikes": 0, "forced": 0, "dropped": 0, "blocked": 0}
        t0 = time.perf_counter()
        for out in self._run_steps(steps, schedule):
            totals["spikes"] += int(out.spike_idx.size)
            totals["forced"] += int(out.forced.sum())
            totals["dropped"] += out.n_forced_dropped
            totals["blocked"] += int((~out.propagated).sum())
            if recorder is not None:
                recorder.record(out)
        self.synchronize()  # device work must be finished before the clock is read
        wall = time.perf_counter() - t0
        return RunSummary(
            steps_run=steps,
            first_step=first,
            end_step=self._step_index,
            t_end_ms=self._step_index * self.disc.dt_ms,
            n_spikes=totals["spikes"],
            n_forced_spikes=totals["forced"],
            n_forced_dropped=totals["dropped"],
            n_not_propagated=totals["blocked"],
            state_finite=self._state_finite(),
            wall_time_s=wall,
        )

    # --- checkpoints (backend-neutral .npz) ----------------------------------------------
    def _meta(self) -> dict[str, object]:
        return {
            "kind": CHECKPOINT_KIND,
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "dtype": self.dtype,
            "seed": self.seed,
            "network_fingerprint_sha256": self.network.fingerprint(),
            "discretization": self.disc.to_dict(),
        }

    def checkpoint(self, path: str | Path) -> None:
        if self.seed is None:
            raise RuntimeError("call reset(seed) before checkpoint()")
        s = self.state()
        lengths = np.array([a.size for a in s.in_flight], dtype=np.int64)
        flat = np.concatenate(s.in_flight) if s.in_flight else _EMPTY_IDX
        with Path(path).open("wb") as fh:
            np.savez(
                fh,
                meta=np.array(json.dumps(self._meta(), sort_keys=True)),
                step=np.array(s.step, dtype=np.int64),
                v=s.v,
                g=s.g,
                last_spike=s.last_spike,
                in_flight=flat.astype(np.int32),
                in_flight_lengths=lengths,
            )

    def restore(self, path: str | Path) -> None:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            mine = self._meta()
            for key in ("kind", "schema_version", "dtype", "network_fingerprint_sha256"):
                if meta.get(key) != mine[key]:
                    raise ValueError(f"checkpoint {key} {meta.get(key)!r} != engine {mine[key]!r}")
            if meta["discretization"] != json.loads(json.dumps(mine["discretization"])):
                raise ValueError("checkpoint discretization differs from this engine's")
            lengths = z["in_flight_lengths"]
            flat = z["in_flight"]
            bounds = np.concatenate([[0], np.cumsum(lengths)])
            slots = tuple(flat[bounds[k] : bounds[k + 1]].copy() for k in range(lengths.size))
            state = EngineState(
                step=int(z["step"]),
                v=z["v"].copy(),
                g=z["g"].copy(),
                last_spike=z["last_spike"].copy(),
                in_flight=slots,
            )
        if len(state.in_flight) != self.disc.delay_steps:
            raise ValueError("checkpoint has a different number of in-flight slots")
        # Load first: if the state is rejected, the engine keeps its previous step and seed.
        self._load_state(state)
        self.seed = int(meta["seed"])
        self._step_index = state.step
