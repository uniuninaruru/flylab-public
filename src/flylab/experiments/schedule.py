"""Compile stimuli and interventions into per-step engine inputs (MODEL_ASSUMPTIONS 5 and 6).

All randomness lives here, never in the engines, so every backend receives identical inputs:
Poisson forced spikes for stimulus k use NumPy PCG64 seeded with SeedSequence([seed, k]).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flylab.data.graph import DirectedGraph
from flylab.experiments.config import InterventionConfig, StimulusConfig
from flylab.neural.engine import StepDrive, StepInterventions
from flylab.neural.params import steps_for

_EMPTY = np.zeros(0, dtype=np.int32)


def _step(t_ms: float, dt_ms: float) -> int:
    # Same exact-ratio rule as the parameters (the config parser already checked the grid).
    return steps_for(t_ms, dt_ms, "time")


@dataclass(frozen=True, eq=False)
class CompiledStimulus:
    index: int
    kind: str
    population: str
    idx: NDArray[np.int32]
    onset_step: int
    offset_step: int  # exclusive
    amplitude_mV: float | None
    rate_hz: float | None
    p_per_step: float | None
    times_steps: frozenset[int] | None


@dataclass(frozen=True, eq=False)
class CompiledSilence:
    index: int
    mode: str
    population: str
    idx: NDArray[np.int32]
    onset_step: int
    offset_step: int  # exclusive


class ExperimentSchedule:
    """Per-step inputs. Must be queried for steps 0, 1, 2, ... in order (Poisson draws)."""

    def __init__(
        self,
        stimuli: Sequence[CompiledStimulus],
        silences: Sequence[CompiledSilence],
        seed: int,
    ) -> None:
        self.stimuli = tuple(stimuli)
        self.silences = tuple(silences)
        self.seed = seed
        self._rngs = {
            s.index: np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, s.index])))
            for s in self.stimuli
            if s.rate_hz is not None
        }
        self._next = 0

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        if step != self._next:
            raise RuntimeError(f"schedule queried for step {step}, expected {self._next}")
        self._next += 1
        cur_idx: list[NDArray[np.int32]] = []
        cur_mv: list[NDArray[np.float64]] = []
        forced: list[NDArray[np.int32]] = []
        for s in self.stimuli:
            if not s.onset_step <= step < s.offset_step:
                continue
            if s.kind == "injected_current":
                assert s.amplitude_mV is not None
                cur_idx.append(s.idx)
                cur_mv.append(np.full(s.idx.size, s.amplitude_mV, dtype=np.float64))
            elif s.times_steps is not None:
                if step in s.times_steps:
                    forced.append(s.idx)
            else:
                assert s.p_per_step is not None
                draws = self._rngs[s.index].random(s.idx.size)
                forced.append(s.idx[draws < s.p_per_step])
        block = [
            c.idx
            for c in self.silences
            if c.mode == "block_output" and c.onset_step <= step < c.offset_step
        ]
        clamp = [
            c.idx
            for c in self.silences
            if c.mode == "clamp_to_rest" and c.onset_step <= step < c.offset_step
        ]
        drive = None
        if cur_idx or forced:
            drive = StepDrive(
                current_idx=np.concatenate(cur_idx) if cur_idx else _EMPTY,
                current_mV=np.concatenate(cur_mv) if cur_mv else np.zeros(0, dtype=np.float64),
                forced_idx=np.unique(np.concatenate(forced)) if forced else _EMPTY,
            )
        iv = None
        if block or clamp:
            iv = StepInterventions(
                block_output_idx=np.concatenate(block) if block else _EMPTY,
                clamp_idx=np.concatenate(clamp) if clamp else _EMPTY,
            )
        return drive, iv


def compile_stimuli(
    stimuli: Sequence[StimulusConfig], pops: Mapping[str, NDArray[np.int32]], dt_ms: float
) -> list[CompiledStimulus]:
    out = []
    for k, s in enumerate(stimuli):
        p = None
        if s.rate_hz is not None:
            # Probability of >= 1 event of a Poisson process of this rate within one step.
            p = 1.0 - math.exp(-s.rate_hz * dt_ms / 1000.0)
        out.append(
            CompiledStimulus(
                index=k,
                kind=s.kind,
                population=s.population,
                idx=pops[s.population],
                onset_step=_step(s.onset_ms, dt_ms),
                offset_step=_step(s.offset_ms, dt_ms),
                amplitude_mV=s.amplitude_mV,
                rate_hz=s.rate_hz,
                p_per_step=p,
                times_steps=None
                if s.times_ms is None
                else frozenset(_step(t, dt_ms) for t in s.times_ms),
            )
        )
    return out


def compile_silences(
    interventions: Sequence[InterventionConfig], pops: Mapping[str, NDArray[np.int32]], dt_ms: float
) -> list[CompiledSilence]:
    out = []
    for k, iv in enumerate(interventions):
        if iv.kind != "silence_neurons":
            continue
        if iv.mode is None or iv.population is None or iv.onset_ms is None or iv.offset_ms is None:
            raise ValueError(f"interventions[{k}]: incomplete silence_neurons entry")
        out.append(
            CompiledSilence(
                index=k,
                mode=iv.mode,
                population=iv.population,
                idx=pops[iv.population],
                onset_step=_step(iv.onset_ms, dt_ms),
                offset_step=_step(iv.offset_ms, dt_ms),
            )
        )
    return out


@dataclass(frozen=True, eq=False)
class RemovedEdges:
    mask: NDArray[np.bool_]  # aligned to the graph's edges
    per_intervention: tuple[tuple[int, NDArray[np.int64]], ...]  # (config index, edge ids)


def select_removed_edges(
    graph: DirectedGraph,
    interventions: Sequence[InterventionConfig],
    pops: Mapping[str, NDArray[np.int32]],
) -> RemovedEdges:
    """Edges deleted by `remove_synapses` (MODEL_ASSUMPTIONS 6). Reproducible given the seed."""
    mask = np.zeros(graph.n_edges, dtype=bool)
    chosen_all: list[tuple[int, NDArray[np.int64]]] = []
    for k, iv in enumerate(interventions):
        if iv.kind != "remove_synapses":
            continue
        ok = np.ones(graph.n_edges, dtype=bool)
        for name, ends in ((iv.pre_population, graph.src), (iv.post_population, graph.dst)):
            if name is not None:
                member = np.zeros(graph.n_neurons, dtype=bool)
                member[pops[name]] = True
                ok &= member[ends]
        candidates = np.flatnonzero(ok).astype(np.int64)
        frac = 1.0 if iv.fraction is None else iv.fraction
        if frac < 1.0:
            # floor(fraction * n); the epsilon absorbs binary rounding such as 0.29 * 100.
            n_pick = math.floor(frac * candidates.size + 1e-9)
            rng = np.random.default_rng(iv.seed)
            pick = np.sort(rng.choice(candidates.size, size=n_pick, replace=False))
            candidates = candidates[pick]
        mask[candidates] = True
        chosen_all.append((k, candidates))
    return RemovedEdges(mask=mask, per_intervention=tuple(chosen_all))
