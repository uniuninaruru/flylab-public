"""Bounded recorder-to-browser sampling, independent of the ASGI event loop."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flylab.experiments.config import ExperimentConfig
from flylab.neural.engine import NeuralStep

HISTORY_MESSAGES = 64
SPIKES_PER_MESSAGE = 128
RASTER_NEURONS = 64
MAX_POPULATIONS = 32


class RunCancelled(Exception):
    """Cooperative stop; an incomplete run is not published as a finished artifact."""


class RunQuotaExceeded(Exception):
    """The server's resource budget was exhausted."""


class EventBuffer:
    """One fixed-size ring per run; subscribers hold only a sequence cursor."""

    def __init__(self, capacity: int = HISTORY_MESSAGES) -> None:
        if not 1 <= capacity <= HISTORY_MESSAGES:
            raise ValueError("invalid event ring capacity")
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.sequence = 0
        self.dropped = 0

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.sequence += 1
            if len(self._events) == self._events.maxlen:
                self.dropped += 1
            self._events.append({"schema_version": 1, **event, "sequence": self.sequence})

    def after(self, cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            available = [e for e in self._events if e["sequence"] > cursor]
            missed = max(0, available[0]["sequence"] - cursor - 1) if available else 0
            return available[:8], missed

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class LiveRecorder:
    def __init__(
        self,
        cfg: ExperimentConfig,
        populations: Mapping[str, NDArray[np.int32]],
        publish: Callable[[dict[str, Any]], None],
        stop: threading.Event,
        deadline: float,
        *,
        max_events: int = 2_000_000,
        pace_ms: int = 0,
    ) -> None:
        self.cfg, self.publish, self.stop = cfg, publish, stop
        self.deadline, self.max_events, self.pace_ms = deadline, max_events, pace_ms
        self.populations = {n: populations[n] for n in cfg.recorded_populations()}
        if len(self.populations) > MAX_POPULATIONS:
            raise RunQuotaExceeded("too many recorded populations")
        self.bin_steps = round(cfg.recording.population_rate_bin_ms / cfg.dt_ms)
        self.counts = dict.fromkeys(self.populations, 0)
        self.spikes: list[dict[str, Any]] = []
        self.omitted = 0
        self.n_events = 0
        # Deterministic, labelled sample, with the all-neuron population only as fallback.
        names = [n for n in self.populations if n != "__all__"] or list(self.populations)
        self.sample: dict[int, str] = {}
        for name in names:
            for idx in self.populations[name][: max(1, RASTER_NEURONS // len(names))]:
                if len(self.sample) < RASTER_NEURONS:
                    self.sample.setdefault(int(idx), name)
        self.sample_idx = np.array(sorted(self.sample), dtype=np.int32)

    def record(self, step: NeuralStep) -> None:
        if self.stop.is_set():
            raise RunCancelled
        if time.monotonic() > self.deadline:
            raise RunQuotaExceeded("wall-time quota exceeded")
        self.n_events += int(step.spike_idx.size)
        if self.n_events > self.max_events:
            raise RunQuotaExceeded("spike-event quota exceeded")
        for name in self.counts:
            self.counts[name] += step.population_spike_counts.get(name, 0)
        selected = np.intersect1d(step.spike_idx, self.sample_idx, assume_unique=True)
        for idx in selected:
            if len(self.spikes) < SPIKES_PER_MESSAGE:
                self.spikes.append(
                    {"t_ms": step.t_ms, "idx": int(idx), "population": self.sample[int(idx)]}
                )
            else:
                self.omitted += 1
        if (step.step + 1) % self.bin_steps:
            return
        end = (step.step + 1) * step.dt_ms
        width = self.bin_steps * step.dt_ms
        self.publish(
            {
                "type": "population_rates",
                "simulation_time_ms": end,
                "bin_ms": width,
                "populations": [
                    {
                        "name": n,
                        "n_neurons": len(self.populations[n]),
                        "rate_hz": count * 1000 / (width * len(self.populations[n]))
                        if len(self.populations[n])
                        else 0.0,
                    }
                    for n, count in self.counts.items()
                ],
                "spikes": self.spikes,
                "spikes_omitted": self.omitted,
                "active_interventions": [
                    {"category": "stimulus", **s.to_dict()}
                    for s in self.cfg.stimuli
                    if s.onset_ms <= end - step.dt_ms < s.offset_ms
                ]
                + [
                    {"category": "intervention", **iv.to_dict()}
                    for iv in self.cfg.interventions
                    if (iv.onset_ms or 0)
                    <= end - step.dt_ms
                    < (iv.offset_ms if iv.offset_ms is not None else self.cfg.duration_ms)
                ],
            }
        )
        self.counts = dict.fromkeys(self.populations, 0)
        self.spikes = []
        self.omitted = 0
        # Optional visual-demo pacing is wall time only; never changes simulated dt or inputs.
        if self.pace_ms and self.stop.wait(self.pace_ms / 1000):
            raise RunCancelled
