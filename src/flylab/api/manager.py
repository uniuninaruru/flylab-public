"""One cooperative worker, frozen config allowlist and explicit session quotas."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from flylab.api.stream import (
    MAX_POPULATIONS,
    EventBuffer,
    LiveRecorder,
    RunCancelled,
    RunQuotaExceeded,
)
from flylab.experiments.config import ConfigError, ExperimentConfig, load_config, parse_config
from flylab.experiments.dataset import LoadedDataset
from flylab.experiments.runner import LIMITATIONS, run_experiment

TERMINAL = frozenset({"succeeded", "cancelled", "failed"})
NOTICE = "Engineering simulation, not validated behaviour."
LOG = logging.getLogger(__name__)


class RequestError(ValueError):
    """A user request is invalid or outside the server's allowlist."""


class BusyError(RuntimeError):
    """Concurrency is one, with no waiting job queue."""


@dataclass(frozen=True)
class Limits:
    max_steps: int = 100_000
    max_runs: int = 32
    max_neurons: int = 200_000
    max_edges: int = 30_000_000
    max_events: int = 2_000_000
    wall_seconds: float = 300.0
    disk_bytes: int = 512 * 1024 * 1024
    max_clients: int = 4


@dataclass
class Run:
    id: str
    config_name: str
    cfg: ExperimentConfig
    state: str = "starting"
    stop: threading.Event = field(default_factory=threading.Event)
    events: EventBuffer = field(default_factory=EventBuffer)
    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    thread: threading.Thread | None = None
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None


class RunManager:
    def __init__(
        self,
        repo_root: Path,
        *,
        out_root: Path | None = None,
        limits: Limits | None = None,
        pace_ms: int = 0,
    ) -> None:
        self.repo = repo_root.resolve()
        self.out = (out_root or self.repo / "runs/dashboard").resolve()
        self.limits = limits or Limits()
        if pace_ms not in (0, 20, 50, 100):
            raise ValueError("unsupported preview pace")
        self.pace_ms = pace_ms
        self._lock = threading.RLock()
        self._runs: dict[str, Run] = {}
        self._active: str | None = None
        self._clients = 0
        self._closed = False
        self._verified: dict[str, dict[str, Any]] = {}
        self._configs: dict[str, ExperimentConfig] = {}
        root = self.repo / "configs"
        if root.is_symlink():
            raise ValueError("the config allowlist directory cannot be a symlink")
        # Snapshot only these two explicit directory levels. Never follow symlinks.
        for directory in (root, root / "bench"):
            if directory.is_symlink() or not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.yaml")):
                if path.is_symlink() or path.stat().st_size > 65_536:
                    continue
                try:
                    cfg = load_config(path)
                    self._configs[path.relative_to(root).as_posix()] = self._absolute_data(cfg)
                except (ConfigError, OSError, yaml.YAMLError):
                    LOG.warning("Skipped invalid allowlisted config: %s", path.name)

    def _absolute_data(self, cfg: ExperimentConfig) -> ExperimentConfig:
        ds = cfg.dataset

        def absolute(path: str | None) -> str | None:
            return str((self.repo / path).resolve()) if path else None

        return replace(
            cfg,
            dataset=replace(
                ds, path=absolute(ds.path), neurons=absolute(ds.neurons), edges=absolute(ds.edges)
            ),
        )

    def configurations(self) -> list[dict[str, Any]]:
        return [
            {
                "name": n,
                "description": c.description,
                "backend": c.backend,
                "steps": c.steps,
                "seed": c.seed,
                "dt_ms": c.dt_ms,
                "dataset_kind": c.dataset.kind,
                "experiment_id": c.experiment_id,
            }
            for n, c in sorted(self._configs.items())
        ]

    def verified_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._verified.values())

    def validate(self, payload: Any) -> tuple[str, ExperimentConfig]:
        if not isinstance(payload, dict) or set(payload) - {"config", "overrides"}:
            raise RequestError("expected config and optional overrides only")
        name = payload.get("config")
        if not isinstance(name, str) or name not in self._configs:
            raise RequestError("config must exactly match an allowed config name")
        overrides = payload.get("overrides", {})
        if not isinstance(overrides, dict) or set(overrides) - {"seed", "backend", "steps"}:
            raise RequestError("only seed, backend and steps may be overridden")
        for key, high in (("seed", 2**32 - 1), ("steps", self.limits.max_steps)):
            if key in overrides:
                value = overrides[key]
                low = 1 if key == "steps" else 0
                if type(value) is not int or not low <= value <= high:
                    raise RequestError(f"{key} must be an integer in [{low}, {high}]")
        if "backend" in overrides and overrides["backend"] not in (
            "numpy",
            "torch-cpu",
            "torch-cuda",
        ):
            raise RequestError("backend must be numpy, torch-cpu or torch-cuda")
        obj = self._configs[name].to_dict()
        obj.update(overrides)
        cfg = parse_config(obj)
        if cfg.steps > self.limits.max_steps or cfg.sensitivity:
            raise RequestError(
                "steps quota exceeded or sensitivity groups unsupported in dashboard"
            )
        if len(cfg.recorded_populations()) > MAX_POPULATIONS:
            raise RequestError("recorded-population quota exceeded")
        ds = cfg.dataset
        if ds.n_neurons is not None and ds.n_neurons > self.limits.max_neurons:
            raise RequestError("neuron quota exceeded")
        if (
            ds.n_neurons is not None
            and ds.mean_out_degree is not None
            and ds.n_neurons * ds.mean_out_degree > self.limits.max_edges
        ):
            raise RequestError("edge quota exceeded")
        return name, cfg

    def start(self, payload: Any) -> dict[str, Any]:
        name, cfg = self.validate(payload)
        with self._lock:
            if self._closed or self._active is not None:
                raise BusyError("one run at a time; stop or finish the active run first")
            if len(self._runs) >= self.limits.max_runs:
                raise RequestError(
                    "session run quota reached; restart the service after archiving runs"
                )
            used = (
                sum(
                    p.stat().st_size
                    for p in self.out.rglob("*")
                    if p.is_file() and not p.is_symlink()
                )
                if self.out.exists()
                else 0
            )
            if used >= self.limits.disk_bytes:
                raise RequestError("output disk quota reached")
            run = Run(uuid.uuid4().hex, name, cfg)
            self._runs[run.id] = run
            self._active = run.id
            run.thread = threading.Thread(target=self._work, args=(run,), daemon=True)
            run.thread.start()
            return self.snapshot(run.id)

    def _publish(self, run: Run, event: dict[str, Any]) -> None:
        if event.get("type") == "population_rates":
            event["wall_seconds"] = time.monotonic() - run.started
            used = sum(
                p.stat().st_size for p in self.out.rglob("*") if p.is_file() and not p.is_symlink()
            )
            if used > self.limits.disk_bytes:
                raise RunQuotaExceeded("output disk quota exceeded")
        run.events.publish({"run_id": run.id, **event})

    def _recorder(
        self,
        run: Run,
        cfg: ExperimentConfig,
        data: LoadedDataset,
        pops: dict[str, NDArray[np.int32]],
    ) -> LiveRecorder:
        graph = data.graph
        if graph.n_neurons > self.limits.max_neurons or graph.n_edges > self.limits.max_edges:
            raise RunQuotaExceeded("graph quota exceeded")
        # Intervention rows are also output: conservatively bound large whole-graph lesions.
        if (
            any(iv.kind == "remove_synapses" for iv in cfg.interventions)
            and graph.n_edges > 100_000
        ):
            raise RunQuotaExceeded("large lesion artifact requires the offline CLI")
        recorder = LiveRecorder(
            cfg,
            pops,
            lambda e: self._publish(run, e),
            run.stop,
            run.started + self.limits.wall_seconds,
            max_events=self.limits.max_events,
            pace_ms=self.pace_ms,
        )
        dataset = {
            "kind": cfg.dataset.kind,
            "dataset_id": graph.dataset_id,
            "n_neurons": graph.n_neurons,
            "n_edges": graph.n_edges,
            "graph_fingerprint_sha256": graph.fingerprint(),
            "biological_data": data.info.get("biological_data", False),
            "files": [{"name": Path(n).name, "sha256": sha} for n, sha in data.files],
            "verified_at_load": True,
        }
        summary = {
            "dataset": dataset,
            "model": {
                "parameter_set": cfg.model.parameter_set,
                "parameters": cfg.model.parameters().to_dict(),
            },
            "evidence": {
                "dataset": "EMPIRICAL" if dataset["biological_data"] else "ENGINEERED",
                "model_output": "ASSUMED",
                "stimulus": "ENGINEERED",
            },
            "limitations": [*LIMITATIONS, NOTICE],
            "gpu_verification": "UNVERIFIED by this dashboard",
            "populations": [
                {"name": n, "n_neurons": len(pops[n])} for n in cfg.recorded_populations()
            ],
            "raster_sample": [{"idx": i, "population": n} for i, n in recorder.sample.items()],
            "stimuli": [s.to_dict() for s in cfg.stimuli],
            "interventions": [iv.to_dict() for iv in cfg.interventions],
            "preview_pace_ms_per_bin": self.pace_ms,
        }
        with self._lock:
            run.summary = summary
            run.state = "cancelling" if run.stop.is_set() else "running"
            self._verified[str(graph.dataset_id)] = dataset
        self._publish(run, {"type": "manifest", "summary": summary})
        return recorder

    def _work(self, run: Run) -> None:
        try:
            results = run_experiment(
                run.cfg,
                self.out,
                run.id,
                progress=None,
                recorder_factory=lambda cfg, data, pops: self._recorder(run, cfg, data, pops),
            )
            with self._lock:
                run.metrics = results[0].metrics
                run.state = "succeeded"
        except RunCancelled:
            with self._lock:
                run.state = "cancelled"
        except RunQuotaExceeded as exc:
            with self._lock:
                run.error, run.state = str(exc), "failed"
        except Exception as exc:
            LOG.exception("Dashboard run failed (%s)", run.id)
            with self._lock:
                run.error = f"{type(exc).__name__}: run failed; check the local service log"
                run.state = "failed"
        finally:
            with self._lock:
                run.finished = time.monotonic()
                self._publish(run, {"type": "status", "state": run.state, "error": run.error})
                self._active = None

    def get(self, rid: str) -> Run:
        with self._lock:
            try:
                return self._runs[rid]
            except KeyError as exc:
                raise KeyError("unknown run") from exc

    def snapshot(self, rid: str) -> dict[str, Any]:
        with self._lock:
            r = self.get(rid)
            return {
                "id": r.id,
                "config": r.config_name,
                "state": r.state,
                "seed": r.cfg.seed,
                "backend": r.cfg.backend,
                "steps": r.cfg.steps,
                "dt_ms": r.cfg.dt_ms,
                "duration_ms": r.cfg.duration_ms,
                "wall_seconds": (r.finished or time.monotonic()) - r.started,
                "manifest_summary": r.summary,
                "metrics": r.metrics,
                "error": r.error,
                "ring_dropped_messages": r.events.dropped,
                "notice": NOTICE,
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "active_run": self._active,
                "concurrency": 1,
                "limits": asdict(self.limits),
                "preview_pace_ms_per_bin": self.pace_ms,
                "gpu_verification": "UNVERIFIED by this dashboard",
                "notice": NOTICE,
            }

    def cancel(self, rid: str) -> dict[str, Any]:
        with self._lock:
            run = self.get(rid)
            if run.state not in TERMINAL:
                run.stop.set()
                run.state = "cancelling"
            return self.snapshot(rid)

    def artifacts(self, rid: str) -> list[dict[str, Any]]:
        run = self.get(rid)
        if run.state != "succeeded":
            return []
        root = self.out / run.id
        # Listing only: there is deliberately no endpoint that takes a file path or serves data.
        return [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(root.iterdir())
            if p.is_file() and not p.is_symlink()
        ]

    def claim_client(self) -> bool:
        with self._lock:
            if self._clients >= self.limits.max_clients:
                return False
            self._clients += 1
            return True

    def release_client(self) -> None:
        with self._lock:
            self._clients -= 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            runs = list(self._runs.values())
            for r in runs:
                r.stop.set()
        for r in runs:
            if r.thread:
                r.thread.join(timeout=5)
