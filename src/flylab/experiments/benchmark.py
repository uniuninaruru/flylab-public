"""`flylab bench`: warmed-up, repeated, synchronized timing of one config (BENCHMARK_PROTOCOL).

What is timed: `engine.simulate(steps)` from a fresh `reset(seed)`, i.e. per-step input
generation (the schedule's Poisson draws), the engine step and a counting recorder. Not timed:
dataset loading, network building, the copy to the device (reported separately in
`phases_s`) and artifact writing (a benchmark writes no run artifacts). The device is
synchronized before the clock is read at both ends.

Every number in the output was measured by this process; anything it cannot measure is null
with a reason. Nothing here judges biological plausibility.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from flylab import SCHEMA_VERSION
from flylab.data.inspect import peak_rss_bytes
from flylab.experiments.config import ExperimentConfig
from flylab.experiments.runner import _engine, _process_globals
from flylab.experiments.workload import Workload, load_workload
from flylab.neural.engine import LIFEngineBase, NeuralStep
from flylab.provenance.environment import capture_environment
from flylab.provenance.hardware import (
    capture_hardware,
    nvidia_smi_gpus,
    nvidia_smi_process_memory_mib,
)

BENCH_KIND = "flylab_benchmark"
#: Spec 7.3 VRAM gate: target <= 10 GiB peak device memory attributable to FlyLab.
VRAM_TARGET_BYTES = 10 * 2**30
TIMED_REGION = (
    "engine.simulate(steps) after reset(seed): schedule inputs + engine step + counting "
    "recorder; device synchronized before both clock reads (time.perf_counter)"
)


_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class BenchmarkError(RuntimeError):
    """The benchmark could not run (bad arguments, backend unavailable)."""


def check_tag(tag: str | None) -> None:
    if tag is not None and not _TAG.match(tag):
        raise BenchmarkError(f"tag {tag!r} must match {_TAG.pattern}")


@dataclass
class _Counter:
    """Minimal recorder: totals only, O(N) memory, no I/O."""

    out_degree: NDArray[np.int64]
    spikes: int = 0
    edge_events: int = 0
    ever: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        self.ever = np.zeros(self.out_degree.size, dtype=bool)

    def record(self, step: NeuralStep) -> None:
        self.spikes += int(step.spike_idx.size)
        sent = step.spike_idx[step.propagated]
        self.edge_events += int(self.out_degree[sent].sum())
        self.ever[step.spike_idx] = True


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """Median, quartiles (inclusive method), IQR, min, max, mean. Empty input gives nulls."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return {"n": 0, "median": None, "p25": None, "p75": None, "iqr": None,
                "min": None, "max": None, "mean": None}  # fmt: skip
    if len(xs) == 1:
        q1 = q3 = xs[0]
    else:
        q1, _, q3 = statistics.quantiles(xs, n=4, method="inclusive")
    return {
        "n": len(xs),
        "median": statistics.median(xs),
        "p25": q1,
        "p75": q3,
        "iqr": q3 - q1,
        "min": xs[0],
        "max": xs[-1],
        "mean": statistics.fmean(xs),
    }


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _one_run(
    engine: LIFEngineBase, wl: Workload, out_degree: NDArray[np.int64]
) -> dict[str, object]:
    cfg = wl.config
    engine.reset(cfg.seed)
    schedule = wl.schedule()
    counter = _Counter(out_degree)
    engine.reset_peak_device_memory()
    engine.synchronize()
    t0 = time.perf_counter()
    summary = engine.simulate(cfg.steps, counter, schedule)  # synchronizes before returning
    wall = time.perf_counter() - t0
    mem = engine.device_memory_stats()
    n = wl.network.n_neurons
    return {
        "wall_s": wall,
        "steps_per_s": cfg.steps / wall if wall > 0 else None,
        "simulated_s_per_wall_s": (cfg.duration_ms / 1000.0) / wall if wall > 0 else None,
        "n_spikes": counter.spikes,
        "spikes_per_wall_s": counter.spikes / wall if wall > 0 else None,
        "synaptic_events": counter.edge_events,
        "synaptic_events_per_wall_s": counter.edge_events / wall if wall > 0 else None,
        "mean_active_fraction": counter.spikes / (n * cfg.steps) if n else None,
        "fraction_spiked_at_least_once": float(counter.ever.mean()) if n else None,
        "state_finite": summary.state_finite,
        "gpu_max_memory_allocated_bytes": None if mem is None else mem["max_allocated_bytes"],
        "gpu_max_memory_reserved_bytes": None if mem is None else mem["max_reserved_bytes"],
    }


def _stats(runs: list[dict[str, object]]) -> dict[str, object]:
    keys = ("wall_s", "steps_per_s", "simulated_s_per_wall_s", "spikes_per_wall_s",
            "synaptic_events_per_wall_s")  # fmt: skip
    out: dict[str, object] = {}
    for k in keys:
        out[k] = summarize([float(v) for r in runs if isinstance(v := r[k], int | float)])
    return out


def _max(runs: list[dict[str, object]], key: str) -> int | None:
    vals = [int(v) for r in runs if isinstance(v := r.get(key), int)]
    return max(vals) if vals else None


class _DeviceMemory(TypedDict):
    device_index: int | None
    free_bytes: int | None
    total_bytes: int | None
    used_bytes: int | None
    reason: str | None


def _device_memory_snapshot(backend: str) -> _DeviceMemory:
    """Sample device-wide usage, including memory outside torch's allocator.

    The first CUDA query may create a context before it returns. The separate pre-init
    nvidia-smi sample can capture that overhead; neither source isolates other processes.
    Optional telemetry must not prevent a benchmark (including an OOM record) being saved.
    """
    result: _DeviceMemory = {
        "device_index": None, "free_bytes": None, "total_bytes": None,
        "used_bytes": None, "reason": "host backend: no device memory",
    }  # fmt: skip
    if backend != "torch-cuda":
        return result
    try:
        from flylab.neural.lif_torch import require_torch

        cuda = require_torch(backend).cuda
        device = int(cuda.current_device())
        free, total = (int(n) for n in cuda.mem_get_info(device))
    except Exception as exc:
        result["reason"] = f"torch.cuda.mem_get_info unavailable ({type(exc).__name__})"
    else:
        return {
            "device_index": device,
            "free_bytes": free,
            "total_bytes": total,
            "used_bytes": total - free,
            "reason": None,
        }
    return result


def _used_delta(before: _DeviceMemory, after: _DeviceMemory) -> int | None:
    first, last = before["used_bytes"], after["used_bytes"]
    if first is None or last is None or before["device_index"] != after["device_index"]:
        return None
    return last - first


def _smi_memory_deltas(
    before: list[dict[str, str]] | None, after: list[dict[str, str]] | None
) -> list[dict[str, int]]:
    """Device-wide nvidia-smi deltas; row indices are not CUDA logical device indices."""
    if before is None or after is None or len(before) != len(after):
        return []
    deltas = []
    for index, (first, last) in enumerate(zip(before, after, strict=True)):
        values = [re.fullmatch(r"(\d+)\s+MiB", row.get("memory.used", ""))
                  for row in (first, last)]  # fmt: skip
        if not all(values):
            continue
        used_before, used_after = (int(m[1]) * 2**20 for m in values if m is not None)
        deltas.append(
            {
                "nvidia_smi_row_index": index,
                "used_before_bytes": used_before,
                "used_while_loaded_bytes": used_after,
                "used_delta_bytes": used_after - used_before,
            }
        )
    return deltas


def run_benchmark(
    config: ExperimentConfig | str | Path,
    *,
    steps: int | None = None,
    backend: str | None = None,
    dtype: str | None = None,
    warmup: int = 3,
    repeats: int = 10,
    max_edges_per_chunk: int | None = None,
    engine_impl: str | None = None,
    tag: str | None = None,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Run the benchmark and return its JSON-serialisable record (see module docstring)."""
    if warmup < 0 or repeats < 1:
        raise BenchmarkError("need warmup >= 0 and repeats >= 1")
    check_tag(tag)
    say = progress or (lambda _msg: None)
    started = dt.datetime.now(dt.UTC)
    wl = load_workload(config, steps=steps, backend=backend, dtype=dtype, engine_impl=engine_impl)
    cfg, net = wl.config, wl.network
    say(f"[bench] {cfg.experiment_id}: {cfg.backend} {cfg.dtype}, {cfg.steps} steps x "
        f"{cfg.dt_ms} ms, {net.n_neurons:,} neurons, {net.n_edges:,} edges")  # fmt: skip
    out_degree = np.diff(net.row_ptr).astype(np.int64)
    gpu: dict[str, object] = {}
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": BENCH_KIND,
        "tag": tag,
        "started_utc": started.isoformat(timespec="seconds"),
        "status": "FAIL",
        "workload": {
            "config_path": wl.config_path,
            "config_sha256": wl.config_sha256,
            "experiment_id": cfg.experiment_id,
            "backend": cfg.backend,
            "dtype": cfg.dtype,
            "dt_ms": cfg.dt_ms,
            "steps": cfg.steps,
            "simulated_ms": cfg.duration_ms,
            "seed": cfg.seed,
            "deterministic": cfg.deterministic,
            "engine_impl_requested": cfg.engine_impl,
            "stimuli": [{**s.to_dict(), "evidence": "ENGINEERED"} for s in cfg.stimuli],
            "interventions": [iv.to_dict() for iv in cfg.interventions],
            "populations": {p.name: int(wl.pops[p.name].size) for p in cfg.populations},
        },
        "dataset": {
            **wl.data.info,
            "files": [{"path": p, "sha256": s} for p, s in wl.data.files],
            "graph_fingerprint_sha256": wl.data.graph.fingerprint(),
            "n_neurons": wl.data.graph.n_neurons,
            "n_unique_edges": wl.data.graph.n_edges,
            "total_synapses": wl.data.graph.total_synapses,
        },
        "network": {
            "n_neurons": net.n_neurons,
            "n_edges_after_sign_policy": net.n_edges,
            "max_out_degree": int(out_degree.max()) if out_degree.size else 0,
            "unknown_sign_policy": net.unknown_sign_policy.value,
            "network_fingerprint_sha256": net.fingerprint(),
        },
        "method": {
            "warmup_runs": warmup,
            "timed_repeats": repeats,
            "timed_region": TIMED_REGION,
            "not_timed": "dataset load, network build, copy to device (phases_s); no artifacts",
            "synaptic_events": "sum of out-degrees of propagated spikes (edges a spike is sent "
            "along); spikes in the last `delay` steps are counted though they arrive after the end",
        },
        "phases_s": dict(wl.phases_s),
        "environment": capture_environment(),
        "hardware": capture_hardware(),
        "gpu": gpu,
    }
    warm: list[dict[str, object]] = []
    timed: list[dict[str, object]] = []
    error: str | None = None
    oom = False
    engine: LIFEngineBase | None = None
    smi_before = nvidia_smi_gpus()
    gpu["nvidia_smi_before"] = smi_before
    before = _device_memory_snapshot(cfg.backend)
    after_init: _DeviceMemory = {
        **before, "free_bytes": None, "total_bytes": None, "used_bytes": None,
        "reason": "engine initialization did not complete",
    }  # fmt: skip
    try:
        with _process_globals(cfg):
            t0 = time.perf_counter()
            engine = _engine(cfg, wl.prep, net, wl.pops, max_edges_per_chunk)
            engine.reset(cfg.seed)  # allocates the state, so it is counted below
            engine.synchronize()
            wl.phases_s["engine_init_and_copy_to_device"] = time.perf_counter() - t0
            after_init = _device_memory_snapshot(cfg.backend)
            record["engine"] = engine.describe()
            record["memory_after_init"] = {
                "engine_retained_bytes": engine.retained_bytes(),
                "device": engine.device_memory_stats(),
            }
            for k in range(warmup + repeats):
                run = _one_run(engine, wl, out_degree)
                (warm if k < warmup else timed).append(run)
                say(f"[bench] {'warm-up' if k < warmup else 'repeat'} "
                    f"{k + 1 if k < warmup else k - warmup + 1}: {run['wall_s']:.3f} s, "
                    f"{run['n_spikes']:,} spikes, finite={run['state_finite']}")  # fmt: skip
    except Exception as exc:  # recorded in the output; anything but OOM is re-raised
        error = f"{type(exc).__name__}: {exc}"
        oom = _is_oom(exc)
        if not oom:
            raise
        if engine is not None:
            record["memory_at_error"] = engine.device_memory_stats()
    finally:
        record["phases_s"] = dict(wl.phases_s)
        end = _device_memory_snapshot(cfg.backend)
        smi_after = nvidia_smi_gpus()
        smi_deltas = _smi_memory_deltas(smi_before, smi_after)
        gpu.update(
            device_memory={"before_engine_init": before, "after_engine_init": after_init,
                           "end": end},
            nvidia_smi_while_loaded=smi_after,
            nvidia_smi_device_memory_deltas=smi_deltas,
            nvidia_smi_this_process=nvidia_smi_process_memory_mib(os.getpid()),
        )  # fmt: skip

    peak_reserved = _max(timed + warm, "gpu_max_memory_reserved_bytes")
    on_gpu = cfg.backend == "torch-cuda"
    init_delta, end_delta = _used_delta(before, after_init), _used_delta(before, end)
    observed_deltas = [v for v in (init_delta, end_delta) if v is not None]
    # nvidia-smi and CUDA logical device indices may differ (CUDA_VISIBLE_DEVICES).
    # Only use its delta in the gate when the device association is unambiguous.
    if len(smi_before or []) == 1 and len(smi_deltas) == 1:
        observed_deltas.append(smi_deltas[0]["used_delta_bytes"])
    observed_delta = max(observed_deltas) if on_gpu and observed_deltas else None
    gates: dict[str, bool | None] = {
        "no_oom": not oom,
        "all_runs_finite": bool(timed) and all(r["state_finite"] is True for r in warm + timed),
        "vram_peak_reserved_within_10GiB_target": (
            None if not on_gpu or peak_reserved is None else peak_reserved <= VRAM_TARGET_BYTES
        ),
    }
    if on_gpu:
        gates["vram_observed_device_delta_within_10GiB_target"] = (
            None if observed_delta is None else observed_delta <= VRAM_TARGET_BYTES
        )
    record |= {
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "error": error,
        "warmup": warm,
        "repeats": timed,
        "stats": _stats(timed),
        "memory": {
            "host_process_peak_rss_bytes": peak_rss_bytes(),
            "host_note": "lifetime maximum of this process (dataset load included)",
            "gpu_max_memory_allocated_bytes": _max(timed + warm, "gpu_max_memory_allocated_bytes"),
            "gpu_max_memory_reserved_bytes": peak_reserved,
            "gpu_device_wide_used_delta_after_init_bytes": init_delta,
            "gpu_device_wide_used_delta_end_bytes": end_delta,
            "gpu_device_wide_observed_used_delta_bytes": observed_delta,
            "gpu_device_wide_delta_minus_peak_reserved_bytes": (
                None
                if observed_delta is None or peak_reserved is None
                else observed_delta - peak_reserved
            ),
            "gpu_device_wide_note": "Device-wide changes include other processes; sampled "
            "deltas are estimates, not a process-specific peak. The first mem_get_info call "
            "may initialize CUDA before its baseline; pre-init nvidia-smi can capture that "
            "overhead. Delta minus allocator peak is a signed estimate, not exact overhead.",
            "gpu_note": None if on_gpu else "host backend: no device memory",
        },
        "gates": gates,
        "status": "PASS" if all(v is not False for v in gates.values()) and not error else "FAIL",
    }
    return record


def write_record(record: dict[str, object], out_dir: str | Path, prefix: str) -> Path:
    tag = record.get("tag")
    check_tag(tag if isinstance(tag, str) else None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{prefix}-{tag}-{stamp}.json" if tag else f"{prefix}-{stamp}.json"
    path = out / name
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return path
