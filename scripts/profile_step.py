"""Where does an engine step spend its time? torch.profiler over a config's first steps.

Profiling evidence for a later performance PR (spec 7.3: optimize only after profiling). Not a
benchmark: the profiler itself slows every operation down. Run on the RTX host, e.g.
  docker compose run --rm gpu python scripts/profile_step.py \\
      --config configs/bench/b1_male_cns_100ms.yaml --tag B1
Writes artifacts/gpu/profile-<tag>-<UTC>.json and prints the top operations.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from flylab import SCHEMA_VERSION
from flylab.experiments.runner import _engine, _process_globals
from flylab.experiments.schedule import ExperimentSchedule
from flylab.experiments.workload import load_workload
from flylab.neural.engine import StepDrive, StepInterventions
from flylab.provenance.environment import capture_environment

# Runtime calls do not identify the direction of a copy. Device memcpy events below do.
SYNC_OPS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync", "cudaMemcpy")


class _TimedSchedule:
    """Forwards `inputs` and adds up the time spent generating them."""

    def __init__(self, inner: ExperimentSchedule) -> None:
        self.inner = inner
        self.seconds = 0.0

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        t0 = time.perf_counter()
        out = self.inner.inputs(step)
        self.seconds += time.perf_counter() - t0
        return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--steps", type=int, default=300, help="profiled steps")
    p.add_argument("--backend", default=None)
    p.add_argument("--engine-impl", default=None, choices=["auto", "eager", "fused"])
    p.add_argument("--tag", default="profile")
    p.add_argument("--out", type=Path, default=Path("artifacts/gpu"))
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args(argv)

    import torch
    from torch.profiler import ProfilerActivity, profile

    # torch exposes this enum at runtime, but its stubs omit the re-export.
    cuda_device_type = torch.autograd.DeviceType.CUDA  # type: ignore[attr-defined, unused-ignore]
    wl = load_workload(args.config, backend=args.backend, engine_impl=args.engine_impl)
    cfg = wl.config
    if args.warmup_steps + args.steps > cfg.steps:
        print(f"config has only {cfg.steps} steps", file=sys.stderr)
        return 2
    with _process_globals(cfg):
        engine = _engine(cfg, wl.prep, wl.network, wl.pops)
        engine.reset(cfg.seed)
        schedule = _TimedSchedule(wl.schedule())
        # simulate(), not step(): it is the path runs and benchmarks take, and engines may batch
        # several steps on the device there (the fused CUDA engine does).
        engine.simulate(args.warmup_steps, None, schedule)
        engine.synchronize()
        acts = [ProfilerActivity.CPU]
        if cfg.backend == "torch-cuda":
            acts.append(ProfilerActivity.CUDA)
        schedule.seconds = 0.0
        t0 = time.perf_counter()
        with profile(activities=acts) as prof:
            engine.simulate(args.steps, None, schedule)
            engine.synchronize()
        wall = time.perf_counter() - t0
        t_sched = schedule.seconds

    rows = []
    for ev in prof.key_averages():
        # Kineto reports kernel time both on the CPU op that launched it and on the
        # device event itself. Only device rows contribute to the device total.
        is_device = ev.device_type == cuda_device_type
        rows.append({
            "name": ev.key,
            "device_type": str(ev.device_type),
            "calls": int(ev.count),
            "calls_per_step": ev.count / args.steps,
            "self_cpu_us": float(ev.self_cpu_time_total),
            "device_us": float(ev.self_device_time_total or 0.0) if is_device else 0.0,
        })  # fmt: skip
    rows.sort(key=lambda r: -r["self_cpu_us"])
    syncs = {r["name"]: r["calls_per_step"] for r in rows if r["name"] in SYNC_OPS}
    total_cpu = sum(r["self_cpu_us"] for r in rows)
    total_dev = sum(r["device_us"] for r in rows)
    device_rows = [r for r in rows if r["device_type"] == str(cuda_device_type)]
    copies = {
        direction: sum(r["calls_per_step"] for r in device_rows if marker in r["name"])
        for direction, marker in (
            ("host_to_device", "Memcpy HtoD"),
            ("device_to_host", "Memcpy DtoH"),
        )
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_step_profile",
        "tag": args.tag,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "note": "profiler overhead included; for attribution only, not timing; device time "
        "counts CUDA device rows only, excluding CPU-op attribution",
        "config_sha256": wl.config_sha256,
        "backend": cfg.backend,
        "dtype": cfg.dtype,
        "n_neurons": wl.network.n_neurons,
        "n_edges": wl.network.n_edges,
        "warmup_steps": args.warmup_steps,
        "profiled_steps": args.steps,
        "wall_s_with_profiler": wall,
        "schedule_inputs_s": t_sched,
        "self_cpu_total_us": total_cpu,
        "device_total_us": total_dev,
        "sync_and_copy_calls_per_step": syncs,
        "copy_calls_per_step": copies,
        "top_by_self_cpu": rows[: args.top],
        "top_by_device": sorted(device_rows, key=lambda r: -r["device_us"])[: args.top],
        "torch_version": str(torch.__version__),
        "environment": capture_environment(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"profile-{args.tag}-{stamp}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    per_step_ms = 1000 * wall / args.steps
    print(f"{cfg.backend}: {per_step_ms:.3f} ms/step with profiler "
          f"(schedule {1000 * t_sched / args.steps:.3f} ms/step)")  # fmt: skip
    print(f"self CPU total {total_cpu / args.steps:.0f} us/step, "
          f"device total {total_dev / args.steps:.0f} us/step")  # fmt: skip
    print("sync/copy calls per step:", syncs)
    print("copy directions per step:", copies)
    print(f"{'op':48s} {'calls/step':>10s} {'cpu us/step':>12s} {'dev us/step':>12s}")
    for r in rows[: args.top]:
        cpu, dev = r["self_cpu_us"] / args.steps, r["device_us"] / args.steps
        print(f"{r['name'][:48]:48s} {r['calls_per_step']:10.1f} {cpu:12.1f} {dev:12.1f}")
    print(f"profile={path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
