"""GPU smoke test for the RTX 5070 host (spec section 2.4, plus sm_120, sparse and LIF checks).

Run inside the FlyLab container on the GPU host:  docker compose run --rm gpu-smoke
Writes artifacts/gpu/smoke-<UTC>.json and prints `key=value` lines.
Exit code 0 only if every check passes. Only measured values are written; others are null.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from flylab.data.synthetic import random_graph
from flylab.provenance.environment import capture_environment

EXPECTED_CAPABILITY = (12, 0)  # RTX 5070, Blackwell


def _nvidia_smi() -> dict[str, str] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    fields = "name,driver_version,memory.total,compute_cap"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.strip().splitlines()[0]  # fmt: skip
    except (subprocess.SubprocessError, OSError, IndexError):
        return None
    return dict(zip(fields.split(","), (v.strip() for v in out.split(",")), strict=False))


def _sparse_propagation_check(torch: Any) -> dict[str, object]:
    """CPU (NumPy) vs CUDA scatter_add_ on a synthetic graph.

    Every addend is an integer (synapse count x sign x 0/1 spike) and every per-neuron sum is
    far below 2**24, so float32 addition is exact in any order. A difference of exactly 0 is
    therefore the justified tolerance, even though GPU atomics reorder the additions.
    """
    graph, report = random_graph(n_neurons=20_000, mean_out_degree=20, seed=0)
    edges = graph.signed_edges()
    spikes = (np.random.default_rng(1).random(graph.n_neurons) < 0.05).astype(np.float32)

    cpu = np.zeros(graph.n_neurons, dtype=np.float64)
    np.add.at(cpu, edges.dst, edges.signed_count.astype(np.float64) * spikes[edges.src])

    dev = torch.device("cuda")
    src = torch.from_numpy(edges.src.astype(np.int64)).to(dev)
    dst = torch.from_numpy(edges.dst.astype(np.int64)).to(dev)
    w = torch.from_numpy(edges.signed_count).to(dev)
    s = torch.from_numpy(spikes).to(dev)
    out = torch.zeros(graph.n_neurons, dtype=torch.float32, device=dev)
    out.scatter_add_(0, dst, w * s[src])
    torch.cuda.synchronize()
    gpu = out.cpu().numpy().astype(np.float64)
    max_abs = float(np.max(np.abs(gpu - cpu)))
    return {
        "graph": report.dataset_id,
        "n_edges": graph.n_edges,
        "max_abs_diff": max_abs,
        "max_abs_input": float(np.max(np.abs(cpu))),
        "pass": max_abs == 0.0,
    }


def _lif_engine_check() -> dict[str, object]:
    """CUDA LIF engine vs the NumPy reference, lockstep, on a recurrent synthetic graph.

    Pass = identical spikes, or only flips justified by a reference potential within the margin
    of threshold (docs/MODEL_ASSUMPTIONS.md section 2.3), with states within that margin."""
    from flylab.data.graph import UnknownSignPolicy
    from flylab.neural.engine import StepDrive
    from flylab.neural.lif_numpy import NumpyLIFEngine
    from flylab.neural.lif_torch import TorchLIFEngine
    from flylab.neural.network import build_network
    from flylab.neural.params import SHIU2024_CODE_V1, discretize
    from flylab.neural.parity import lockstep

    graph, _ = random_graph(n_neurons=2000, mean_out_degree=30, seed=1)
    net = build_network(graph, UnknownSignPolicy.ERROR, 0.6)
    disc = discretize(SHIU2024_CODE_V1, 0.1)
    rng = np.random.default_rng(5)
    base = rng.uniform(4, 14, net.n_neurons)
    idx = np.arange(net.n_neurons, dtype=np.int32)

    class Noisy:
        def inputs(self, step: int) -> tuple[StepDrive, None]:
            return StepDrive(idx, base + rng.normal(0, 2, net.n_neurons)), None

    ref = NumpyLIFEngine(net, disc)
    gpu = TorchLIFEngine(net, disc, device="cuda")
    ref.reset(1)
    gpu.reset(1)
    rep = lockstep(ref, gpu, 500, Noisy(), state_every=50)
    return {**rep.to_dict(), "pass": rep.ok and rep.n_spikes_reference > 0}


def run_checks() -> dict[str, object]:
    checks: dict[str, object] = {}
    result: dict[str, object] = {
        "schema_version": 1,
        "kind": "flylab_gpu_smoke",
        "started_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "environment": capture_environment(),
        "nvidia_smi": _nvidia_smi(),
        "checks": checks,
        "status": "FAIL",
    }
    try:
        import torch
    except ImportError as exc:
        checks["torch_import"] = f"FAIL: {exc}"
        return result

    result["torch_version"] = torch.__version__
    result["torch_cuda_build"] = torch.version.cuda
    checks["cuda_available"] = bool(torch.cuda.is_available())
    if not checks["cuda_available"]:
        return result

    cap = tuple(torch.cuda.get_device_capability(0))
    arch_list = list(torch.cuda.get_arch_list())
    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["capability"] = list(cap)
    result["arch_list"] = arch_list
    checks["capability_is_12_0"] = cap == EXPECTED_CAPABILITY
    checks["arch_list_has_sm_120"] = "sm_120" in arch_list

    torch.cuda.reset_peak_memory_stats()
    x = torch.randn((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    checks["matmul_finite"] = bool(torch.isfinite(y).all().item())

    sparse = _sparse_propagation_check(torch)
    result["sparse_propagation"] = sparse
    checks["sparse_matches_cpu"] = sparse["pass"]

    lif = _lif_engine_check()
    result["lif_engine_lockstep"] = lif
    checks["lif_cuda_matches_numpy"] = lif["pass"]

    result["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
    ok = all(v is True for v in checks.values())
    result["status"] = "PASS" if ok else "FAIL"
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--out", type=Path, default=Path("artifacts/gpu"))
    args = p.parse_args(argv)

    result = run_checks()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = args.out / f"smoke-{stamp}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for key in ("torch_version", "torch_cuda_build", "gpu_name", "capability"):
        print(f"{key}={result.get(key)}")
    checks = result["checks"]
    assert isinstance(checks, dict)
    for name, value in checks.items():
        print(f"check.{name}={value}")
    print(f"smoke_test={result['status']}")
    print(f"log={path}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
