"""PR-03 review: profiling and memory accounting (findings 7 and 8 of docs/audit/PR-03-review.md).

No GPU is used. Finding 7 replaces torch's profiler with a stub that returns one CPU-op row
and its kernel row, exactly the pair torch 2.15's `key_averages()` produces in kineto mode
(an op's `self_device_time_total` is the time of the kernels it launched, and each kernel is
also its own row). Finding 8 runs `flylab bench` on the tiny fixture with an engine and an
`nvidia-smi` that report CUDA-like numbers.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from flylab.experiments import benchmark
from flylab.neural.lif_numpy import NumpyLIFEngine

REPORT = "docs/audit/PR-03-review.md"
REPO = Path(__file__).resolve().parents[2]
GiB, MiB = 2**30, 2**20


@pytest.fixture
def tiny(monkeypatch: pytest.MonkeyPatch) -> Path:
    if not (REPO / "configs").is_dir():
        pytest.skip("configs/ is not part of this checkout (e.g. the CPU Docker image)")
    monkeypatch.chdir(REPO)  # the fixture paths in the example config are repo-relative
    return REPO / "configs" / "tiny_fixture.yaml"


# --- Finding 7 ---------------------------------------------------------------------------------


def _profile_step_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flylab_profile_step", REPO / "scripts" /
                                                  "profile_step.py")  # fmt: skip
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("with_copies", [False, True])
def test_finding_7_profile_counts_device_time_once(
    tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_copies: bool
) -> None:
    """Finding 7: op attribution is excluded; device memcpy rows identify direction."""
    torch = pytest.importorskip("torch")
    DeviceType = torch.autograd.DeviceType

    class Row:
        def __init__(self, key: str, device_type: Any, cpu_us: float, device_us: float) -> None:
            self.key, self.count, self.device_type = key, 2, device_type
            self.self_cpu_time_total, self.self_device_time_total = cpu_us, device_us
            self.is_legacy, self.is_user_annotation = False, False

    class FakeProfile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeProfile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def key_averages(self) -> list[Row]:
            # One elementwise op and the kernel it launched: 6.83 us of GPU time in total,
            # as for aten::bitwise_and in docs/acceptance/evidence/profile-B1-*.json.
            rows = [
                Row("aten::bitwise_and", DeviceType.CPU, 71.9, 6.83),
                Row("void at::native::vectorized_elementwise_kernel<...>", DeviceType.CUDA,
                    0.0, 6.83),
            ]  # fmt: skip
            if with_copies:
                rows += [
                    Row("cudaMemcpyAsync", DeviceType.CPU, 11.0, 3.0),
                    Row("Memcpy HtoD (Pageable -> Device)", DeviceType.CUDA, 0.0, 1.0),
                    Row("Memcpy DtoH (Device -> Pageable)", DeviceType.CUDA, 0.0, 2.0),
                ]
            return rows

    monkeypatch.setattr(torch.profiler, "profile", FakeProfile)
    mod = _profile_step_module()
    rc = mod.main(["--config", str(tiny), "--backend", "torch-cpu", "--warmup-steps", "1",
                   "--steps", "2", "--tag", "audit", "--out", str(tmp_path)])  # fmt: skip
    assert rc == 0
    [path] = tmp_path.glob("profile-audit-*.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_us = 9.83 if with_copies else 6.83
    assert record["device_total_us"] == pytest.approx(expected_us), record["device_total_us"]
    assert record["copy_calls_per_step"] == {
        "host_to_device": int(with_copies),
        "device_to_host": int(with_copies),
    }
    assert all(row["device_type"] == "DeviceType.CUDA" for row in record["top_by_device"])


# --- Finding 8 ---------------------------------------------------------------------------------


class _CudaLikeEngine(NumpyLIFEngine):
    """The NumPy engine, reporting torch-allocator peaks of a large CUDA run."""

    backend = "torch-cuda"

    def device_memory_stats(self) -> dict[str, int] | None:
        return {
            "allocated_bytes": int(9.3 * GiB),
            "reserved_bytes": int(9.4 * GiB),
            "max_allocated_bytes": int(9.4 * GiB),
            "max_reserved_bytes": int(9.5 * GiB),
        }


def _numbers(obj: object) -> Iterator[float]:
    """Every number in a nested record, skipping host-memory fields (RSS could collide)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        yield float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if "host" not in str(k) and "rss" not in str(k):
                yield from _numbers(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from _numbers(v)


def test_finding_8_bench_reports_the_device_wide_increase(
    tiny: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 8: snapshots and device-wide deltas expose non-allocator memory."""
    import flylab.neural.lif_torch as lif_torch

    calls: list[int] = []

    def fake_gpus() -> list[dict[str, str]]:
        used = "1000 MiB" if not calls else "11800 MiB"  # before / while loaded
        calls.append(1)
        return [{"name": "FAKE GPU", "driver_version": "0", "memory.total": "12227 MiB",
                 "memory.used": used, "compute_cap": "12.0"}]  # fmt: skip

    def fake_engine(cfg: Any, prep: Any, net: Any, pops: dict[str, NDArray[np.int32]],
                    chunk: int | None = None) -> NumpyLIFEngine:  # fmt: skip
        return _CudaLikeEngine(net, prep.disc, cfg.dtype, pops)

    monkeypatch.setattr(lif_torch, "check_backend", lambda _backend: None)
    # Querying free memory need not share nvidia-smi's pre-context baseline. Both sources
    # belong in the record: mem_get_info can initialize CUDA before its first sample.
    available = iter([(11000 * MiB, 12227 * MiB), (2400 * MiB, 12227 * MiB),
                      (2200 * MiB, 12227 * MiB)])  # fmt: skip
    monkeypatch.setattr(
        lif_torch,
        "require_torch",
        lambda _backend: SimpleNamespace(
            cuda=SimpleNamespace(
                current_device=lambda: 0, mem_get_info=lambda _device: next(available)
            )
        ),
    )
    monkeypatch.setattr(benchmark, "_process_globals", lambda _cfg: contextlib.nullcontext())
    monkeypatch.setattr(benchmark, "_engine", fake_engine)
    monkeypatch.setattr(benchmark, "nvidia_smi_gpus", fake_gpus)
    monkeypatch.setattr(
        benchmark, "nvidia_smi_process_memory_mib",
        lambda _pid=None: {"visible": True, "reason": "[N/A] under WSL2", "used_mib": None},
    )  # fmt: skip
    rec = benchmark.run_benchmark(tiny, backend="torch-cuda", warmup=0, repeats=1,
                                  progress=None)  # fmt: skip
    # Today the 9.5 GiB allocator peak passes the 10 GiB target
    # (gates.vram_peak_reserved_within_10GiB_target is True), while nvidia-smi saw the
    # device's memory grow by 10,800 MiB (10.5 GiB) with FlyLab loaded. Spec 7.3 asks for the
    # allocator peaks, the nvidia-smi figure "and any difference"; neither the increase nor
    # its non-allocator part (10,800 - 9,728 MiB) is in the record, in MiB or bytes.
    increase, non_allocator = 10800, 10800 - 9.5 * 1024
    wanted = [increase, increase * MiB, non_allocator, non_allocator * MiB]
    reported = list(_numbers(rec["memory"])) + list(_numbers(rec["gpu"]))
    assert any(abs(x - w) <= 0.01 * w for x in reported for w in wanted), rec["memory"]
    memory: Any = rec["memory"]
    gpu: Any = rec["gpu"]
    assert memory["gpu_device_wide_used_delta_after_init_bytes"] == 8600 * MiB
    assert memory["gpu_device_wide_used_delta_end_bytes"] == 8800 * MiB
    assert memory["gpu_device_wide_observed_used_delta_bytes"] == 10800 * MiB
    assert memory["gpu_device_wide_delta_minus_peak_reserved_bytes"] == non_allocator * MiB
    assert gpu["device_memory"]["end"]["free_bytes"] == 2200 * MiB
    assert gpu["device_memory"]["before_engine_init"]["total_bytes"] == 12227 * MiB
    gates: Any = rec["gates"]
    assert gates["vram_peak_reserved_within_10GiB_target"] is True
    assert gates["vram_observed_device_delta_within_10GiB_target"] is False
    assert rec["status"] == "FAIL"


def test_finding_8_unavailable_memory_query_has_a_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finding 8: unavailable device telemetry stays null with a reason."""
    import flylab.neural.lif_torch as lif_torch

    def unavailable(_backend: str) -> None:
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(lif_torch, "require_torch", unavailable)
    snapshot = benchmark._device_memory_snapshot("torch-cuda")
    assert snapshot["used_bytes"] is None
    assert snapshot["reason"] == "torch.cuda.mem_get_info unavailable (RuntimeError)"
