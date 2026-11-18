"""Hardware facts for benchmark records (spec 11.1), without identifying the host.

Recorded: CPU model string, logical CPU count, total RAM, the container memory limit, and
`nvidia-smi` device facts. Not recorded: serial numbers, UUIDs, PCI bus ids, host names.
Every value is None when it cannot be read; nothing is guessed.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

_GPU_FIELDS = ("name", "driver_version", "memory.total", "memory.used", "compute_cap")


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _sysctl(name: str) -> str | None:
    exe = shutil.which("sysctl")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "-n", name], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def cpu_model() -> str | None:
    info = _read("/proc/cpuinfo")
    if info:
        for line in info.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return _sysctl("machdep.cpu.brand_string")


def total_ram_bytes() -> int | None:
    info = _read("/proc/meminfo")
    if info:
        for line in info.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    value = _sysctl("hw.memsize")
    return int(value) if value and value.isdigit() else None


def cgroup_memory_limit_bytes() -> int | None:
    """Memory limit of this container (cgroup v2 `memory.max`), None if unlimited/unknown."""
    text = _read("/sys/fs/cgroup/memory.max")
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.isdigit() else None


def capture_hardware() -> dict[str, object]:
    return {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "machine": platform.machine(),
        "total_ram_bytes": total_ram_bytes(),
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes(),
        "note": "inside a WSL2 container total_ram_bytes is the WSL VM's memory, not the PC's",
    }


def nvidia_smi_gpus() -> list[dict[str, str]] | None:
    """One dict per GPU from `nvidia-smi --query-gpu`, or None if nvidia-smi is unavailable.
    `memory.used` is device-wide: it includes every other process on the GPU."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={','.join(_GPU_FIELDS)}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout  # fmt: skip
    except (subprocess.SubprocessError, OSError):
        return None
    rows = [ln for ln in out.splitlines() if ln.strip()]
    return [dict(zip(_GPU_FIELDS, (v.strip() for v in ln.split(",")), strict=False)) for ln in rows]


def nvidia_smi_process_memory_mib(pid: int | None = None) -> dict[str, object]:
    """GPU memory nvidia-smi attributes to this process, if it can see the process at all.

    Inside a container (and under WSL2) nvidia-smi usually lists host PIDs or no processes, so
    the result is often `visible: False`; that is reported, not papered over."""
    exe = shutil.which("nvidia-smi")
    pid = os.getpid() if pid is None else pid
    if exe is None:
        return {"visible": False, "reason": "nvidia-smi not found", "used_mib": None}
    try:
        out = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout  # fmt: skip
    except (subprocess.SubprocessError, OSError) as exc:
        return {"visible": False, "reason": f"nvidia-smi failed: {exc}", "used_mib": None}
    listed = 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        listed += 1
        if int(parts[0]) == pid:
            if parts[1].isdigit():
                return {"visible": True, "reason": None, "used_mib": int(parts[1])}
            return {
                "visible": True,
                "reason": f"nvidia-smi lists this PID but reports used_memory {parts[1]!r}: "
                "per-process memory is not available here (e.g. under WSL2)",
                "used_mib": None,
            }
    return {
        "visible": False,
        "reason": f"this PID is not among the {listed} compute process(es) nvidia-smi lists "
        "(container PID namespace or WSL2)",
        "used_mib": None,
    }
