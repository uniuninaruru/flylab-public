"""PR-03 review: host scripts (findings 4 and 9 of docs/audit/PR-03-review.md).

The scripts drive the Windows GPU host over SSH. Here a fake `ssh` on PATH stands in for the
host: for finding 4 it decodes the `powershell -EncodedCommand` payload, extracts the WSL bash
script and runs it locally against a temporary "host home"; for finding 9 it prints one line a
WSL shell would print. Nothing contacts a real host. The scripts are copied into a temporary
repository so that their `artifacts/` output never lands in this checkout.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPORT = "docs/audit/PR-03-review.md"
REPO = Path(__file__).resolve().parents[2]

# Decodes what scripts/remote/common.sh sends and runs the inner WSL script with HOME set to
# FAKE_HOST_HOME. Mirrors common.sh: pcps -> -EncodedCommand (UTF-16LE base64); wslbash ->
# `echo <base64> | base64 -d | bash` inside it.
_FAKE_HOST_SSH = """\
import base64, os, re, subprocess, sys
if "-G" in sys.argv:
    print("hostname resolved.internal.local\\nuser windows-owner")
    sys.exit(0)
ps = base64.b64decode(re.search(r"-EncodedCommand (\\S+)", sys.argv[-1]).group(1))
inner = re.search(r"echo (\\S+) \\| base64 -d \\| bash", ps.decode("utf-16-le")).group(1)
env = dict(os.environ, HOME=os.environ["FAKE_HOST_HOME"])
# BSD base64 on the Mac already wraps at 76 columns; the real WSL host uses GNU -w.
script = base64.b64decode(inner).decode().replace("base64 -w 76", "base64")
sys.exit(subprocess.run(["bash", "-c", script], env=env).returncode)
"""


def _need(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")


def _fake_repo(tmp_path: Path, *files: str) -> Path:
    root = tmp_path / "repo"
    for rel in files:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, dst)
    return root


def _fake_bin(tmp_path: Path, name: str, body: str) -> Path:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / name
    exe.write_text(body, encoding="utf-8")
    exe.chmod(0o755)
    return bindir


@pytest.mark.parametrize("pattern", ["--remove-files", ".*", "bench/.*", "bench/.hidden"])
def test_finding_4_fetch_refuses_option_like_patterns(tmp_path: Path, pattern: str) -> None:
    """Finding 4: reject option-like and dot-prefixed patterns before running tar."""
    _need("bash", "tar", "base64", "iconv")
    repo = _fake_repo(tmp_path, "scripts/remote/common.sh", "scripts/remote/fetch_artifacts.sh")
    host_home = tmp_path / "host"
    record = host_home / "flylab" / "artifacts" / "bench" / "bench-B1.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"status": "PASS"}\n', encoding="utf-8")
    bindir = _fake_bin(tmp_path, "ssh", f"#!{sys.executable}\n{_FAKE_HOST_SSH}")
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               FAKE_HOST_HOME=str(host_home))  # fmt: skip
    proc = subprocess.run(
        ["bash", str(repo / "scripts/remote/fetch_artifacts.sh"), "fakehost",
         pattern, "bench/*.json"],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    # The script documents itself as "Read-only on the host" and validates patterns; a pattern
    # that is really a tar option must be refused before anything runs on the host.
    assert record.exists(), "the host's file was deleted by `tar --remove-files`"
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "refused pattern" in proc.stderr


def test_finding_4_fetch_glob_cannot_turn_a_filename_into_a_tar_option(tmp_path: Path) -> None:
    """Finding 4: tar's -- protects even a wildcard-expanded option-like filename."""
    _need("bash", "tar", "base64", "iconv")
    repo = _fake_repo(
        tmp_path,
        "scripts/remote/common.sh",
        "scripts/remote/fetch_artifacts.sh",
        "scripts/redact.py",
    )
    host_home = tmp_path / "host"
    artifacts = host_home / "flylab" / "artifacts"
    artifacts.mkdir(parents=True)
    for filename in ("--remove-files", "result.json"):
        (artifacts / filename).write_text("retained\n", encoding="utf-8")
    bindir = _fake_bin(tmp_path, "ssh", f"#!{sys.executable}\n{_FAKE_HOST_SSH}")
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               FAKE_HOST_HOME=str(host_home))  # fmt: skip
    proc = subprocess.run(
        ["bash", str(repo / "scripts/remote/fetch_artifacts.sh"), "fakehost", "*"],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for filename in ("--remove-files", "result.json"):
        assert (artifacts / filename).read_text(encoding="utf-8") == "retained\n"
        assert (repo / "artifacts" / filename).read_text(encoding="utf-8") == "retained\n"


@pytest.mark.parametrize(
    ("script", "args"),
    [
        ("run_gpu_bench.sh", []),
        ("run_gpu_smoke.sh", []),
        ("compose_run.sh", ["gpu"]),
        ("windows_audit.sh", []),
    ],
)
def test_finding_9_remote_log_keeps_the_wsl_account_name(
    tmp_path: Path, script: str, args: list[str]
) -> None:
    """Finding 9: redact WSL and resolved SSH identities from every Mac-side log."""
    _need("bash", "base64", "iconv")
    repo = _fake_repo(
        tmp_path,
        "scripts/remote/common.sh",
        f"scripts/remote/{script}",
        "scripts/windows/audit_host.ps1",
        "scripts/redact.py",
    )
    # What the WSL shell prints when ~/flylab is missing (code not synced yet): it comes from
    # `cd "$HOME/flylab"`, before scripts/run_local_gpu_bench.sh and its redaction start.
    line = (
        "bash: line 1: cd: /home/alice-wsl/flylab: No such file or directory; "
        "distro=lab-distro host=resolved.internal.local user=windows-owner extra=custom-secret"
    )
    fake_ssh = (
        "#!/bin/sh\n"
        'for arg do if [ "$arg" = -G ]; then\n'
        "printf 'hostname resolved.internal.local\\nuser windows-owner\\n'; exit 0; fi; done\n"
        f"echo '{line}' >&2\nexit 1\n"
    )
    bindir = _fake_bin(tmp_path, "ssh", fake_ssh)
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
               FLYLAB_WSL_USER="alice-wsl", FLYLAB_WSL_DISTRO="lab-distro",
               FLYLAB_REDACT="custom-secret")  # fmt: skip
    proc = subprocess.run(
        ["bash", str(repo / f"scripts/remote/{script}"), "fakehost", *args],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )  # fmt: skip
    assert proc.returncode != 0
    text = proc.stdout + proc.stderr
    for log in (repo / "artifacts").rglob("*"):
        if log.is_file():
            text += log.read_text(encoding="utf-8")
    assert "No such file or directory" in text  # the line was captured ...
    for name in (
        "alice-wsl",
        "lab-distro",
        "resolved.internal.local",
        "windows-owner",
        "custom-secret",
    ):
        assert name not in text, text  # ... but identifiers must not be
