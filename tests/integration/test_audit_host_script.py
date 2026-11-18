"""The host audit script must run anywhere (it only reads) and must not leak identity."""

from __future__ import annotations

import getpass
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def test_audit_runs_read_only_and_redacts(repo_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "audit.txt"
    res = subprocess.run(
        ["bash", str(repo_root / "scripts" / "audit_host.sh"), "--out", str(out)],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# FlyLab host audit (read-only)")
    for heading in ("## Kernel and OS", "## NVIDIA GPU", "## Containers", "## Remote access"):
        assert heading in text
    host = socket.gethostname().split(".")[0]
    user = getpass.getuser()
    for secret in (host, str(Path.home())):
        if len(secret) >= 3:
            assert secret.lower() not in text.lower()
    if user.lower() not in {"root", "ubuntu", "runner"} and len(user) >= 3:
        assert user.lower() not in text.lower()
