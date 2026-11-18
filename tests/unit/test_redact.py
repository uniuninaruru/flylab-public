from __future__ import annotations

import getpass
import importlib.util
import socket
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "redact.py"


@pytest.fixture(scope="module")
def redact_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flylab_redact", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flylab_redact"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tailscale ip 100.64.0.10 ok", "tailscale ip <ipv4> ok"),
        ("listen 0.0.0.0:22 and 127.0.0.1:8765", "listen 0.0.0.0:22 and 127.0.0.1:8765"),
        ("peer fd7a:115c:a1e0::1234 up", "peer <ipv6> up"),
        ("v6 2001:db8:4300:410:bc8d:3a5e:a2d9:b57c", "v6 <ipv6>"),
        ("mac 00:1a:2b:3c:4d:5e", "mac <mac>"),
        ("node gpu-box.tail1234.ts.net", "node <tailnet-host>"),
        ("mail someone@example.com", "mail <email>"),
        ("at 13:54:47 GMT", "at 13:54:47 GMT"),  # clock times survive
        ("Ubuntu 24.04.1 LTS", "Ubuntu 24.04.1 LTS"),  # versions are not IPv4
        ("sha256:fe81560990fae86d", "sha256:fe81560990fae86d"),
        ("tcp4  192.168.1.5.22  *.*  LISTEN", "tcp4  <ipv4>.22  *.*  LISTEN"),  # macOS netstat
        ("tcp4  127.0.0.1.6463  *.*  LISTEN", "tcp4  127.0.0.1.6463  *.*  LISTEN"),
        ("WSL version: 2.4.13.0", "WSL version: 2.4.13.0"),
        (
            "Linux 5.15.167.4-microsoft-standard-WSL2 x86_64",
            "Linux 5.15.167.4-microsoft-standard-WSL2 x86_64",
        ),
        ("LISTEN 0 128 10.0.0.7:22 0.0.0.0:*", "LISTEN 0 128 <ipv4>:22 0.0.0.0:*"),
        ("a 100.64.0.10\nb 203.0.113.9\n", "a <ipv4>\nb <ipv4>\n"),
        ("WSL バージョン: 2.6.1.0", "WSL バージョン: 2.6.1.0"),  # Japanese Windows
        ("remote=100.64.0.0/10,fd7a:115c:a1e0::/48", "remote=100.64.0.0/10,fd7a:115c:a1e0::/48"),
        ("peer fd7a:115c:a1e0::9", "peer <ipv6>"),  # a host inside the range is still hidden
        ("remote=100.64.0.0/255.192.0.0", "remote=100.64.0.0/255.192.0.0"),  # netmask kept
    ],
)
def test_patterns(redact_mod: ModuleType, text: str, expected: str) -> None:
    # Patterns only: the machine's own host/user names (e.g. a Mac named "Mac") must not matter.
    assert redact_mod.redact(text, extra=[], lits=[]) == expected


def test_this_machine_is_removed(redact_mod: ModuleType) -> None:
    host = socket.gethostname()
    user = getpass.getuser()
    home = str(Path.home())
    out = redact_mod.redact(f"host={host} user={user} home={home}/x")
    for s in (host, home):
        if len(s) >= 3:
            assert s not in out
    if len(user) >= 3 and user.lower() not in redact_mod._GENERIC_NAMES:
        assert user not in out


def test_cli_streams_line_by_line() -> None:
    import subprocess

    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    proc.stdin.write("first 203.0.113.9\n")
    proc.stdin.flush()
    # The first line must arrive before stdin is closed.
    assert proc.stdout.readline() == "first <ipv4>\n"
    proc.stdin.write("second\n")
    proc.stdin.close()
    assert proc.stdout.read() == "second\n"
    assert proc.wait(timeout=10) == 0


def test_extra_literals(redact_mod: ModuleType) -> None:
    assert redact_mod.redact("ssh alice@winbox", extra=["winbox"]) == "ssh alice@<redacted>"
    assert redact_mod.redact("target WINBOX", extra=["winbox"]) == "target <redacted>"
