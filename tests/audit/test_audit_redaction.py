"""Audit finding: under-redaction in scripts/redact.py and scripts/windows/audit_host.ps1.

scripts/redact.py says "Over-redaction is accepted; under-redaction is a bug." Each case below
is ordinary log text in which an address or account name survives redaction. Addresses are
from the documentation ranges (RFC 5737) or the Tailscale CGNAT range; names are made up.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ISSUE = "https://github.com/uniuninaruru/flylab/issues/7"

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "redact.py"
PS1 = REPO / "scripts" / "windows" / "audit_host.ps1"

# (log line, fragment that must not survive)
LEAKS = {
    "ipv4_before_period": ("Connected to 203.0.113.9.", "203.0.113.9"),
    "ipv4_mapped_ipv6": ("LISTEN 0 4096 [::ffff:100.101.102.103]:22 [::]:*", "101.102.103"),
    "version_word_on_line": ("peer 203.0.113.9 speaks protocol version 3", "203.0.113.9"),
    "version_inside_word": ("unit conversion done for 203.0.113.9", "203.0.113.9"),
    "ipv4_dash_suffix": ("addr=203.0.113.9-eth0", "203.0.113.9"),
}


@pytest.fixture(scope="module")
def redact_mod() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flylab_redact_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("case", sorted(LEAKS))
def test_redact_py_pattern_gap(redact_mod: ModuleType, case: str) -> None:
    text, secret = LEAKS[case]
    assert secret not in redact_mod.redact(text, extra=[], lits=[])


def test_redact_py_short_account_names(
    redact_mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # literals() drops every literal shorter than 3 characters, so a 2-letter user or host
    # name is never redacted anywhere in the log.
    monkeypatch.setattr(redact_mod.getpass, "getuser", lambda: "yu")
    monkeypatch.setattr(redact_mod.socket, "gethostname", lambda: "pc")
    monkeypatch.setenv("USERNAME", "yu")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.delenv("FLYLAB_REDACT", raising=False)
    out = redact_mod.redact("login user=yu on host=pc")
    assert "user=yu" not in out
    assert "host=pc" not in out


def _run_ps1_redaction(lines: list[str], pwsh: str, tmp_path: Path) -> str:
    """Run only the redaction block of audit_host.ps1 (no system commands) on `lines`."""
    text = PS1.read_text(encoding="ascii")
    block = text[text.index("# ---- redaction ----") :]
    quoted = ",".join("'" + s.replace("'", "''") + "'" for s in lines)
    probe = tmp_path / "probe.ps1"
    probe.write_text(
        "$env:COMPUTERNAME = 'PC'; $env:USERNAME = 'yu'; $env:USERPROFILE = 'C:\\Users\\yu'\n"
        f"$lines = @({quoted})\n{block}",
        encoding="ascii",
    )
    env = {**os.environ, "POWERSHELL_TELEMETRY_OPTOUT": "1"}
    res = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(probe)],
        capture_output=True, text=True, timeout=120, check=True, env=env,
    )  # fmt: skip
    return res.stdout


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="needs PowerShell (pwsh)")
def test_audit_host_ps1_has_the_same_gaps(tmp_path: Path) -> None:
    pwsh = shutil.which("pwsh")
    assert pwsh is not None
    cases = [LEAKS[k] for k in sorted(LEAKS)] + [("login user=yu", "user=yu")]
    out = _run_ps1_redaction([t for t, _ in cases], pwsh, tmp_path)
    leaked = [secret for _, secret in cases if secret in out]
    assert leaked == []
