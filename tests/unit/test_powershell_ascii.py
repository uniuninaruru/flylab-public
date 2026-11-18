"""Windows PowerShell 5.1 reads BOM-less scripts as the ANSI code page (CP932 on Japanese
Windows), so any non-ASCII byte in a .ps1 is silently corrupted on the host. Use \\uXXXX
escapes inside regexes instead of literal characters."""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "scripts").rglob("*.ps1"))


def test_there_are_powershell_scripts() -> None:
    assert SCRIPTS


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_ps1_is_ascii_only(path: Path) -> None:
    data = path.read_bytes()
    bad = [i for i, b in enumerate(data) if b > 0x7F]
    assert not bad, f"{path.name}: non-ASCII byte at offset {bad[0]}"
