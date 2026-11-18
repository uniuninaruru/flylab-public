"""Redact host-identifying text from audit/smoke logs before they are saved or shared.

Reads stdin, writes stdout. Removes: this machine's hostname and user name, the home
directory path, extra literals from FLYLAB_REDACT (comma-separated), IPv4 addresses (except
0.0.0.0 and 127.x, which show *whether* a port is public), IPv6 addresses, MAC addresses,
e-mail addresses and *.ts.net (Tailscale MagicDNS) names.

IPv4 exceptions, so diagnostics survive: a number written right after the word "version" (or
Japanese "バージョン") is kept (e.g. "WSL version: 2.4.13.0"), and four numbers followed by
"-microsoft" are a WSL kernel version (e.g. "5.15.167.4-microsoft-standard-WSL2"). Names
shorter than 3 characters are redacted as whole words only.

Over-redaction is accepted; under-redaction is a bug. Review output before committing it.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import re
import socket
import sys
from pathlib import Path

_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_IPV6 = re.compile(
    r"(?<![0-9A-Za-z:])(?:"
    r"(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}"  # 5+ groups: not a clock time
    r"|(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)?"  # with ::
    r"|::(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*)"
    r")(?![0-9A-Za-z])"
)
# Not followed by a digit or ".digit" (a sentence-final period is fine). "-microsoft" right
# after four numbers is a WSL kernel version, not an address.
_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?!\d)(?!\.\d)(?!-microsoft)")
# macOS netstat writes address.port with a dot: 192.168.1.5.22
_IPV4_DOTTED_PORT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})\.(\d{1,5})(?![\d.])")
# IPv4-mapped IPv6 as printed by ss: [::ffff:203.0.113.9]. Handled before the IPv6 pass.
_IPV4_MAPPED = re.compile(r"(::ffff:)(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])", re.IGNORECASE)
# A number right after the word "version" (or Japanese "バージョン") is a version, not an address.
_VERSION_NUMBER = re.compile(r"(\bversion|バージョン)(\s*[:=]?\s*v?)(\d+(?:\.\d+)+)", re.IGNORECASE)
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")
# Tailscale's published ranges (100.64.0.0/10, fd7a:115c:a1e0::/48) are not host addresses.
_KEEP_IPV4 = {"0.0.0.0", "100.64.0.0"}  # noqa: S104 - matching, not binding
_KEEP_IPV6 = {"fd7a:115c:a1e0::"}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TSNET = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.ts\.net\b", re.IGNORECASE)


def _is_public_marker(addr: str) -> bool:
    """0.0.0.0, 127.x, netmasks (255.x) and published ranges identify nobody."""
    return addr in _KEEP_IPV4 or addr.startswith(("127.", "255."))


def _ipv4_line(line: str) -> str:
    kept: list[str] = []

    def _protect(m: re.Match[str]) -> str:
        kept.append(m.group(3))
        return f"{m.group(1)}{m.group(2)}\x00{len(kept) - 1}\x00"

    line = _VERSION_NUMBER.sub(_protect, line)
    line = _IPV4_DOTTED_PORT.sub(
        lambda m: m.group(0) if _is_public_marker(m.group(1)) else f"<ipv4>.{m.group(2)}", line
    )
    line = _IPV4.sub(lambda m: m.group(0) if _is_public_marker(m.group(1)) else "<ipv4>", line)
    return _PLACEHOLDER.sub(lambda m: kept[int(m.group(1))], line)


# Account names that identify nobody; redacting them would erase words like "Ubuntu 24.04".
_GENERIC_NAMES = {"root", "ubuntu", "runner", "user", "admin", "administrator", "docker"}


def literals() -> list[str]:
    """Machine-specific strings to remove, longest first so paths go before names."""
    found: set[str] = set()
    try:
        host = socket.gethostname()
        found.update({host, host.split(".")[0]})
    except OSError:
        pass
    for getter in (getpass.getuser, lambda: os.environ.get("USERNAME", "")):
        with contextlib.suppress(OSError, KeyError):
            found.add(getter())
    found.add(str(Path.home()))
    found.update(s.strip() for s in os.environ.get("FLYLAB_REDACT", "").split(","))
    keep = (s for s in found if s and s.lower() not in _GENERIC_NAMES)
    return sorted(keep, key=len, reverse=True)


def _literal_pattern(lit: str) -> str:
    """Short literals (e.g. user "yu", host "pc") match whole words only, so ordinary words
    are not shredded; longer ones match anywhere."""
    if len(lit) < 3:
        return rf"(?<![A-Za-z0-9]){re.escape(lit)}(?![A-Za-z0-9])"
    return re.escape(lit)


def redact(text: str, extra: list[str] | None = None, lits: list[str] | None = None) -> str:
    for lit in [*(extra or []), *(literals() if lits is None else lits)]:
        text = re.sub(_literal_pattern(lit), "<redacted>", text, flags=re.IGNORECASE)
    text = _EMAIL.sub("<email>", text)
    text = _TSNET.sub("<tailnet-host>", text)
    text = _MAC.sub("<mac>", text)
    text = _IPV4_MAPPED.sub(
        lambda m: m.group(0) if _is_public_marker(m.group(2)) else f"{m.group(1)}<ipv4>", text
    )
    text = _IPV6.sub(lambda m: m.group(0) if m.group(0) in _KEEP_IPV6 else "<ipv6>", text)
    return "".join(_ipv4_line(ln) for ln in text.splitlines(keepends=True))


def main() -> int:
    # Line by line, flushed, so long remote jobs show progress as they run.
    # Every pattern above matches within one line, so nothing is lost by streaming.
    lits = literals()
    for line in sys.stdin:
        sys.stdout.write(redact(line, lits=lits))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
