"""Streaming file hashes and sha256sum-compatible checksum files.

Files are read in chunks so multi-GB connectome tables never need to fit in memory.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def _digest(path: Path, algo: str) -> bytes:
    # Integrity checks only (MD5 matches what GCS publishes); not a security boundary.
    h = hashlib.new(algo, usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.digest()


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of a file (the hash FlyLab records for every input and output)."""
    return _digest(Path(path), "sha256").hex()


def md5_base64_file(path: str | Path) -> str:
    """Base64 MD5, the format Google Cloud Storage reports in `x-goog-hash: md5=...`.

    Used only to compare a download against the upstream object; SHA-256 stays the
    hash of record.
    """
    return base64.b64encode(_digest(Path(path), "md5")).decode("ascii")


def write_checksums(paths: Iterable[str | Path], out: str | Path, root: str | Path) -> None:
    """Write `<sha256>  <path relative to root>` lines sorted by path, like `sha256sum`."""
    root_p = Path(root).resolve()
    entries = []
    for p in paths:
        p_abs = Path(p).resolve()
        entries.append((p_abs.relative_to(root_p).as_posix(), sha256_file(p_abs)))
    lines = [f"{digest}  {rel}" for rel, digest in sorted(entries)]
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ChecksumMismatch:
    path: str
    expected: str
    actual: str | None  # None means the file is missing


def verify_checksums(checksum_file: str | Path, root: str | Path) -> list[ChecksumMismatch]:
    """Return every entry whose file is missing or whose hash differs. Empty list = all good."""
    root_p = Path(root)
    problems: list[ChecksumMismatch] = []
    for line in Path(checksum_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        target = root_p / rel.strip()
        if not target.is_file():
            problems.append(ChecksumMismatch(rel, expected, None))
            continue
        actual = sha256_file(target)
        if actual != expected:
            problems.append(ChecksumMismatch(rel, expected, actual))
    return problems
