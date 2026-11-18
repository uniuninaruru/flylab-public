from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from flylab.provenance.hashing import (
    md5_base64_file,
    sha256_file,
    verify_checksums,
    write_checksums,
)


def test_sha256_known_vector(tmp_path: Path) -> None:
    p = tmp_path / "abc.txt"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_md5_base64_matches_gcs_format(tmp_path: Path) -> None:
    p = tmp_path / "abc.txt"
    p.write_bytes(b"abc")
    expected = base64.b64encode(bytes.fromhex("900150983cd24fb0d6963f7d28e17f72")).decode()
    assert md5_base64_file(p) == expected


def test_multi_chunk_file(tmp_path: Path) -> None:
    data = bytes(range(256)) * 9000  # about 2.3 MB, crosses the 1 MiB chunk boundary
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()


def test_checksum_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    a, b = tmp_path / "a.txt", tmp_path / "sub" / "b.txt"
    a.write_text("one", encoding="utf-8")
    b.write_text("two", encoding="utf-8")
    sums = tmp_path / "checksums.sha256"
    write_checksums([b, a], sums, root=tmp_path)
    lines = sums.read_text(encoding="utf-8").splitlines()
    assert [ln.split("  ")[1] for ln in lines] == ["a.txt", "sub/b.txt"]
    assert verify_checksums(sums, tmp_path) == []

    a.write_text("changed", encoding="utf-8")
    b.unlink()
    problems = {p.path: p for p in verify_checksums(sums, tmp_path)}
    assert problems["a.txt"].actual is not None
    assert problems["sub/b.txt"].actual is None
