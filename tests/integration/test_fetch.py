"""`flylab data fetch` against a local HTTP server serving synthetic files."""

from __future__ import annotations

import base64
import dataclasses
import functools
import hashlib
import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from flylab.data.fetch import FetchError, fetch_release, release_dir
from flylab.data.sources import DatasetRelease, UpstreamFile
from flylab.provenance.labels import Evidence

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB, synthetic


def _md5_b64(data: bytes) -> str:
    return base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()


def _release(files: tuple[UpstreamFile, ...]) -> DatasetRelease:
    return DatasetRelease(
        dataset_id="synthetic:fetch-test",
        release_date="2026-01-01",
        license_spdx="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        attribution="synthetic test data",
        download_page="https://example.invalid/",
        release_notes="https://example.invalid/",
        gs_prefix="gs://example-invalid",
        https_prefix="https://example.invalid/should-not-be-used",
        files=files,
        verified_on="2026-01-01",
        verified_by="test",
        evidence=Evidence.ENGINEERED,
    )


GOOD = UpstreamFile(
    name="good.feather",
    role="synthetic",
    size_bytes=len(PAYLOAD),
    md5_base64=_md5_b64(PAYLOAD),
    crc32c_base64="AAAAAA==",
    last_modified="n/a",
)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


class _QuietServer(http.server.ThreadingHTTPServer):
    def handle_error(self, request: object, client_address: object) -> None:
        pass  # the client hangs up early on purpose in the mismatch tests


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    root = tmp_path / "srv"
    root.mkdir()
    (root / "good.feather").write_bytes(PAYLOAD)
    (root / "corrupt.feather").write_bytes(PAYLOAD[:-1] + b"\x00")
    (root / "short.feather").write_bytes(PAYLOAD[:-10])
    handler = functools.partial(_Quiet, directory=str(root))
    httpd = _QuietServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_fetch_verifies_and_records_provenance(server: str, tmp_path: Path) -> None:
    rel = _release((GOOD,))
    dest = tmp_path / "raw"
    [res] = fetch_release(rel, dest, ["good.feather"], base_url=server, progress=None)
    assert res.status == "downloaded"
    assert res.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    out = release_dir(dest, rel)
    assert (out / "good.feather").read_bytes() == PAYLOAD
    assert not list(out.glob("*.part"))

    record = json.loads((out / "SOURCE.json").read_text(encoding="utf-8"))
    assert record["dataset_id"] == "synthetic:fetch-test"
    assert record["license"] == "CC-BY-4.0"
    entry = record["files"]["good.feather"]
    assert entry["path"] == "good.feather"  # no absolute host path
    assert entry["sha256"] == res.sha256
    assert entry["md5_base64"] == GOOD.md5_base64

    [again] = fetch_release(rel, dest, ["good.feather"], base_url=server, progress=None)
    assert again.status == "already_present"
    assert again.sha256 == res.sha256


def test_md5_mismatch_keeps_nothing(server: str, tmp_path: Path) -> None:
    bad = dataclasses.replace(GOOD, name="corrupt.feather")  # same size, wrong bytes
    with pytest.raises(FetchError, match="MD5"):
        fetch_release(
            _release((bad,)), tmp_path, ["corrupt.feather"], base_url=server, progress=None
        )
    out = release_dir(tmp_path, _release((bad,)))
    assert not (out / "corrupt.feather").exists()
    assert not list(out.glob("*.part"))


def test_size_mismatch_is_refused(server: str, tmp_path: Path) -> None:
    short = dataclasses.replace(GOOD, name="short.feather")
    with pytest.raises(FetchError, match="bytes"):
        fetch_release(
            _release((short,)), tmp_path, ["short.feather"], base_url=server, progress=None
        )


def test_corrupted_local_copy_is_replaced(server: str, tmp_path: Path) -> None:
    rel = _release((GOOD,))
    out = release_dir(tmp_path, rel)
    out.mkdir(parents=True)
    (out / "good.feather").write_bytes(b"\x00" * len(PAYLOAD))  # right size, wrong content
    [res] = fetch_release(rel, tmp_path, ["good.feather"], base_url=server, progress=None)
    assert res.status == "downloaded"
    assert (out / "good.feather").read_bytes() == PAYLOAD


def test_unregistered_name_refused(server: str, tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        fetch_release(
            _release((GOOD,)), tmp_path, ["../../etc/passwd"], base_url=server, progress=None
        )
