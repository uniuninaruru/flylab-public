"""Download registered upstream dataset files with integrity checks (`flylab data fetch`).

Only files listed in `flylab.data.sources` can be fetched, from the release's own URL prefix.
Each file is streamed to `<name>.part`, checked against the upstream size and MD5 (the value
Google Cloud Storage publishes), then renamed. SHA-256 is computed on the way and is the hash of
record. `SOURCE.json` next to the files records URL, access time, license and hashes.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from flylab import SCHEMA_VERSION, __version__
from flylab.data.sources import DatasetRelease

CHUNK = 8 << 20  # 8 MiB
TIMEOUT_S = 60
USER_AGENT = f"flylab/{__version__} (dataset fetch)"


class FetchError(RuntimeError):
    """A download did not match the upstream size or hash; nothing was kept."""


@dataclass(frozen=True)
class FetchResult:
    name: str
    url: str
    path: str
    size_bytes: int
    md5_base64: str
    sha256: str
    status: str  # "downloaded" | "already_present"


def release_dir(dest: str | Path, release: DatasetRelease) -> Path:
    """data/raw/male-cns-v1.0 for dataset id male-cns:v1.0."""
    return Path(dest) / release.dataset_id.replace(":", "-")


def hash_file_md5_sha256(path: Path) -> tuple[str, str, int]:
    """(base64 MD5 as GCS reports it, hex SHA-256, size) in one streaming pass."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    return base64.b64encode(md5.digest()).decode("ascii"), sha.hexdigest(), size


def _download(
    url: str, part: Path, expected_size: int, progress: Callable[[str], None] | None
) -> tuple[str, str, int]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    next_report = 0.0
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, part.open("wb") as out:  # noqa: S310
        length = resp.headers.get("Content-Length")
        if length is not None and int(length) != expected_size:
            raise FetchError(f"{url}: server reports {length} bytes, registry says {expected_size}")
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
            if progress and expected_size and size / expected_size >= next_report:
                progress(f"  {part.name[:-5]}: {size / 1e6:,.0f} / {expected_size / 1e6:,.0f} MB")
                next_report += 0.1
    return base64.b64encode(md5.digest()).decode("ascii"), sha.hexdigest(), size


def fetch_release(
    release: DatasetRelease,
    dest: str | Path,
    names: Sequence[str],
    *,
    base_url: str | None = None,
    progress: Callable[[str], None] | None = print,
) -> list[FetchResult]:
    """Fetch `names` (registered files of `release`) into `dest/<dataset>/`.

    Files already present with the right size and MD5 are not downloaded again.
    `base_url` replaces the release's URL prefix (tests use a local server).
    """
    out_dir = release_dir(dest, release)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (base_url or release.https_prefix).rstrip("/")
    results: list[FetchResult] = []

    for name in names:
        spec = release.file(name)  # KeyError for unregistered names
        url = f"{prefix}/{name}"
        target = out_dir / name
        if target.is_file() and target.stat().st_size == spec.size_bytes:
            md5_b64, sha, size = hash_file_md5_sha256(target)
            if md5_b64 == spec.md5_base64:
                results.append(
                    FetchResult(name, url, str(target), size, md5_b64, sha, "already_present")
                )
                if progress:
                    progress(f"{name}: already present, MD5 matches upstream")
                continue
        part = target.with_name(target.name + ".part")
        if progress:
            progress(f"{name}: downloading {spec.size_bytes / 1e6:,.0f} MB from {url}")
        try:
            md5_b64, sha, size = _download(url, part, spec.size_bytes, progress)
            if size != spec.size_bytes:
                raise FetchError(f"{name}: got {size} bytes, expected {spec.size_bytes}")
            if md5_b64 != spec.md5_base64:
                raise FetchError(f"{name}: MD5 {md5_b64} does not match upstream {spec.md5_base64}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(target)
        results.append(FetchResult(name, url, str(target), size, md5_b64, sha, "downloaded"))
        if progress:
            progress(f"{name}: OK (size and MD5 match upstream; sha256 {sha[:16]}...)")

    _write_source_record(out_dir, release, results)
    return results


def _write_source_record(
    out_dir: Path, release: DatasetRelease, results: Sequence[FetchResult]
) -> None:
    """Merge results into SOURCE.json (provenance for every raw file)."""
    record_path = out_dir / "SOURCE.json"
    record: dict[str, object] = {}
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
    files = record.get("files")
    files_by_name: dict[str, object] = dict(files) if isinstance(files, dict) else {}
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for r in results:
        entry = asdict(r)
        entry["path"] = Path(r.path).name  # never record absolute host paths
        entry["checked_utc"] = now
        files_by_name[r.name] = entry
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": release.dataset_id,
            "release_date": release.release_date,
            "license": release.license_spdx,
            "license_url": release.license_url,
            "attribution": release.attribution,
            "download_page": release.download_page,
            "fetched_with": f"flylab {__version__}",
            "files": dict(sorted(files_by_name.items())),
        }
    )
    tmp = record_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(record_path)
