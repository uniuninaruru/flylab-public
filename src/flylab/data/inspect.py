"""Schema and value audit of raw Feather (Arrow IPC) files (`flylab data inspect`).

Reads one record batch at a time from a memory map, so multi-GB files never sit in RAM as a
whole. Reports only what it measures: schema, row counts, null counts, numeric min/max,
value counts for low-cardinality text columns and a few sample rows.
"""

from __future__ import annotations

import json
import resource
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from flylab import SCHEMA_VERSION

MAX_TRACKED_VALUES = 64  # text columns with more distinct values are reported as high-cardinality
SAMPLE_ROWS = 5


def peak_rss_bytes() -> int:
    """Peak resident memory of this process (ru_maxrss is KiB on Linux, bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class _ColumnStats:
    def __init__(self, field: pa.Field) -> None:
        self.field = field
        self.nulls = 0
        self.min: Any = None
        self.max: Any = None
        self.values: Counter[Any] | None = None
        t = field.type
        if pa.types.is_dictionary(t):
            t = t.value_type
        self.numeric = pa.types.is_integer(t) or pa.types.is_floating(t)
        self.textual = (
            pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_boolean(t)
        )
        if self.textual:
            self.values = Counter()

    def update(self, arr: pa.ChunkedArray | pa.Array) -> None:
        if pa.types.is_dictionary(arr.type):
            arr = arr.cast(arr.type.value_type)
        self.nulls += arr.null_count
        if self.numeric and len(arr) > arr.null_count:
            mm = pc.min_max(arr)
            lo, hi = mm["min"].as_py(), mm["max"].as_py()
            self.min = lo if self.min is None else min(self.min, lo)
            self.max = hi if self.max is None else max(self.max, hi)
        if self.values is not None:
            for item in pc.value_counts(arr).to_pylist():
                self.values[_json_scalar(item["values"])] += item["counts"]
            if len(self.values) > MAX_TRACKED_VALUES:
                self.values = None  # high cardinality: stop tracking

    def report(self) -> dict[str, object]:
        out: dict[str, object] = {"type": str(self.field.type), "nulls": self.nulls}
        if self.numeric:
            out["min"], out["max"] = _json_scalar(self.min), _json_scalar(self.max)
        if self.textual:
            if self.values is None:
                out["value_counts"] = f"high cardinality (> {MAX_TRACKED_VALUES} distinct)"
            else:
                out["value_counts"] = dict(self.values.most_common())
        return out


def inspect_feather(path: str | Path) -> dict[str, object]:
    """Audit one Feather v2 / Arrow IPC file batch by batch."""
    p = Path(path)
    with pa.memory_map(str(p), "r") as source:
        reader = pa.ipc.open_file(source)
        schema = reader.schema
        stats = [_ColumnStats(f) for f in schema]
        rows = 0
        sample: list[dict[str, object]] = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            rows += batch.num_rows
            for col, st in zip(batch.columns, stats, strict=True):
                st.update(col)
            if len(sample) < SAMPLE_ROWS:
                head = batch.slice(0, SAMPLE_ROWS - len(sample)).to_pylist()
                sample.extend({k: _json_scalar(v) for k, v in r.items()} for r in head)
        metadata = schema.metadata or {}
        return {
            "file": p.name,
            "size_bytes": p.stat().st_size,
            "num_record_batches": reader.num_record_batches,
            "num_rows": rows,
            "schema_metadata_keys": sorted(k.decode("utf-8", "replace") for k in metadata),
            "columns": {f.name: st.report() for f, st in zip(schema, stats, strict=True)},
            "sample_rows": sample,
        }


def inspect_dir(raw_dir: str | Path) -> dict[str, object]:
    """Audit every *.feather in `raw_dir`, plus its SOURCE.json if present."""
    d = Path(raw_dir)
    files = sorted(d.glob("*.feather"))
    if not files:
        raise FileNotFoundError(f"no .feather files in {d}")
    source_path = d / "SOURCE.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_raw_data_audit",
        "raw_dir": d.name,
        "source": source,
        "files": [inspect_feather(f) for f in files],
        "peak_rss_bytes": peak_rss_bytes(),
    }
