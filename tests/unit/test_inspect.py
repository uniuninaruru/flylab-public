from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flylab.cli.main import main
from flylab.data.inspect import MAX_TRACKED_VALUES, inspect_dir, inspect_feather


def _write(path: Path) -> None:
    n = 250
    table = pa.table(
        {
            "body_id": pa.array([9_000_000_000_000 + i for i in range(n)], pa.int64()),
            "weight": pa.array([float(i % 7) if i % 50 else None for i in range(n)], pa.float32()),
            "kind": pa.array(["PreSyn" if i % 3 else "PostSyn" for i in range(n)]),
            "label": pa.array([f"cell_{i}" for i in range(n)]),  # high cardinality
            "cls": pa.array(["a", "b", None, "a"] * (n // 4) + ["a"] * (n % 4)).dictionary_encode(),
        }
    )
    feather.write_feather(table, str(path), chunksize=64)  # several record batches


def test_inspect_feather_measures_columns(tmp_path: Path) -> None:
    f = tmp_path / "synthetic.feather"
    _write(f)
    r = inspect_feather(f)
    assert r["num_rows"] == 250
    assert r["num_record_batches"] == 4  # ceil(250 / 64)
    cols = r["columns"]
    assert isinstance(cols, dict)
    assert cols["body_id"]["min"] == 9_000_000_000_000
    assert cols["body_id"]["max"] == 9_000_000_000_249
    assert cols["weight"]["nulls"] == 5  # i = 0, 50, 100, 150, 200
    assert cols["weight"]["max"] == 6.0
    assert cols["kind"]["value_counts"] == {"PreSyn": 166, "PostSyn": 84}
    assert str(cols["label"]["value_counts"]).startswith("high cardinality")
    assert cols["cls"]["value_counts"]["a"] == 2 * 62 + 2
    assert cols["cls"]["nulls"] == 62
    sample = r["sample_rows"]
    assert isinstance(sample, list)
    assert len(sample) == 5


def test_high_cardinality_threshold_is_respected(tmp_path: Path) -> None:
    f = tmp_path / "x.feather"
    feather.write_feather(
        pa.table({"s": [str(i) for i in range(MAX_TRACKED_VALUES)]}), str(f), chunksize=10
    )
    cols = inspect_feather(f)["columns"]
    assert isinstance(cols, dict)
    assert len(cols["s"]["value_counts"]) == MAX_TRACKED_VALUES


def test_inspect_dir_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write(raw / "a.feather")
    (raw / "SOURCE.json").write_text('{"dataset_id": "synthetic:x"}', encoding="utf-8")
    report = inspect_dir(raw)
    assert report["source"] == {"dataset_id": "synthetic:x"}
    assert isinstance(report["peak_rss_bytes"], int)

    out = tmp_path / "audit.json"
    assert main(["data", "inspect", "--raw", str(raw), "--report", str(out)]) == 0
    assert "a.feather: 250 rows" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["files"][0]["num_rows"] == 250


def test_inspect_empty_dir_is_an_error(tmp_path: Path) -> None:
    assert main(["data", "inspect", "--raw", str(tmp_path), "--report", str(tmp_path / "r")]) == 3
