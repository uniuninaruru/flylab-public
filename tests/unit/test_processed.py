from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from flylab.data.processed import load_graph, verify_processed, write_processed
from flylab.data.synthetic import random_graph


def _write(tmp_path: Path) -> Path:
    g, r = random_graph(500, 6, seed=3)
    labels = pa.array([f"t{i % 7}" if i % 11 else None for i in range(g.n_neurons)])
    return write_processed(
        tmp_path / "proc", g, r, {"cell_type": labels}, {"node_set": {"rule": "synthetic"}}
    )


def test_roundtrip_and_verify(tmp_path: Path) -> None:
    out = _write(tmp_path)
    g, _ = random_graph(500, 6, seed=3)
    loaded = load_graph(out)
    assert loaded.fingerprint() == g.fingerprint()
    rep = verify_processed(out)
    assert rep["ok"] is True, rep["problems"]
    measured = rep["measured"]
    assert isinstance(measured, dict)
    assert measured["n_neurons"] == 500
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["node_set"] == {"rule": "synthetic"}
    assert manifest["orientation"].startswith("src=presynaptic")


def test_verify_detects_tampering(tmp_path: Path) -> None:
    out = _write(tmp_path)
    with np.load(out / "graph.npz") as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["dst"][0] = 10_000  # dangling index
    np.savez(out / "graph.npz", **arrays)
    rep = verify_processed(out)
    assert rep["ok"] is False
    problems = " ".join(map(str, rep["problems"]))  # type: ignore[call-overload]
    assert "checksum mismatch: graph.npz" in problems
    assert "dangling" in problems


def test_verify_reports_missing_files(tmp_path: Path) -> None:
    out = _write(tmp_path)
    (out / "edges.parquet").unlink()
    assert verify_processed(out)["ok"] is False


def test_neuron_column_length_checked(tmp_path: Path) -> None:
    g, r = random_graph(10, 2, seed=0)
    with pytest.raises(ValueError, match="rows"):
        write_processed(tmp_path, g, r, {"x": pa.array([1, 2])}, {})
