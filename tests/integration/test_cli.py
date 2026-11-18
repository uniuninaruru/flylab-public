from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from flylab.cli.main import EXIT_DATA_ERROR, main


def _json_out(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    out: dict[str, object] = json.loads(capsys.readouterr().out)
    return out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.startswith("flylab 0.0.1")


def test_env(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env"]) == 0
    assert _json_out(capsys)["schema_version"] == 1


def test_sources(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sources"]) == 0
    out = _json_out(capsys)
    files = out["files"]
    assert isinstance(files, list)
    assert len(files) == 7
    assert sum(1 for f in files if f["needed_for_mvp"]) == 3
    assert out["license"] == "CC-BY-4.0"


def test_graph_inspect_tiny(
    tiny_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_path = tmp_path / "report.json"
    rc = main(
        [
            "graph", "inspect",
            "--neurons", str(tiny_dir / "neurons.csv"),
            "--edges", str(tiny_dir / "edges.csv"),
            "--duplicates", "sum",
            "--report", str(report_path),
        ]
    )  # fmt: skip
    assert rc == 0
    out = _json_out(capsys)
    report = out["report"]
    assert isinstance(report, dict)
    assert report["n_unique_edges"] == 7
    assert report["total_synapses"] == 25
    assert json.loads(report_path.read_text(encoding="utf-8")) == out


def test_graph_inspect_refuses_silent_aggregation(
    tiny_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "graph", "inspect",
            "--neurons", str(tiny_dir / "neurons.csv"),
            "--edges", str(tiny_dir / "edges.csv"),
        ]
    )  # fmt: skip
    assert rc == EXIT_DATA_ERROR
    assert "repeated (pre, post)" in capsys.readouterr().err


def test_memory(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["memory", "--neurons", "166691", "--edges", "1000"]) == 0
    assert _json_out(capsys)["coo_bytes"] == 12_000


def test_python_dash_m_entry_point() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "flylab", "version"], capture_output=True, text=True, check=True
    )
    assert res.stdout.startswith("flylab ")
