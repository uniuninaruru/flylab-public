"""`flylab bench` / `flylab parity` on host backends: record schema, statistics, CLI codes."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from flylab.cli.main import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUN_ERROR, main
from flylab.experiments.benchmark import (
    BenchmarkError,
    run_benchmark,
    summarize,
    write_record,
)
from flylab.experiments.parity import run_parity
from flylab.provenance import hardware


def test_summarize_reports_quartiles_not_only_the_best() -> None:
    s = summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s == {"n": 5, "median": 3.0, "p25": 2.0, "p75": 4.0, "iqr": 2.0,
                 "min": 1.0, "max": 5.0, "mean": 3.0}  # fmt: skip
    assert summarize([7.0])["iqr"] == 0.0
    assert summarize([])["median"] is None


@pytest.fixture
def tiny(repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if not (repo_root / "configs").is_dir():
        pytest.skip("configs/ is not part of this checkout (e.g. the CPU Docker image)")
    monkeypatch.chdir(repo_root)  # the fixture paths in the example config are repo-relative
    return repo_root / "configs" / "tiny_fixture.yaml"


def test_benchmark_record_on_numpy(tiny: Path) -> None:
    rec = run_benchmark(tiny, warmup=1, repeats=3, tag="T0", progress=None)
    assert rec["kind"] == "flylab_benchmark"
    assert rec["status"] == "PASS"
    assert rec["gates"] == {"no_oom": True, "all_runs_finite": True,
                            "vram_peak_reserved_within_10GiB_target": None}  # fmt: skip
    runs: Any = rec["repeats"]
    assert len(runs) == 3
    warm: Any = rec["warmup"]
    assert len(warm) == 1
    # Same seed, same inputs: every repeat produces the same spikes.
    assert len({r["n_spikes"] for r in runs}) == 1
    assert runs[0]["n_spikes"] > 0
    assert runs[0]["synaptic_events"] > 0
    assert 0 < runs[0]["mean_active_fraction"] < 1
    assert rec["stats"]["wall_s"]["n"] == 3  # type: ignore[index]
    phases: Any = rec["phases_s"]
    assert set(phases) >= {"load_dataset", "build_network", "engine_init_and_copy_to_device"}
    assert rec["memory"]["gpu_max_memory_reserved_bytes"] is None  # type: ignore[index]
    retained: Any = rec["memory_after_init"]["engine_retained_bytes"]  # type: ignore[index]
    assert retained["edges"] == 0  # NumPy reads the network's arrays in place
    assert retained["state"] > 0  # measured after reset(), so v, g, last_spike exist
    assert rec["dataset"]["biological_data"] is False  # type: ignore[index]
    json.dumps(rec)  # serialisable as is


def test_overrides_are_validated_like_the_config(tiny: Path) -> None:
    from flylab.experiments.config import ConfigError

    with pytest.raises(ConfigError, match="steps"):
        run_benchmark(tiny, steps=0, repeats=1, progress=None)
    with pytest.raises(BenchmarkError, match="repeats"):
        run_benchmark(tiny, repeats=0, progress=None)
    with pytest.raises(BenchmarkError, match="tag"):
        run_benchmark(tiny, repeats=1, tag="../x", progress=None)


def test_write_record_names_files_by_tag(tmp_path: Path) -> None:
    p = write_record({"tag": "B1", "status": "PASS"}, tmp_path / "o", "bench")
    assert p.name.startswith("bench-B1-")
    assert p.suffix == ".json"
    assert json.loads(p.read_text())["status"] == "PASS"
    with pytest.raises(BenchmarkError):
        write_record({"tag": "a/b"}, tmp_path, "bench")


def test_parity_refuses_comparing_the_reference_with_itself(tiny: Path) -> None:
    with pytest.raises(BenchmarkError, match="reference"):
        run_parity(tiny, progress=None)


def test_cli_parity_rejects_invalid_margin_cleanly(
    tiny: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["parity", "--config", str(tiny), "--backend", "torch-cpu",
                 "--margin-mv", "nan", "--out", str(tmp_path)]) == EXIT_RUN_ERROR  # fmt: skip
    assert "margin_mV must be finite and positive" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.json"))


def test_parity_torch_cpu_against_numpy(tiny: Path) -> None:
    pytest.importorskip("torch")
    rec = run_parity(tiny, backend="torch-cpu", progress=None)
    assert rec["status"] == "PASS"
    lock: Any = rec["lockstep"]
    assert lock["status"] == "identical"
    assert lock["steps_identical"] == lock["steps"]
    assert rec["seed_control"] is None  # only run after a divergence
    pops: Any = rec["populations"]
    assert all(row["abs_diff"] == 0 for row in pops.values())


def test_cli_bench_and_parity(tiny: Path, tmp_path: Path) -> None:
    out = tmp_path / "bench"
    rc = main(["bench", "--config", str(tiny), "--warmup", "0", "--repeats", "1",
               "--out", str(out), "--tag", "cli"])  # fmt: skip
    assert rc == EXIT_OK
    [path] = out.glob("bench-cli-*.json")
    assert json.loads(path.read_text())["status"] == "PASS"
    assert main(["parity", "--config", str(tiny), "--out", str(out)]) == EXIT_RUN_ERROR
    bad = tmp_path / "bad.yaml"
    doc = yaml.safe_load(tiny.read_text())
    doc["steps"] = -1
    bad.write_text(yaml.safe_dump(doc))
    assert main(["bench", "--config", str(bad), "--out", str(out)]) == EXIT_CONFIG_ERROR


def test_cli_cuda_without_a_device_is_a_run_error(tiny: Path, tmp_path: Path) -> None:
    from flylab.neural.lif_torch import cuda_status

    if cuda_status()["available"]:
        pytest.skip("this host has CUDA")
    rc = main(["bench", "--config", str(tiny), "--backend", "torch-cuda", "--repeats", "1",
               "--out", str(tmp_path)])  # fmt: skip
    assert rc == EXIT_RUN_ERROR
    assert not list(tmp_path.glob("*.json"))


def test_hardware_facts_are_anonymous_and_nullable(monkeypatch: pytest.MonkeyPatch) -> None:
    hw = hardware.capture_hardware()
    assert set(hw) == {"cpu_model", "logical_cpus", "machine", "total_ram_bytes",
                       "cgroup_memory_limit_bytes", "note"}  # fmt: skip
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert hardware.nvidia_smi_gpus() is None
    pm = hardware.nvidia_smi_process_memory_mib()
    assert pm == {"visible": False, "reason": "nvidia-smi not found", "used_mib": None}


def test_benchmark_configs_parse(repo_root: Path) -> None:
    """configs/bench/* stay valid; parsing needs neither torch, a GPU nor the dataset."""
    from flylab.experiments.config import load_config

    paths = sorted((repo_root / "configs" / "bench").glob("*.yaml"))
    if not paths:
        pytest.skip("configs/ is not part of this checkout")
    assert [p.stem[:2] for p in paths] == ["b0", "b1", "b2"]
    durations = {p.stem[:2]: load_config(p).duration_ms for p in paths}
    assert durations == pytest.approx({"b0": 100.0, "b1": 100.0, "b2": 1000.0})
    assert {load_config(p).backend for p in paths} == {"torch-cuda"}
