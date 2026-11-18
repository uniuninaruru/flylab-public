"""End-to-end `flylab run`: artifacts (spec 8.4), provenance, determinism, data validation."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from flylab.cli.main import EXIT_CONFIG_ERROR, EXIT_DATA_ERROR, EXIT_OK, main
from flylab.data.processed import write_processed
from flylab.data.synthetic import random_graph
from flylab.experiments.artifacts import read_parquet_metadata
from flylab.experiments.config import ConfigError, load_config, parse_config
from flylab.experiments.dataset import DatasetError
from flylab.experiments.runner import RunError, run_experiment
from flylab.provenance.hashing import verify_checksums

RUN_FILES = {
    "config.yaml",
    "manifest.json",
    "dataset_sha256.txt",
    "git_commit.txt",
    "spikes.parquet",
    "population_rates.parquet",
    "interventions.parquet",
    "metrics.json",
    "benchmark.json",
    "checksums.sha256",
}
V0, VTH, A, D, DELAY = -52.0, -45.0, math.exp(-0.1 / 20), math.exp(-0.1 / 5), 18


def _tiny_cfg(repo_root: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load((repo_root / "configs" / "tiny_fixture.yaml").read_text())
    fixture = repo_root / "tests" / "fixtures" / "tiny_graph"
    cfg["dataset"]["neurons"] = str(fixture / "neurons.csv")
    cfg["dataset"]["edges"] = str(fixture / "edges.csv")
    return cfg


@pytest.fixture
def tiny_cfg(repo_root: Path) -> dict[str, Any]:
    if not (repo_root / "configs").is_dir():
        pytest.skip("configs/ is not part of this checkout (e.g. the CPU Docker image)")
    return _tiny_cfg(repo_root)


def _write(cfg: dict[str, Any], path: Path) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _spikes(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(run_dir / "spikes.parquet").to_pylist()
    return rows


def _first_crossing(w_mV: float) -> int:
    """Steps after the arrival step until a single arrival of w crosses threshold (closed form)."""
    return next(k for k in range(1, 10_000) if w_mV * 5 / (5 - 20) * (D**k - A**k) > VTH - V0)


def test_cli_run_writes_the_artifact_contract(
    tiny_cfg: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _write(tiny_cfg, tmp_path / "tiny.yaml")
    rc = main(
        [
            "run",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "runs"),
            "--run-id",
            "t1",
            "--with-baseline",
        ]
    )
    assert rc == EXIT_OK
    out = json.loads(capsys.readouterr().out.split("\n", 4)[-1])
    assert [r["variant"] for r in out["runs"]] == ["baseline", "main"]
    main_dir, base_dir = tmp_path / "runs" / "t1", tmp_path / "runs" / "t1--baseline"
    for d in (main_dir, base_dir):
        assert {p.name for p in d.iterdir()} == RUN_FILES
        assert verify_checksums(d / "checksums.sha256", d) == []

    manifest = json.loads((main_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run_group"] == {
        "group_id": "t1",
        "members": {"baseline": "t1--baseline", "main": "t1"},
    }
    params = manifest["model"]["parameters"]
    assert params["w_syn"]["evidence"] == "ASSUMED"  # the fixture override, with its rationale
    assert params["tau_m"]["evidence"] == "LITERATURE_DERIVED"
    assert manifest["time"]["dt"]["evidence"] == "ENGINEERED"
    assert manifest["synapse_sign"]["unknown_sign_policy"] == "exclude"
    assert manifest["network"]["n_unknown_sign_edges_before_policy"] == 1
    assert manifest["network"]["n_edges"] == 6  # 7 unique edges minus the unknown-sign one
    assert manifest["dataset"]["biological_data"] is False
    assert manifest["environment"]["git"]["source"] in ("git", "env:FLYLAB_GIT_COMMIT", None)
    assert "hostname" not in json.dumps(manifest).lower()
    sha_lines = (main_dir / "dataset_sha256.txt").read_text().splitlines()
    assert [ln.split("  ")[1].rsplit("/", 1)[-1] for ln in sha_lines] == [
        "neurons.csv",
        "edges.csv",
    ]

    md = read_parquet_metadata(main_dir / "spikes.parquet")
    assert md["flylab.schema_version"] == "1"
    assert json.loads(md["flylab.units"])["t_ms"].startswith("ms")
    frozen = load_config(main_dir / "config.yaml")
    assert frozen.to_dict() == parse_config(tiny_cfg).to_dict()


def test_tiny_fixture_spike_times_match_closed_form(
    tiny_cfg: dict[str, Any], tmp_path: Path
) -> None:
    """Hand check of the disinhibition demo (configs/tiny_fixture.yaml, w_syn = 25 mV).

    a=900001 is forced at 10, 30, 50 ms. a->b has 5 synapses (+125 mV), a->c 2 (+50 mV);
    b->d is inhibitory (5 synapses), c->d excitatory (4 synapses, +100 mV). Each spike of a
    makes b and c fire after the closed-form crossing time; with b's output blocked, d fires
    after the closed-form crossing time of c's +100 mV arrival; in the baseline it never fires.
    """
    results = run_experiment(
        _write(tiny_cfg, tmp_path / "t.yaml"),
        tmp_path / "runs",
        "t",
        with_baseline=True,
        progress=None,
    )
    base, blocked = (_spikes(r.run_dir) for r in results)
    k_b, k_c, k_d = _first_crossing(125.0), _first_crossing(50.0), _first_crossing(100.0)
    expected_base = []
    for onset in (100, 300, 500):
        expected_base += [
            (onset, 900001, "forced", True),
            (onset + DELAY + k_b, 900002, "threshold", True),
            (onset + DELAY + k_c, 900003, "threshold", True),
        ]
    got_base = [(r["step"], r["body_id"], r["cause"], r["propagated"]) for r in base]
    assert got_base == sorted(expected_base)
    expected_blocked = []
    for onset in (100, 300, 500):
        t_c = onset + DELAY + k_c
        expected_blocked += [
            (onset, 900001, "forced", True),
            (onset + DELAY + k_b, 900002, "threshold", False),
            (t_c, 900003, "threshold", True),
            (t_c + DELAY + k_d, 900004, "threshold", True),
        ]
    got_blocked = [(r["step"], r["body_id"], r["cause"], r["propagated"]) for r in blocked]
    assert got_blocked == sorted(expected_blocked)
    vs = results[1].metrics["vs_baseline"]
    assert isinstance(vs, dict)
    assert vs["populations"]["downstream"]["difference_hz"] > 0


def test_same_seed_same_outputs_different_seed_different_poisson(tmp_path: Path) -> None:
    cfg: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "det",
        "backend": "numpy",
        "dt_ms": 0.1,
        "steps": 600,
        "seed": 11,
        "dataset": {
            "kind": "synthetic_random",
            "n_neurons": 300,
            "mean_out_degree": 15,
            "graph_seed": 2,
        },
        "model": {
            "parameter_set": "shiu2024_code_v1",
            "overrides": {
                "w_syn": {"value": 2.0, "units": "mV", "evidence": "ASSUMED", "rationale": "test"}
            },
        },
        "synapse_sign": {"source": "graph", "unknown_sign_policy": "error"},
        "populations": {"p": {"body_ids": [9_000_000_000_000 + i for i in range(30)]}},
        "stimuli": [
            {
                "kind": "forced_spikes",
                "population": "p",
                "onset_ms": 0.0,
                "offset_ms": 60.0,
                "rate_hz": 80.0,
            }
        ],
        "interventions": [
            {"kind": "remove_synapses", "pre_population": "p", "fraction": 0.3, "seed": 5}
        ],
        "recording": {"population_rate_bin_ms": 5.0},
    }
    path = _write(cfg, tmp_path / "det.yaml")
    r1 = run_experiment(path, tmp_path / "runs", "r1", progress=None)[0]
    r2 = run_experiment(path, tmp_path / "runs", "r2", progress=None)[0]
    for name in ("spikes.parquet", "population_rates.parquet", "interventions.parquet"):
        t1, t2 = pq.read_table(r1.run_dir / name), pq.read_table(r2.run_dir / name)
        assert t1.equals(t2), name
    assert len(_spikes(r1.run_dir)) > 50
    other = copy.deepcopy(cfg)
    other["seed"] = 12
    r3 = run_experiment(
        _write(other, tmp_path / "s12.yaml"), tmp_path / "runs", "r3", progress=None
    )[0]
    forced = [(r["step"], r["idx"]) for r in _spikes(r1.run_dir) if r["cause"] == "forced"]
    forced3 = [(r["step"], r["idx"]) for r in _spikes(r3.run_dir) if r["cause"] == "forced"]
    assert forced != forced3
    # The removed edge subset depends on the intervention seed, not the run seed.
    removed = pq.read_table(r1.run_dir / "interventions.parquet").filter(
        pa.compute.equal(
            pq.read_table(r1.run_dir / "interventions.parquet")["kind"], "remove_synapses"
        )
    )
    removed3 = pq.read_table(r3.run_dir / "interventions.parquet").filter(
        pa.compute.equal(
            pq.read_table(r3.run_dir / "interventions.parquet")["kind"], "remove_synapses"
        )
    )
    assert removed.equals(removed3)


def _processed(tmp_path: Path, nt: Sequence[str | None] | None = None) -> Path:
    g, r = random_graph(200, 12, seed=4)
    rng = np.random.default_rng(0)
    values = ["acetylcholine", "gaba", "glutamate", "dopamine", "unclear", None]
    nt = nt or [values[i] for i in rng.integers(0, len(values), g.n_neurons)]
    unsigned = type(g)(
        g.index,
        g.src,
        g.dst,
        g.synapse_count,
        np.zeros_like(g.sign),
        g.duplicate_rule,
        "synthetic:nt",
    )
    return write_processed(
        tmp_path / "proc",
        unsigned,
        r,
        {
            "type": pa.array([f"T{i % 5}" for i in range(g.n_neurons)]),
            "predicted_nt": pa.array(nt),
            "consensus_nt": pa.array(nt),
        },
        {"node_set": {"rule": "synthetic"}},
    )


def _processed_cfg(proc: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "proc",
        "backend": "numpy",
        "dt_ms": 0.1,
        "steps": 400,
        "seed": 3,
        "dtype": "float32",
        "dataset": {"kind": "processed", "path": str(proc), "expected_dataset_id": "synthetic:nt"},
        "model": {"parameter_set": "shiu2024_code_v1"},
        "synapse_sign": {
            "source": "transmitter",
            "transmitter_column": "consensus_nt",
            "mapping": "fast_ionotropic_v1",
            "unknown_sign_policy": "exclude",
        },
        "populations": {"t0": {"column": "type", "values": ["T0"]}},
        "stimuli": [
            {
                "kind": "injected_current",
                "population": "t0",
                "onset_ms": 5.0,
                "offset_ms": 30.0,
                "amplitude_mV": 25.0,
            }
        ],
        "interventions": [],
        "recording": {"population_rate_bin_ms": 10.0},
        "sensitivity": [
            {"name": "glu-exc", "synapse_sign": {"overrides": {"glutamate": 1}}},
            {"name": "unknown-exc", "synapse_sign": {"unknown_sign_policy": "as_excitatory"}},
        ],
    }


def test_processed_dataset_with_transmitter_signs_and_sensitivity(tmp_path: Path) -> None:
    proc = _processed(tmp_path)
    cfg = _processed_cfg(proc)
    results = run_experiment(
        _write(cfg, tmp_path / "p.yaml"), tmp_path / "runs", "p", progress=None
    )
    assert [r.variant for r in results] == ["main", "sens-glu-exc", "sens-unknown-exc"]
    reports = [
        json.loads((r.run_dir / "manifest.json").read_text())["synapse_sign"] for r in results
    ]
    main_r, glu, unk = reports
    assert main_r["final_table"]["glutamate"] == -1
    assert glu["final_table"]["glutamate"] == 1
    assert unk["unknown_sign_policy"] == "as_excitatory"
    assert sum(main_r["neurons_per_value"].values()) == 200
    assert main_r["mapping_evidence"] == "ASSUMED"
    nets = [json.loads((r.run_dir / "manifest.json").read_text())["network"] for r in results]
    assert nets[0]["n_edges"] < nets[2]["n_edges"]  # exclude drops unknown-sign edges
    assert nets[1]["n_inhibitory_edges"] < nets[0]["n_inhibitory_edges"]
    m = json.loads((results[0].run_dir / "manifest.json").read_text())
    assert {f["path"].rsplit("/", 1)[-1] for f in m["dataset"]["files"]} == {
        "graph.npz",
        "neurons.parquet",
        "manifest.json",
    }
    assert m["run_group"]["members"] == {
        "main": "p",
        "sens-glu-exc": "p--sens-glu-exc",
        "sens-unknown-exc": "p--sens-unknown-exc",
    }


def test_processed_dataset_integrity_and_column_checks(tmp_path: Path) -> None:
    proc = _processed(tmp_path)
    cfg = _processed_cfg(proc)
    bad = copy.deepcopy(cfg)
    bad["synapse_sign"]["transmitter_column"] = "nt_typo"
    with pytest.raises(ConfigError, match="'nt_typo' is not a neuron-table column"):
        run_experiment(_write(bad, tmp_path / "b.yaml"), tmp_path / "runs", progress=None)
    bad = copy.deepcopy(cfg)
    bad["dataset"]["expected_dataset_id"] = "male-cns:v1.0"
    with pytest.raises(DatasetError, match=r"expected 'male-cns:v1\.0'"):
        run_experiment(_write(bad, tmp_path / "b2.yaml"), tmp_path / "runs", progress=None)
    with (proc / "graph.npz").open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(DatasetError, match=r"does not match checksums\.sha256"):
        run_experiment(_write(cfg, tmp_path / "c.yaml"), tmp_path / "runs", progress=None)
    assert not (tmp_path / "runs").exists()  # nothing written on failure


def test_unmapped_transmitter_is_a_config_error(tmp_path: Path) -> None:
    proc = _processed(tmp_path, nt=["acetylcholine"] * 199 + ["tyramine"])
    cfg = _processed_cfg(proc)
    with pytest.raises(ConfigError, match="tyramine"):
        run_experiment(_write(cfg, tmp_path / "u.yaml"), tmp_path / "runs", progress=None)


def test_error_policy_with_unknown_signs_is_a_config_error(tmp_path: Path) -> None:
    cfg = _processed_cfg(_processed(tmp_path))
    cfg["synapse_sign"]["unknown_sign_policy"] = "error"
    cfg["sensitivity"] = []
    with pytest.raises(ConfigError, match="unknown sign"):
        run_experiment(_write(cfg, tmp_path / "e.yaml"), tmp_path / "runs", progress=None)


def test_refuses_to_overwrite_and_validate_only_writes_nothing(
    tiny_cfg: dict[str, Any], tmp_path: Path
) -> None:
    path = _write(tiny_cfg, tmp_path / "t.yaml")
    assert run_experiment(path, tmp_path / "runs", "same", validate_only=True, progress=None) == []
    assert not (tmp_path / "runs").exists()
    run_experiment(path, tmp_path / "runs", "same", progress=None)
    with pytest.raises(RunError, match="refusing to overwrite"):
        run_experiment(path, tmp_path / "runs", "same", progress=None)
    with pytest.raises(RunError, match="run id"):
        run_experiment(path, tmp_path / "runs", "../escape", progress=None)
    no_iv = copy.deepcopy(tiny_cfg)
    no_iv["interventions"] = []
    no_iv_path = _write(no_iv, tmp_path / "n.yaml")
    with pytest.raises(ConfigError, match="no interventions"):
        run_experiment(no_iv_path, tmp_path / "runs", with_baseline=True, progress=None)


def test_cli_exit_codes(
    tmp_path: Path, tiny_cfg: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    bad = copy.deepcopy(tiny_cfg)
    bad["populations"]["source_a"] = {"body_ids": [42]}
    assert main(["run", "--config", str(_write(bad, tmp_path / "b.yaml"))]) == EXIT_CONFIG_ERROR
    assert "not in neuron table" in capsys.readouterr().err
    missing = copy.deepcopy(tiny_cfg)
    missing["dataset"]["edges"] = str(tmp_path / "nope.csv")
    assert main(["run", "--config", str(_write(missing, tmp_path / "m.yaml"))]) == EXIT_DATA_ERROR
    ok = _write(tiny_cfg, tmp_path / "ok.yaml")
    assert main(["run", "--config", str(ok), "--validate-only"]) == EXIT_OK
    assert "config OK" in capsys.readouterr().out


def test_example_configs(repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = repo_root / "configs"
    if not configs.is_dir():
        pytest.skip("configs/ is not part of this checkout (e.g. the CPU Docker image)")
    monkeypatch.chdir(repo_root)  # example paths are relative to the repository root
    for name in ("tiny_fixture.yaml", "synthetic.yaml"):
        assert run_experiment(configs / name, tmp_path, validate_only=True, progress=None) == []
    # The LB3c -> MN9 config passes the schema stage; its body ids need the real MaleCNS data.
    lb3c = load_config(configs / "male_cns_lb3c_mn9.yaml")
    assert lb3c.synapse_sign.mapping == "shiu2024_methods_v1"
    assert [v.name for v in lb3c.sensitivity] == [
        "fast-ionotropic-map",
        "predicted-nt-column",
        "glutamate-excitatory",
    ]
    # The MaleCNS template must be rejected until a real population is chosen (spec 8.1).
    template: dict[str, Any] = yaml.safe_load((configs / "male_cns_template.yaml").read_text())
    template["dataset"]["path"] = str(_processed(tmp_path))
    template["dataset"]["expected_dataset_id"] = "synthetic:nt"
    with pytest.raises(ConfigError, match="PLACEHOLDER_VERIFY_WITH_DATA' matches no neuron"):
        run_experiment(_write(template, tmp_path / "tpl.yaml"), tmp_path / "runs", progress=None)


def test_cuda_backend_without_cuda_is_a_run_error(tiny_cfg: dict[str, Any], tmp_path: Path) -> None:
    from flylab.neural.lif_torch import cuda_status

    if cuda_status()["available"]:
        pytest.skip("this host has CUDA; tests/gpu covers torch-cuda runs")
    cfg = copy.deepcopy(tiny_cfg)
    cfg["backend"] = "torch-cuda"
    path = _write(cfg, tmp_path / "cuda.yaml")
    for validate_only in (True, False):
        with pytest.raises(RunError, match=r"needs (a CUDA device|PyTorch)"):
            run_experiment(path, tmp_path / "runs", "c", progress=None, validate_only=validate_only)
    assert not (tmp_path / "runs").exists()


def test_benchmark_json_marks_host_backends_without_gpu_memory(
    tiny_cfg: dict[str, Any], tmp_path: Path
) -> None:
    [res] = run_experiment(_write(tiny_cfg, tmp_path / "t.yaml"), tmp_path / "runs", "t",
                           progress=None)  # fmt: skip
    bench = json.loads((res.run_dir / "benchmark.json").read_text(encoding="utf-8"))
    m = bench["measured"]
    assert m["gpu_max_memory_allocated_bytes"] is None
    assert m["gpu_max_memory_reserved_bytes"] is None
    assert m["engine_retained_bytes"]["edges"] == 0  # NumPy reads the network in place
    assert m["engine_retained_bytes"]["state"] > 0


def test_torch_backend_matches_numpy_end_to_end(tiny_cfg: dict[str, Any], tmp_path: Path) -> None:
    from flylab.neural.lif_torch import torch_available

    cfg = copy.deepcopy(tiny_cfg)
    cfg["backend"] = "torch-cpu"
    path = _write(cfg, tmp_path / "torch.yaml")
    if not torch_available():
        with pytest.raises(RunError, match="needs PyTorch"):
            run_experiment(path, tmp_path / "runs", "t", progress=None)
        pytest.skip("torch not installed; checked the error path only")
    rt = run_experiment(path, tmp_path / "runs", "t", progress=None)[0]
    rn = run_experiment(
        _write(tiny_cfg, tmp_path / "np.yaml"), tmp_path / "runs", "n", progress=None
    )[0]
    assert pq.read_table(rt.run_dir / "spikes.parquet").equals(
        pq.read_table(rn.run_dir / "spikes.parquet")
    )
    engine = json.loads((rt.run_dir / "manifest.json").read_text())["engine"]
    assert engine["backend"] == "torch-cpu"
    assert engine["deterministic_algorithms"] is True


def test_interrupted_run_is_left_as_partial(
    tiny_cfg: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flylab.neural.lif_numpy import NumpyLIFEngine

    def boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(NumpyLIFEngine, "simulate", boom)
    path = _write(tiny_cfg, tmp_path / "t.yaml")
    with pytest.raises(KeyboardInterrupt):
        run_experiment(path, tmp_path / "runs", "cut", progress=None)
    assert not (tmp_path / "runs" / "cut").exists()
    assert (tmp_path / "runs" / ".cut.partial" / "config.yaml").is_file()
    monkeypatch.undo()
    with pytest.raises(RunError, match="already exists"):
        run_experiment(path, tmp_path / "runs", "cut", progress=None)


def test_pr01_unknown_signs_are_never_silently_zeroed(tmp_path: Path) -> None:
    """Issue #10: PR-01 stores sign 0 (unknown) for every edge. PR-02 never multiplies count by
    the raw sign; with `source: graph` the run either refuses (error) or records the policy."""
    cfg = _processed_cfg(_processed(tmp_path))
    cfg["synapse_sign"] = {"source": "graph", "unknown_sign_policy": "error"}
    cfg["sensitivity"] = []
    with pytest.raises(ConfigError, match="unknown sign"):
        run_experiment(_write(cfg, tmp_path / "err.yaml"), tmp_path / "runs", progress=None)
    for policy, n_edges in (("exclude", 0), ("as_excitatory", None)):
        cfg["synapse_sign"]["unknown_sign_policy"] = policy
        r = run_experiment(_write(cfg, tmp_path / f"{policy}.yaml"), tmp_path / "runs", policy,
                           progress=None)[0]  # fmt: skip
        net = json.loads((r.run_dir / "manifest.json").read_text())["network"]
        total = net["n_unknown_sign_edges_before_policy"]
        assert total > 0
        assert net["unknown_sign_policy"] == policy
        assert net["n_zero_weight_edges"] == 0
        assert net["n_edges"] == (total if n_edges is None else n_edges)
