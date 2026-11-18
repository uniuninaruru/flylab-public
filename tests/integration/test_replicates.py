"""Replicate groups (`flylab replicate`): expansion, pairing, statistics, resume, CLI.

Synthetic random graphs only: these test the bookkeeping and statistics, not flies.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pytest
import yaml

from flylab.cli.main import EXIT_CONFIG_ERROR, EXIT_OK, main
from flylab.data.synthetic import SYNTHETIC_ID_BASE
from flylab.experiments.config import ConfigError, parse_config
from flylab.experiments.dataset import load_dataset
from flylab.experiments.replicates import (
    deep_merge,
    expand_members,
    parse_group,
    recompute_statistics,
    run_group,
    weakest,
)
from flylab.experiments.runner import RunError, run_experiment
from flylab.provenance.hashing import verify_checksums

BASE: dict[str, Any] = {
    "schema_version": 1,
    "experiment_id": "rep_base",
    "dataset": {"kind": "synthetic_random", "n_neurons": 300, "mean_out_degree": 15,
                "graph_seed": 11},
    "backend": "numpy",
    "dt_ms": 0.1,
    "steps": 600,
    "seed": 1,
    "model": {
        "parameter_set": "shiu2024_code_v1",
        "overrides": {"w_syn": {"value": 2.0, "units": "mV", "evidence": "ASSUMED",
                                "rationale": "synthetic test graph"}},
    },
    "synapse_sign": {"source": "graph", "unknown_sign_policy": "error"},
    "populations": {
        "input": {"body_ids": [SYNTHETIC_ID_BASE + i for i in range(10)]},
        "probe": {"body_ids": [SYNTHETIC_ID_BASE + i for i in range(100, 110)]},
    },
    "stimuli": [{"kind": "forced_spikes", "population": "input", "onset_ms": 0.0,
                 "offset_ms": 60.0, "rate_hz": 150.0}],
    "interventions": [],
    "recording": {"spike_events": False, "population_rate_bin_ms": 10.0},
}  # fmt: skip


def _group(base_path: Path, **extra: Any) -> dict[str, Any]:
    g: dict[str, Any] = {
        "schema_version": 1,
        "kind": "flylab_replicate_group",
        "group_id": "g",
        "base_config": str(base_path),
        "seeds": {"first": 1, "count": 3},
        "conditions": [
            {"name": "intact"},
            {"name": "same", "patch": {}},
            {"name": "silenced", "patch": {"interventions": [
                {"kind": "silence_neurons", "mode": "block_output", "population": "input"}]}},
        ],
        "comparisons": [{"condition": "same", "reference": "intact"},
                        {"condition": "silenced", "reference": "intact"}],
        "statistics": {"n_boot": 500, "psth": True, "population_contrasts": [
            {"population": "input", "reference": "probe"}]},
    }  # fmt: skip
    g.update(extra)
    return g


@pytest.fixture
def base_path(tmp_path: Path) -> Path:
    p = tmp_path / "base.yaml"
    p.write_text(yaml.safe_dump(BASE), encoding="utf-8")
    return p


def test_deep_merge_semantics() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}
    out = deep_merge(base, {"a": {"y": 5, "z": 6}, "b": [9], "c": None})
    assert out == {"a": {"x": 1, "y": 5, "z": 6}, "b": [9]}
    assert base == {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 3}  # input unchanged


def test_expansion_sets_seed_and_ids(base_path: Path) -> None:
    group = parse_group(_group(base_path, seeds=[4, 9]))
    members = expand_members(group)
    assert [(m.condition, m.seed) for m in members] == [
        ("intact", 4), ("intact", 9), ("same", 4), ("same", 9), ("silenced", 4), ("silenced", 9)
    ]  # fmt: skip
    assert members[1].config.seed == 9
    assert members[1].run_id == "intact--s9"
    assert members[1].config.experiment_id == "g.intact"
    assert members[4].config.interventions[0].kind == "silence_neurons"
    # "same" has an empty patch: its members are the "intact" configs except for the id
    assert members[0].config.to_dict() | {"experiment_id": ""} == members[2].config.to_dict() | {
        "experiment_id": ""
    }


def test_sweep_and_replicate_seed_for_null_graph(base_path: Path) -> None:
    g = _group(
        base_path,
        conditions=[],
        comparisons=[],
        sweep=[
            {"axis": "w", "levels": [
                {"label": "w1", "patch": {"model": {"overrides": {"w_syn": {
                    "value": 1.0, "units": "mV", "evidence": "ASSUMED", "rationale": "t"}}}}},
                {"label": "w2", "patch": {}}]},
            {"axis": "graph", "levels": [
                {"label": "real", "patch": {}},
                {"label": "null", "patch": {"graph_null": {
                    "method": "degree_preserving_swap", "seed": "replicate"}}}]},
        ],
    )  # fmt: skip
    group = parse_group(g)
    assert [c.name for c in group.conditions] == ["w1_real", "w1_null", "w2_real", "w2_null"]
    members = {(m.condition, m.seed): m.config for m in expand_members(group)}
    assert members[("w1_real", 1)].model.parameters().w_syn.value == 1.0
    assert members[("w2_real", 1)].model.parameters().w_syn.value == 2.0
    gn = members[("w1_null", 3)].graph_null
    assert gn is not None
    assert gn.seed == 3
    assert members[("w1_real", 3)].graph_null is None


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"kind": "something"}, "kind"),
        ({"comparisons": [{"condition": "nope", "reference": "intact"}]}, "not a condition"),
        ({"comparisons": [{"condition": "same", "reference": "same"}]}, "must differ"),
        ({"conditions": [{"name": "a"}, {"name": "a"}]}, "duplicate"),
        ({"seeds": [1, 1]}, "duplicates"),
        ({"extra_key": 1}, "unknown key"),
        ({"statistics": {"confidence": 1.5}}, "confidence"),
        ({"statistics": {"metrics": ["mean_rate_hz", "nope"]}}, "metrics"),
        (
            {"statistics": {"population_contrasts": [{"population": "a", "reference": "a"}]}},
            "must differ",
        ),
    ],
)
def test_group_validation_errors(base_path: Path, change: dict[str, Any], match: str) -> None:
    g = _group(base_path)
    g.update(change)
    with pytest.raises(ConfigError, match=match):
        expand_members(parse_group(g))


def test_member_config_errors_name_the_condition(tmp_path: Path) -> None:
    base = copy.deepcopy(BASE)
    base["sensitivity"] = [{"name": "v", "synapse_sign": {"unknown_sign_policy": "exclude"}}]
    p = tmp_path / "b.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(ConfigError, match="sensitivity"):
        expand_members(parse_group(_group(p)))
    bad = _group(tmp_path / "b.yaml", base_patch={"sensitivity": None})
    bad["conditions"].append({"name": "broken", "patch": {"dt_ms": -1}})
    with pytest.raises(ConfigError, match="condition 'broken'"):
        expand_members(parse_group(bad))


def _stats(group_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((group_dir / "group_stats.json").read_text())
    return out


def test_run_group_pairs_by_seed_and_reports_statistics(base_path: Path, tmp_path: Path) -> None:
    group = parse_group(_group(base_path))
    res = run_group(group, out_root=tmp_path / "runs", progress=None)
    assert (res.n_members, res.n_run, res.n_reused) == (9, 9, 0)
    gd = res.group_dir
    assert not verify_checksums(gd / "checksums.sha256", gd)
    st = _stats(gd)
    assert st["evidence"]["label"] == "ASSUMED"  # w_syn override is ASSUMED
    assert st["statistics_method"]["multiple_comparisons"]["method"] == "Holm step-down"
    probe = st["conditions"]["intact"]["populations"]["probe"]["mean_rate_hz"]
    assert probe["n"] == 3
    assert set(probe["values_by_seed"]) == {"1", "2", "3"}
    psth = st["conditions"]["intact"]["psth_mean_rate_hz"]["input"]
    assert len(psth) == 6
    assert st["conditions"]["intact"]["psth_bin_ms"] == pytest.approx(10.0)
    comps = {(c["condition"], c["population"], c["metric"]): c for c in st["comparisons"]}
    # identical configs with identical seeds are identical runs: difference exactly 0
    same = comps[("same", "__all__", "mean_rate_hz")]
    assert same["mean_difference"] == 0.0
    assert same["ci_mean_difference"]["method"] == "degenerate"
    # silencing the stimulated population's output removes the evoked activity downstream
    sil = comps[("silenced", "__all__", "mean_rate_hz")]
    assert sil["mean_difference"] < 0
    assert sil["n_pairs"] == 3
    assert all("p_holm" in c for c in st["comparisons"])
    # population contrast: input vs probe inside the same runs, one entry per condition x metric
    con = {(c["condition"], c["metric"]): c for c in st["population_contrasts"]}
    assert len(con) == 3 * 3
    rates = st["conditions"]["intact"]["populations"]
    by_seed = [
        rates["input"]["mean_rate_hz"]["values_by_seed"][k]
        - rates["probe"]["mean_rate_hz"]["values_by_seed"][k]
        for k in ("1", "2", "3")
    ]
    assert con[("intact", "mean_rate_hz")]["mean_difference"] == pytest.approx(sum(by_seed) / 3)
    assert st["statistics_method"]["multiple_comparisons"]["m"] == sum(
        c.get("sign_flip_test", {}).get("p_value") is not None
        for c in st["comparisons"] + st["population_contrasts"]
    )
    manifest = json.loads((gd / "group_manifest.json").read_text())
    assert len(manifest["members"]) == 9
    assert "host" in manifest
    assert "commit" in manifest["git"]


def test_resume_reuses_and_refuses_changes(base_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    g = _group(base_path, seeds=[1, 2])
    run_group(parse_group(g), out_root=out, progress=None)
    with pytest.raises(RunError, match="exists"):
        run_group(parse_group(g), out_root=out, progress=None)
    # an interrupted member is removed and rerun; complete members are reused
    gd = out / "g"
    (gd / "members" / ".intact--s1.partial").mkdir()
    res = run_group(parse_group(g), out_root=out, resume=True, progress=None)
    assert (res.n_run, res.n_reused) == (0, 6)
    assert not (gd / "members" / ".intact--s1.partial").exists()
    g2 = copy.deepcopy(g)
    g2["statistics"]["n_boot"] = 600
    with pytest.raises(RunError, match="differs"):
        run_group(parse_group(g2), out_root=out, resume=True, progress=None)
    before = _stats(gd)
    recompute_statistics(gd)
    assert _stats(gd)["comparisons"] == before["comparisons"]


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_parallel_jobs_give_the_same_results(base_path: Path, tmp_path: Path) -> None:
    g = _group(base_path, seeds=[1, 2])
    a = run_group(parse_group(g), out_root=tmp_path / "a", jobs=1, progress=None)
    b = run_group(parse_group(g), out_root=tmp_path / "b", jobs=2, progress=None)
    assert _stats(a.group_dir)["conditions"] == _stats(b.group_dir)["conditions"]


def test_preloaded_dataset_must_match_the_config(tmp_path: Path) -> None:
    cfg = parse_config(BASE)
    other = copy.deepcopy(BASE)
    other["dataset"]["graph_seed"] = 12
    data = load_dataset(parse_config(other).dataset)
    with pytest.raises(ConfigError, match="preloaded"):
        run_experiment(cfg, out_root=tmp_path, progress=None, data=data)


def test_graph_null_config_runs_and_is_recorded(tmp_path: Path) -> None:
    raw = copy.deepcopy(BASE)
    raw["graph_null"] = {"method": "degree_preserving_swap", "seed": 5}
    cfg = parse_config(raw)
    assert "graph_null" in cfg.to_yaml()
    assert parse_config(yaml.safe_load(cfg.to_yaml())) == cfg
    res = run_experiment(cfg, out_root=tmp_path, run_id="n", progress=None)
    man = json.loads((res[0].run_dir / "manifest.json").read_text())
    assert man["graph_null"]["method"] == "degree_preserving_swap"
    assert man["graph_null"]["in_degree_preserved"] is True
    null_fp = man["graph_null"]["null_fingerprint_sha256"]
    assert man["dataset"]["graph_fingerprint_sha256"] == null_fp
    assert any("CONTROL graph" in x for x in man["limitations"])
    raw["graph_null"] = {"method": "nope", "seed": 1}
    with pytest.raises(ConfigError, match=r"graph_null\.method"):
        parse_config(raw)


def test_weakest_label_order() -> None:
    assert weakest(["EMPIRICAL", "LITERATURE_DERIVED"]) == "LITERATURE_DERIVED"
    assert weakest(["LEARNED", "ENGINEERED", "ASSUMED"]) == "ASSUMED"
    assert weakest(["UNKNOWN", "ASSUMED"]) == "UNKNOWN"


def test_cli_replicate(base_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gp = tmp_path / "group.yaml"
    gp.write_text(yaml.safe_dump(_group(base_path, seeds=[1, 2])), encoding="utf-8")
    assert main(["replicate", "run", "--config", str(gp), "--validate-only"]) == EXIT_OK
    assert "6 members" in capsys.readouterr().out
    out = tmp_path / "runs"
    assert main(["replicate", "run", "--config", str(gp), "--out", str(out)]) == EXIT_OK
    text = capsys.readouterr().out
    summary = json.loads(text[text.index("{\n") :])
    assert summary["members"] == 6
    assert summary["run_now"] == 6
    assert (out / "g" / "group_stats.json").is_file()
    assert main(["replicate", "stats", "--group-dir", str(out / "g")]) == EXIT_OK
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"kind": "x"}), encoding="utf-8")
    assert main(["replicate", "run", "--config", str(bad)]) == EXIT_CONFIG_ERROR


def test_metric_restriction_shrinks_the_tested_family(base_path: Path, tmp_path: Path) -> None:
    g = _group(base_path, seeds=[1, 2])
    g["statistics"] = {"n_boot": 300, "metrics": ["mean_rate_hz"], "populations": ["__all__"]}
    res = run_group(parse_group(g), out_root=tmp_path, progress=None)
    st = _stats(res.group_dir)
    assert {c["metric"] for c in st["comparisons"]} == {"mean_rate_hz"}
    assert {c["population"] for c in st["comparisons"]} == {"__all__"}
    assert list(st["conditions"]["intact"]["populations"]) == ["__all__"]
    assert st["statistics_method"]["multiple_comparisons"]["m"] == 2


def test_members_record_the_code_state_of_the_group_start(
    base_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers run the code loaded when the group started; a commit made while the group runs
    must not be recorded as the code of later members."""
    import flylab.experiments.replicates as rep

    start = {"commit": "a" * 40, "dirty": False, "source": "git", "reason": None}
    monkeypatch.setattr(rep, "git_state", lambda: dict(start))
    res = run_group(parse_group(_group(base_path, seeds=[1])), out_root=tmp_path, progress=None)
    for d in (res.group_dir / "members").iterdir():
        assert (d / "git_commit.txt").read_text().split("\n")[0] == "a" * 40
        env = json.loads((d / "manifest.json").read_text())["environment"]
        assert env["git"]["commit"] == "a" * 40
        assert "git_at_write" in env  # the real HEAD differs from the fake start state
    man = json.loads((res.group_dir / "group_manifest.json").read_text())
    assert man["members_code_state"]["commit"] == "a" * 40
    assert man["member_commits"] == ["a" * 40]


def test_single_run_records_state_captured_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import flylab.experiments.runner as runner

    calls: list[int] = []

    def fake() -> dict[str, object]:
        calls.append(1)
        return {"commit": f"c{len(calls)}", "dirty": False, "source": "git", "reason": None}

    monkeypatch.setattr(runner, "git_state", fake)
    res = run_experiment(parse_config(BASE), out_root=tmp_path, run_id="r", progress=None)
    assert len(calls) == 1  # captured once, at the start
    assert (res[0].run_dir / "git_commit.txt").read_text().startswith("c1\n")
