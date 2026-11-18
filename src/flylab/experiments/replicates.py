"""Replicate groups (PR-04): N seeds x conditions of one base experiment, plus statistics.

A group config (YAML, `kind: flylab_replicate_group`, `schema_version: 1`) names a base
experiment config and how each condition differs from it:

    schema_version: 1
    kind: flylab_replicate_group
    group_id: m4_controls
    base_config: configs/male_cns_lb3c_mn9.yaml
    base_patch: {sensitivity: null, recording: {spike_events: false}}  # applied to every member
    seeds: {first: 1, count: 10}            # or an explicit list [1, 2, 3]
    conditions:
      - {name: intact}                      # empty patch = the base config itself
      - {name: null_graph, patch: {graph_null: {method: degree_preserving_swap, seed: replicate}}}
    sweep:                                  # optional: cartesian product of axis levels
      - axis: w_syn
        levels: [{label: w0.15, patch: {model: {overrides: {w_syn: {...}}}}}]
    comparisons:
      - {condition: null_graph, reference: intact}
    statistics: {n_boot: 10000, confidence: 0.95, seed: 0, psth: false}

Patches are deep-merged into the base config's YAML mapping: mappings merge key by key, any other
value replaces, and `null` deletes the key. The merged document is validated by
`flylab.experiments.config.parse_config` like any `flylab run` config, so a condition cannot
introduce an unchecked field. `seed` of every member is set to the replicate seed, and a
`graph_null.seed` of `replicate` is replaced by it too (each replicate then gets its own null
graph). `sensitivity:` is refused in members: sign variants are conditions here.

Matched comparisons pair runs with the same seed: identical Poisson draws for identical
stimuli (the schedule seeds stimulus k with SeedSequence([seed, k])), identical initial state.

Layout of `<out>/<group_id>/`:

    group.yaml            canonical group config (frozen; a resume must match it exactly)
    members/<condition>--s<seed>/   ordinary run directories (spec 8.4)
    group_manifest.json   members, config hashes, status, base-config hash, code commit
    group_stats.json      per-condition descriptions and paired comparisons (stats.py)
    checksums.sha256      SHA-256 of the three group files above
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import itertools
import json
import multiprocessing as mp
import shutil
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from flylab import SCHEMA_VERSION, __version__
from flylab.experiments.artifacts import write_json
from flylab.experiments.config import (
    _NAME,
    ConfigError,
    DatasetConfig,
    ExperimentConfig,
    _int,
    _keys,
    _list,
    _mapping,
    _num,
    _str,
    parse_config,
)
from flylab.experiments.dataset import LoadedDataset, load_dataset
from flylab.experiments.runner import RunError, run_experiment
from flylab.experiments.stats import (
    DEFAULT_CONFIDENCE,
    DEFAULT_N_BOOT,
    derive_seed,
    describe,
    holm_adjust,
    paired_comparison,
)
from flylab.provenance.environment import git_state
from flylab.provenance.hashing import write_checksums
from flylab.provenance.labels import Evidence

GROUP_KIND = "flylab_replicate_group"
GROUP_SCHEMA_VERSION = 1
REPLICATE_SEED = "replicate"
METRICS: tuple[str, ...] = ("mean_rate_hz", "fraction_spiking", "latency_median_ms")
UNIT_OF_REPLICATION = (
    "one stimulus seed: one realisation of the Poisson input (and of the null graph where "
    "graph_null.seed = replicate) on one fixed connectome and parameter set. Seeds are not "
    "animals; their spread is the model's sensitivity to its own input noise."
)
#: Weakest first. A simulated value, read as a statement about the fly, is no stronger than the
#: weakest label among the inputs that produced it.
EVIDENCE_ORDER = (
    Evidence.UNKNOWN,
    Evidence.ASSUMED,
    Evidence.ENGINEERED,
    Evidence.LEARNED,
    Evidence.LITERATURE_DERIVED,
    Evidence.EMPIRICAL,
)


# --- config ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    patch: Mapping[str, Any]
    description: str | None = None


@dataclass(frozen=True)
class GroupStatsConfig:
    n_boot: int = DEFAULT_N_BOOT
    confidence: float = DEFAULT_CONFIDENCE
    seed: int = 0
    psth: bool = False
    populations: tuple[str, ...] | None = None  # None: every population recorded by all members
    # (population, reference population): paired by seed within the same runs of a condition
    population_contrasts: tuple[tuple[str, str], ...] = ()
    metrics: tuple[str, ...] = METRICS  # metrics described, compared and tested

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "n_boot": self.n_boot,
            "confidence": self.confidence,
            "seed": self.seed,
            "psth": self.psth,
        }
        if self.populations is not None:
            d["populations"] = list(self.populations)
        if self.metrics != METRICS:
            d["metrics"] = list(self.metrics)
        if self.population_contrasts:
            d["population_contrasts"] = [
                {"population": a, "reference": b} for a, b in self.population_contrasts
            ]
        return d


@dataclass(frozen=True)
class GroupConfig:
    group_id: str
    base_config: str
    base_text: str  # the base config file as read (its hash goes into the manifest)
    base_patch: Mapping[str, Any]
    seeds: tuple[int, ...]
    conditions: tuple[Condition, ...]
    comparisons: tuple[tuple[str, str], ...]  # (condition, reference)
    statistics: GroupStatsConfig = field(default_factory=GroupStatsConfig)
    description: str | None = None

    def condition(self, name: str) -> Condition:
        for c in self.conditions:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "schema_version": GROUP_SCHEMA_VERSION,
            "kind": GROUP_KIND,
            "group_id": self.group_id,
        }
        if self.description is not None:
            d["description"] = self.description
        d |= {
            "base_config": self.base_config,
            "base_config_sha256": hashlib.sha256(self.base_text.encode()).hexdigest(),
            "base_patch": copy.deepcopy(dict(self.base_patch)),
            "seeds": list(self.seeds),
            "conditions": [
                {"name": c.name, "patch": copy.deepcopy(dict(c.patch))}
                | ({"description": c.description} if c.description else {})
                for c in self.conditions
            ],
            "comparisons": [{"condition": c, "reference": r} for c, r in self.comparisons],
            "statistics": self.statistics.to_dict(),
        }
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively, other values replace, `None` deletes. Inputs are not changed."""
    out = copy.deepcopy(dict(base))
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_seeds(obj: object) -> tuple[int, ...]:
    p = "seeds"
    if isinstance(obj, dict):
        d = _mapping(obj, p)
        _keys(d, p, ("first", "count"), ())
        first = _int(d["first"], f"{p}.first", lo=0)
        count = _int(d["count"], f"{p}.count", lo=1, hi=10_000)
        seeds = tuple(range(first, first + count))
    else:
        seeds = tuple(_int(s, f"{p}[{i}]", lo=0, hi=2**63 - 1) for i, s in enumerate(_list(obj, p)))
    if not seeds:
        raise ConfigError(f"{p}: needs at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ConfigError(f"{p}: contains duplicates")
    return seeds


def _parse_conditions(obj: object, sweep: object) -> tuple[Condition, ...]:
    out: list[Condition] = []
    for i, spec in enumerate(_list(obj or [], "conditions")):
        q = f"conditions[{i}]"
        s = _mapping(spec, q)
        _keys(s, q, ("name",), ("patch", "description"))
        desc = s.get("description")
        out.append(
            Condition(
                name=_str(s["name"], f"{q}.name", _NAME),
                patch=_mapping(s.get("patch") or {}, f"{q}.patch"),
                description=None if desc is None else _str(desc, f"{q}.description"),
            )
        )
    if sweep is not None:
        axes = []
        for i, ax in enumerate(_list(sweep, "sweep")):
            q = f"sweep[{i}]"
            a = _mapping(ax, q)
            _keys(a, q, ("axis", "levels"), ())
            _str(a["axis"], f"{q}.axis", _NAME)
            levels = []
            for j, lv in enumerate(_list(a["levels"], f"{q}.levels")):
                lq = f"{q}.levels[{j}]"
                ld = _mapping(lv, lq)
                _keys(ld, lq, ("label", "patch"), ())
                levels.append((_str(ld["label"], f"{lq}.label", _NAME), _mapping(ld["patch"], lq)))
            if not levels:
                raise ConfigError(f"{q}.levels: must not be empty")
            axes.append(levels)
        for combo in itertools.product(*axes):
            patch: dict[str, Any] = {}
            for _, p in combo:
                patch = deep_merge(patch, p)
            out.append(Condition(name="_".join(lbl for lbl, _ in combo), patch=patch))
    if not out:
        raise ConfigError("conditions: define at least one condition (or a sweep)")
    names = [c.name for c in out]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"conditions: duplicate names {dupes}")
    return tuple(out)


def _parse_statistics(obj: object) -> GroupStatsConfig:
    p = "statistics"
    d = _mapping(obj or {}, p)
    _keys(
        d,
        p,
        (),
        ("n_boot", "confidence", "seed", "psth", "populations", "population_contrasts", "metrics"),
    )
    metrics: tuple[str, ...] = METRICS
    if "metrics" in d:
        metrics = tuple(
            _str(x, f"{p}.metrics[{i}]") for i, x in enumerate(_list(d["metrics"], f"{p}.metrics"))
        )
        bad = [m for m in metrics if m not in METRICS]
        if bad or not metrics or len(set(metrics)) != len(metrics):
            raise ConfigError(f"{p}.metrics: must be distinct values from {list(METRICS)}")
    conf = _num(d.get("confidence", DEFAULT_CONFIDENCE), f"{p}.confidence", lo=0.0, lo_open=True)
    if conf >= 1.0:
        raise ConfigError(f"{p}.confidence: must be < 1")
    psth = d.get("psth", False)
    if not isinstance(psth, bool):
        raise ConfigError(f"{p}.psth: expected true or false")
    pops = None
    if "populations" in d:
        pops = tuple(
            _str(x, f"{p}.populations[{i}]")
            for i, x in enumerate(_list(d["populations"], f"{p}.populations"))
        )
    contrasts = []
    for i, spec in enumerate(
        _list(d.get("population_contrasts") or [], f"{p}.population_contrasts")
    ):
        q = f"{p}.population_contrasts[{i}]"
        c = _mapping(spec, q)
        _keys(c, q, ("population", "reference"), ())
        a, b = _str(c["population"], f"{q}.population"), _str(c["reference"], f"{q}.reference")
        if a == b:
            raise ConfigError(f"{q}: population and reference must differ")
        contrasts.append((a, b))
    return GroupStatsConfig(
        n_boot=_int(d.get("n_boot", DEFAULT_N_BOOT), f"{p}.n_boot", lo=100, hi=1_000_000),
        confidence=conf,
        seed=_int(d.get("seed", 0), f"{p}.seed", lo=0, hi=2**62),
        psth=psth,
        populations=pops,
        population_contrasts=tuple(contrasts),
        metrics=metrics,
    )


def parse_group(obj: object, base_text: str | None = None) -> GroupConfig:
    """Validate a group document. `base_text` overrides reading `base_config` (tests)."""
    d = _mapping(obj, "group")
    _keys(
        d,
        "group",
        ("schema_version", "kind", "group_id", "base_config", "seeds"),
        ("description", "base_patch", "conditions", "sweep", "comparisons", "statistics",
         "base_config_sha256"),
    )  # fmt: skip
    if _int(d["schema_version"], "schema_version") != GROUP_SCHEMA_VERSION:
        raise ConfigError(f"schema_version: replicate groups are version {GROUP_SCHEMA_VERSION}")
    if d["kind"] != GROUP_KIND:
        raise ConfigError(f"kind: expected {GROUP_KIND!r}")
    base = _str(d["base_config"], "base_config")
    if base_text is None:
        try:
            base_text = Path(base).read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"base_config: cannot read {base}: {exc}") from exc
    if "base_config_sha256" in d:  # a frozen group.yaml: the base file must not have changed
        sha = hashlib.sha256(base_text.encode()).hexdigest()
        if d["base_config_sha256"] != sha:
            raise ConfigError(f"base_config: {base} changed since this group was frozen")
    conditions = _parse_conditions(d.get("conditions"), d.get("sweep"))
    names = {c.name for c in conditions}
    comps = []
    for i, spec in enumerate(_list(d.get("comparisons") or [], "comparisons")):
        q = f"comparisons[{i}]"
        s = _mapping(spec, q)
        _keys(s, q, ("condition", "reference"), ())
        c, r = _str(s["condition"], f"{q}.condition"), _str(s["reference"], f"{q}.reference")
        for k, v in (("condition", c), ("reference", r)):
            if v not in names:
                raise ConfigError(f"{q}.{k}: {v!r} is not a condition")
        if c == r:
            raise ConfigError(f"{q}: condition and reference must differ")
        comps.append((c, r))
    desc = d.get("description")
    return GroupConfig(
        group_id=_str(d["group_id"], "group_id", _NAME),
        description=None if desc is None else _str(desc, "description"),
        base_config=base,
        base_text=base_text,
        base_patch=_mapping(d.get("base_patch") or {}, "base_patch"),
        seeds=_parse_seeds(d["seeds"]),
        conditions=conditions,
        comparisons=tuple(comps),
        statistics=_parse_statistics(d.get("statistics")),
    )


def load_group(path: str | Path) -> GroupConfig:
    p = Path(path)
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read group config {p}: {exc}") from exc
    return parse_group(obj)


# --- members -----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Member:
    condition: str
    seed: int
    run_id: str
    config: ExperimentConfig

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self.config.to_yaml().encode()).hexdigest()


def member_config(group: GroupConfig, condition: Condition, seed: int) -> ExperimentConfig:
    try:
        base = yaml.safe_load(group.base_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"base_config {group.base_config}: not valid YAML: {exc}") from exc
    raw = deep_merge(deep_merge(_mapping(base, "base_config"), group.base_patch), condition.patch)
    q = f"condition {condition.name!r}"
    if "sensitivity" in raw:
        raise ConfigError(
            f"{q}: `sensitivity` is not allowed in replicate members (make each variant a "
            "condition; add `sensitivity: null` to base_patch)"
        )
    raw["seed"] = seed
    exp_id = f"{group.group_id}.{condition.name}"
    if len(exp_id) > 100:
        raise ConfigError(f"{q}: experiment id {exp_id!r} is longer than 100 characters")
    raw["experiment_id"] = exp_id
    gn = raw.get("graph_null")
    if isinstance(gn, dict) and gn.get("seed") == REPLICATE_SEED:
        raw["graph_null"] = {**gn, "seed": seed}
    try:
        return parse_config(raw)
    except ConfigError as exc:
        raise ConfigError(f"{q}: {exc}") from exc


def expand_members(group: GroupConfig) -> list[Member]:
    return [
        Member(c.name, s, f"{c.name}--s{s}", member_config(group, c, s))
        for c in group.conditions
        for s in group.seeds
    ]


# --- running -----------------------------------------------------------------------------------

_WORK: dict[str, Any] = {}  # set by run_group before forking workers; read by _run_member


def _run_member(i: int) -> tuple[int, str | None]:
    m: Member = _WORK["members"][i]
    data: LoadedDataset = _WORK["datasets"][m.config.dataset]
    try:
        run_experiment(
            m.config,
            out_root=_WORK["members_dir"],
            run_id=m.run_id,
            progress=None,
            data=data,
            code_state=_WORK["code"],
        )
    except Exception:  # reported per member; the group fails at the end
        return i, traceback.format_exc()
    return i, None


def _validate(members: Sequence[Member], datasets: Mapping[DatasetConfig, LoadedDataset]) -> None:
    """Data-dependent checks (populations, transmitter column, sign table, removals) once per
    distinct combination of the sections they depend on."""
    seen: set[str] = set()
    for m in members:
        c = m.config.to_dict()
        key = json.dumps(
            [c["dataset"], c["populations"], c["synapse_sign"], c["interventions"]],
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        try:
            run_experiment(
                m.config, progress=None, validate_only=True, data=datasets[m.config.dataset]
            )
        except ConfigError as exc:
            raise ConfigError(f"condition {m.condition!r}: {exc}") from exc


def _completed(member: Member, members_dir: Path) -> bool:
    d = members_dir / member.run_id
    if not d.is_dir():
        return False
    try:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"{d}: unreadable manifest ({exc}); remove the directory to rerun") from exc
    if manifest.get("config_sha256") != member.config_sha256:
        raise RunError(f"{d}: was run with a different config; refusing to reuse it")
    return True


@dataclass(frozen=True)
class GroupResult:
    group_dir: Path
    n_members: int
    n_run: int
    n_reused: int
    stats: dict[str, object]


def run_group(
    group: GroupConfig,
    out_root: str | Path = "runs",
    jobs: int = 1,
    resume: bool = False,
    progress: Callable[[str], None] | None = print,
) -> GroupResult:
    """Run every (condition, seed) member, then write group statistics."""
    if jobs < 1:
        raise RunError("jobs must be >= 1")
    code = git_state()  # the code every member runs (workers fork from this process)
    members = expand_members(group)
    datasets: dict[DatasetConfig, LoadedDataset] = {}
    for m in members:
        if m.config.dataset not in datasets:
            datasets[m.config.dataset] = load_dataset(m.config.dataset)
    _validate(members, datasets)

    group_dir = Path(out_root) / group.group_id
    members_dir = group_dir / "members"
    frozen = group.to_yaml()
    if group_dir.exists():
        if not resume:
            raise RunError(f"{group_dir} exists; pass resume to continue it")
        old = (group_dir / "group.yaml").read_text(encoding="utf-8")
        if old != frozen:
            raise RunError(f"{group_dir}/group.yaml differs from this group config")
        for partial in members_dir.glob(".*.partial"):
            if progress:
                progress(f"removing interrupted member {partial.name}")
            shutil.rmtree(partial)
    else:
        members_dir.mkdir(parents=True)
        (group_dir / "group.yaml").write_text(frozen, encoding="utf-8")

    todo = [i for i, m in enumerate(members) if not _completed(m, members_dir)]
    n_reused = len(members) - len(todo)
    if progress:
        progress(f"group {group.group_id}: {len(members)} members "
                 f"({len(group.conditions)} conditions x {len(group.seeds)} seeds), "
                 f"{n_reused} already complete, {len(todo)} to run, jobs={jobs}")  # fmt: skip
    _WORK.update(members=members, datasets=datasets, members_dir=members_dir, code=code)
    errors: dict[int, str] = {}
    done = 0
    try:
        if jobs == 1 or len(todo) <= 1:
            results = map(_run_member, todo)
            _drain(results, members, errors, len(todo), progress)
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RunError("jobs > 1 needs the 'fork' start method (Linux/macOS)")
            # fork: workers share the loaded graph copy-on-write; one member per worker process
            # so each run's memory is returned to the OS when it ends.
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=jobs, maxtasksperchild=1) as pool:
                _drain(pool.imap_unordered(_run_member, todo), members, errors, len(todo),
                       progress)  # fmt: skip
        done = len(todo) - len(errors)
    finally:
        _WORK.clear()
    if errors:
        for i, tb in sorted(errors.items()):
            if progress:
                progress(f"member {members[i].run_id} FAILED:\n{tb}")
        raise RunError(f"{len(errors)} of {len(todo)} member run(s) failed; fix and resume")
    stats = write_group_outputs(group, group_dir, members, code)
    return GroupResult(group_dir, len(members), done, n_reused, stats)


def _drain(
    results: Any,
    members: Sequence[Member],
    errors: dict[int, str],
    total: int,
    progress: Callable[[str], None] | None,
) -> None:
    for k, (i, err) in enumerate(results, 1):
        if err is not None:
            errors[i] = err
        if progress:
            status = "FAILED" if err else "ok"
            progress(f"[{k}/{total}] {members[i].run_id}: {status}")


# --- statistics --------------------------------------------------------------------------------


def _metric(pop_metrics: Mapping[str, Any], metric: str) -> float | None:
    if metric == "latency_median_ms":
        v = pop_metrics["first_spike_latency_ms"]["median"]
        return None if v is None else float(v)
    return float(pop_metrics[metric])


def _evidence_inputs(manifest: Mapping[str, Any]) -> dict[str, str]:
    out = {f"model.{k}": str(v["evidence"]) for k, v in manifest["model"]["parameters"].items()}
    out["time.dt"] = str(manifest["time"]["dt"]["evidence"])
    sign = manifest["synapse_sign"]
    if "mapping_evidence" in sign:
        out["synapse_sign.mapping"] = str(sign["mapping_evidence"])
    if sign.get("unknown_sign_policy") in ("as_excitatory", "as_inhibitory"):
        out["synapse_sign.unknown_sign_policy"] = Evidence.ASSUMED.value
    out["stimuli"] = Evidence.ENGINEERED.value
    if manifest.get("graph_null"):
        out["graph_null"] = Evidence.ENGINEERED.value
    return out


def weakest(labels: Sequence[str]) -> str:
    return min((Evidence(x) for x in labels), key=EVIDENCE_ORDER.index).value


def _psth(run_dir: Path, populations: Sequence[str]) -> tuple[float, dict[str, list[float]]]:
    """(bin width in ms, per-population rate per bin) from a member's population_rates.parquet."""
    t = pq.read_table(
        run_dir / "population_rates.parquet",
        columns=["bin", "t_start_ms", "t_end_ms", "population", "rate_hz"],
    )
    pop = np.asarray(t["population"].to_pylist(), dtype=object)
    b = t["bin"].to_numpy()
    r = t["rate_hz"].to_numpy()
    width = float(t["t_end_ms"][0].as_py() - t["t_start_ms"][0].as_py()) if t.num_rows else 0.0
    out = {}
    for p in populations:
        sel = pop == p
        order = np.argsort(b[sel])
        out[p] = [float(x) for x in r[sel][order]]
    return width, out


def group_statistics(group: GroupConfig, members_dir: Path) -> dict[str, object]:
    """Describe every condition and every paired comparison from the member run directories."""
    st = group.statistics
    metrics_of: dict[tuple[str, int], dict[str, Any]] = {}
    manifests: dict[tuple[str, int], dict[str, Any]] = {}
    for c in group.conditions:
        for s in group.seeds:
            d = members_dir / f"{c.name}--s{s}"
            metrics_of[(c.name, s)] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            manifests[(c.name, s)] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    recorded = [set(m["populations"]) for m in metrics_of.values()]
    common = sorted(set.intersection(*recorded))
    pops = list(st.populations) if st.populations is not None else common
    missing = [p for p in pops if p not in common]
    if missing:
        raise ConfigError(f"statistics.populations: {missing} not recorded by every member")
    missing = [x for ab in st.population_contrasts for x in ab if x not in pops]
    if missing:
        raise ConfigError(f"statistics.population_contrasts: {missing} not in the statistics")

    inputs: dict[str, set[str]] = {}
    for man in manifests.values():
        for k, v in _evidence_inputs(man).items():
            inputs.setdefault(k, set()).add(v)
    label = weakest([v for vs in inputs.values() for v in vs])

    conditions: dict[str, object] = {}
    values: dict[tuple[str, str, str], list[float | None]] = {}
    for c in group.conditions:
        per_pop: dict[str, object] = {}
        for p in pops:
            per_metric: dict[str, object] = {}
            for metric in st.metrics:
                vals = [_metric(metrics_of[(c.name, s)]["populations"][p], metric)
                        for s in group.seeds]  # fmt: skip
                values[(c.name, p, metric)] = vals
                present = [v for v in vals if v is not None]
                desc = describe(
                    present,
                    n_boot=st.n_boot,
                    confidence=st.confidence,
                    seed=derive_seed(st.seed, "describe", c.name, p, metric),
                )
                desc["n_missing"] = len(vals) - len(present)
                desc["values_by_seed"] = dict(zip(map(str, group.seeds), vals, strict=True))
                del desc["values"]
                per_metric[metric] = desc
            per_pop[p] = per_metric
        totals = [metrics_of[(c.name, s)]["totals"] for s in group.seeds]
        entry: dict[str, object] = {
            "description": c.description,
            "n_runs": len(group.seeds),
            "all_state_finite": all(bool(t["state_finite_at_end"]) for t in totals),
            "forced_spikes_dropped_by_seed": {
                str(s): int(t["n_forced_dropped_refractory_or_clamped"])
                for s, t in zip(group.seeds, totals, strict=True)
            },
            "populations": per_pop,
        }
        if st.psth:
            runs = [_psth(members_dir / f"{c.name}--s{s}", pops) for s in group.seeds]
            entry["psth_bin_ms"] = runs[0][0]
            entry["psth_mean_rate_hz"] = {
                p: [round(float(np.mean(col)), 6)
                    for col in zip(*(r[1][p] for r in runs), strict=True)]
                for p in pops
            }  # fmt: skip
        conditions[c.name] = entry

    def compare(xs: list[float | None], ys: list[float | None], *labels: str) -> dict[str, object]:
        pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if x is not None and y is not None]
        res: dict[str, object] = {"n_pairs_dropped_missing": len(xs) - len(pairs)}
        if not pairs:
            return res | {"n_pairs": 0}
        return res | paired_comparison(
            [x for x, _ in pairs],
            [y for _, y in pairs],
            n_boot=st.n_boot,
            confidence=st.confidence,
            seed=derive_seed(st.seed, *labels),
        )

    comparisons: list[dict[str, object]] = []
    for cond, ref in group.comparisons:
        for p in pops:
            for metric in st.metrics:
                res = compare(values[(cond, p, metric)], values[(ref, p, metric)],
                              "compare", cond, ref, p, metric)  # fmt: skip
                comparisons.append(
                    {"condition": cond, "reference": ref, "population": p, "metric": metric} | res
                )
    contrasts: list[dict[str, object]] = []
    for c in group.conditions:
        for a, b in st.population_contrasts:
            for metric in st.metrics:
                res = compare(values[(c.name, a, metric)], values[(c.name, b, metric)],
                              "contrast", c.name, a, b, metric)  # fmt: skip
                contrasts.append(
                    {"condition": c.name, "population": a, "reference_population": b,
                     "metric": metric} | res
                )  # fmt: skip
    pvals: list[float | None] = []
    family = comparisons + contrasts
    for comp in family:
        test = comp.get("sign_flip_test")
        pv = test.get("p_value") if isinstance(test, dict) else None
        pvals.append(float(pv) if isinstance(pv, int | float) else None)
    for comp, adj in zip(family, holm_adjust(pvals), strict=True):
        comp["p_holm"] = adj

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_replicate_group_stats",
        "group_id": group.group_id,
        "group_config_sha256": hashlib.sha256(group.to_yaml().encode()).hexdigest(),
        "seeds": list(group.seeds),
        "unit_of_replication": UNIT_OF_REPLICATION,
        "units": {
            "mean_rate_hz": "Hz per neuron, whole run",
            "fraction_spiking": "fraction of the population with >= 1 spike",
            "latency_median_ms": "median over the "
            "population's neurons of the first spike at or after the earliest stimulus "
            "onset (ms); missing when no neuron spiked",
        },
        "evidence": {
            "label": label,
            "applies_to": "every simulated value, difference and interval in this file, read as "
            "a statement about the fly",
            "rationale": "a model output is no stronger than the weakest label of its inputs "
            f"(order weakest first: {[e.value for e in EVIDENCE_ORDER]})",
            "inputs": {k: sorted(v) for k, v in sorted(inputs.items())},
        },
        "statistics_method": {
            "evidence": Evidence.ENGINEERED.value,
            "ci_mean": "BCa bootstrap over replicates (percentile fallback if replicates are "
            "not finite); flylab.experiments.stats",
            "paired": "same seed in condition and reference; mean difference (BCa), Cohen's "
            "d_z (percentile), relative change mean(d)/mean(ref) (paired percentile), "
            "two-sided sign-flip permutation test (exact for <= 16 pairs)",
            "multiple_comparisons": {
                "method": "Holm step-down",
                "family": "every comparison and population contrast x metric in this file "
                "with a p-value",
                "m": sum(p is not None for p in pvals),
            },
            "n_boot": st.n_boot,
            "confidence": st.confidence,
            "seed": st.seed,
            "seed_rule": "per statistic: flylab.experiments.stats.derive_seed(seed, labels)",
        },
        "conditions": conditions,
        "comparisons": comparisons,
        "population_contrasts": contrasts,
        "caveats": [
            "Uniform LIF parameters and an ASSUMED transmitter-to-sign table; rates are model "
            "outputs, not predictions validated against recordings.",
            "No behaviour is simulated or claimed.",
        ],
    }


def write_group_outputs(
    group: GroupConfig,
    group_dir: Path,
    members: Sequence[Member],
    code_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """`code_state`: git state when the members' code was loaded (None: recomputed statistics,
    the members' own git_commit.txt files then say what ran)."""
    members_dir = group_dir / "members"
    stats = group_statistics(group, members_dir)
    git = git_state()
    member_commits = sorted(
        {
            (members_dir / m.run_id / "git_commit.txt").read_text(encoding="utf-8").split("\n")[0]
            for m in members
        }
    )
    write_json(
        group_dir / "group_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "kind": "flylab_replicate_group_manifest",
            "group_id": group.group_id,
            "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "flylab_version": __version__,
            "git": {"commit": git.get("commit"), "dirty": git.get("dirty")},
            "git_note": "state when these group files were written (statistics code)",
            "members_code_state": None if code_state is None else dict(code_state),
            "member_commits": member_commits,
            "base_config": group.base_config,
            "base_config_sha256": hashlib.sha256(group.base_text.encode()).hexdigest(),
            "group_config_sha256": hashlib.sha256(group.to_yaml().encode()).hexdigest(),
            "members": [
                {
                    "condition": m.condition,
                    "seed": m.seed,
                    "run_id": m.run_id,
                    "dir": f"members/{m.run_id}",
                    "config_sha256": m.config_sha256,
                }
                for m in members
            ],
            "host": "not recorded (spec 8.4)",
        },
    )
    write_json(group_dir / "group_stats.json", stats)
    files = [group_dir / n for n in ("group.yaml", "group_manifest.json", "group_stats.json")]
    write_checksums(files, group_dir / "checksums.sha256", root=group_dir)
    return stats


def recompute_statistics(group_dir: str | Path) -> dict[str, object]:
    """Re-read a finished group directory (its frozen group.yaml) and rewrite its statistics."""
    gd = Path(group_dir)
    group = parse_group(yaml.safe_load((gd / "group.yaml").read_text(encoding="utf-8")))
    members = expand_members(group)
    for m in members:
        if not _completed(m, gd / "members"):
            raise RunError(f"member {m.run_id} is missing; resume the group first")
    return write_group_outputs(group, gd, members)
