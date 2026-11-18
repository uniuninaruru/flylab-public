"""`flylab run`: validate a config, run it (plus baseline / sign variants), write artifacts.

Every variant is validated before the first run directory is created, so a bad population or
sensitivity variant never leaves half-written output behind.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import random
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from flylab import SCHEMA_VERSION, __version__
from flylab.data.graph import SIGN_UNKNOWN, DirectedGraph, UnknownSignError, UnknownSignPolicy
from flylab.data.inspect import peak_rss_bytes
from flylab.data.null_models import NullModelError, make_null_graph
from flylab.experiments.artifacts import (
    OUTPUT_DESCRIPTIONS,
    RunRecorder,
    write_interventions,
    write_json,
)
from flylab.experiments.config import (
    ALL_POPULATION,
    ConfigError,
    ExperimentConfig,
    load_config,
)
from flylab.experiments.dataset import LoadedDataset, load_dataset
from flylab.experiments.populations import resolve_populations
from flylab.experiments.schedule import (
    ExperimentSchedule,
    RemovedEdges,
    compile_silences,
    compile_stimuli,
    select_removed_edges,
)
from flylab.neural.engine import LIFEngineBase, NeuralStep, Recorder
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import Discretization, discretize, labeled_dt
from flylab.neural.signs import SignMappingError, SignSource, resolve_signs
from flylab.provenance.environment import capture_environment, git_state
from flylab.provenance.hashing import write_checksums

EQUATIONS = (
    "tau_m dv/dt = (v_rest - v) + g + e;  tau_syn dg/dt = -g;  spike if v > v_threshold; "
    "reset v = v_reset, g = 0; refractory t_ref; arrival g += w after `delay`"
)
LIMITATIONS = (
    "Uniform LIF parameters for every neuron; not measured for MaleCNS neurons.",
    "w_syn is ASSUMED (copied from a female-brain model's free parameter), not calibrated.",
    "Synapse signs come from an ASSUMED transmitter mapping; see synapse_sign in this manifest.",
    "Current-based synapses, one uniform delay, no gap junctions, neuromodulation or plasticity.",
    "A single run is not a statistical result; replicate seeds before comparing conditions.",
)
GRAPH_NULL_LIMITATION = (
    "graph_null is set: this run used a shuffled CONTROL graph, not the connectome "
    "(see manifest graph_null)."
)
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,150}$")


class RunError(RuntimeError):
    """The run could not start (e.g. output directory exists, backend unavailable)."""


RecorderFactory = Callable[
    [ExperimentConfig, LoadedDataset, dict[str, NDArray[np.int32]]], Recorder
]


@dataclass
class _ObservedRecorder:
    artifacts: RunRecorder
    observer: Recorder

    def record(self, step: NeuralStep) -> None:
        # The observer may cooperatively cancel before the artifact writer accepts this step.
        self.observer.record(step)
        self.artifacts.record(step)


@dataclass(frozen=True, eq=False)
class PreparedRun:
    """A validated variant. The synaptic network is built only when the variant runs
    (`_build_network`), so at most one full-size network is alive at a time."""

    variant: str
    config: ExperimentConfig
    signed_graph: DirectedGraph
    sign_report: dict[str, object]
    removed: RemovedEdges
    disc: Discretization


@dataclass(frozen=True)
class RunResult:
    run_id: str
    variant: str
    run_dir: Path
    metrics: dict[str, object]
    rates_hz: dict[str, float]  # mean rate per recorded population


def _prepare(
    variant: str, cfg: ExperimentConfig, data: LoadedDataset, pops: dict[str, NDArray[np.int32]]
) -> PreparedRun:
    spec = cfg.synapse_sign
    values = None
    if spec.source is SignSource.TRANSMITTER:
        assert spec.transmitter_column is not None
        if spec.transmitter_column not in data.neurons.columns:
            raise ConfigError(
                f"synapse_sign.transmitter_column: {spec.transmitter_column!r} is not a "
                f"neuron-table column; available: {list(data.neurons.columns)}"
            )
        values = data.neurons.column(spec.transmitter_column)
    try:
        signed, report = resolve_signs(data.graph, spec, values)
    except SignMappingError as exc:
        raise ConfigError(f"synapse_sign ({variant}): {exc}") from exc
    removed = select_removed_edges(signed, cfg.interventions, pops)
    disc = discretize(cfg.model.parameters(), cfg.dt_ms)
    # Same check build_network makes (UnknownSignPolicy.ERROR refuses unknown signs), without
    # allocating the weights: kept edges whose sign is still unknown.
    n_unknown = int(((signed.sign == SIGN_UNKNOWN) & ~removed.mask).sum())
    if n_unknown and spec.unknown_sign_policy is UnknownSignPolicy.ERROR:
        raise ConfigError(
            f"synapse_sign.unknown_sign_policy ({variant}): {n_unknown} edge(s) have unknown "
            "sign; pick exclude / as_excitatory / as_inhibitory explicitly"
        )
    report["unknown_sign_edges_after_removal"] = n_unknown
    return PreparedRun(variant, cfg, signed, report, removed, disc)


def _build_network(prep: PreparedRun) -> SynapticNetwork:
    params = prep.config.model.parameters()
    try:
        return build_network(
            prep.signed_graph,
            prep.config.synapse_sign.unknown_sign_policy,
            params.w_syn.value,
            prep.removed.mask,
        )
    except UnknownSignError as exc:  # pragma: no cover - _prepare refuses this case first
        raise ConfigError(f"synapse_sign.unknown_sign_policy ({prep.variant}): {exc}") from exc


@contextlib.contextmanager
def _process_globals(cfg: ExperimentConfig) -> Iterator[None]:
    """Seed the legacy global RNGs (for third-party code; FlyLab uses explicit Generators) and
    set torch's determinism flag for one run, then restore whatever the process had before."""
    py_state = random.getstate()
    np_state = np.random.get_state()  # noqa: NPY002
    torch_mod = None
    torch_state: tuple[bool, object, object] | None = None
    if cfg.backend != "numpy":
        from flylab.neural.lif_torch import TorchUnavailableError, check_backend, require_torch

        try:
            check_backend(cfg.backend)
        except TorchUnavailableError as exc:
            raise RunError(str(exc)) from exc
        torch_mod = require_torch(cfg.backend)
        cuda_rng = torch_mod.cuda.get_rng_state_all() if cfg.backend == "torch-cuda" else None
        torch_state = (
            torch_mod.are_deterministic_algorithms_enabled(),
            torch_mod.get_rng_state(),
            cuda_rng,
        )
    random.seed(cfg.seed)
    np.random.seed(cfg.seed % 2**32)  # noqa: NPY002
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)  # noqa: NPY002
        if torch_mod is not None and torch_state is not None:
            torch_mod.use_deterministic_algorithms(torch_state[0])
            torch_mod.set_rng_state(torch_state[1])
            if torch_state[2] is not None:
                torch_mod.cuda.set_rng_state_all(torch_state[2])


def _engine(
    cfg: ExperimentConfig,
    prep: PreparedRun,
    net: SynapticNetwork,
    pops: dict[str, NDArray[np.int32]],
    max_edges_per_chunk: int | None = None,
) -> LIFEngineBase:
    recorded = {k: pops[k] for k in cfg.recorded_populations()}
    if cfg.backend == "numpy":
        return NumpyLIFEngine(net, prep.disc, cfg.dtype, recorded)
    from flylab.neural.lif_torch import (
        DEFAULT_EDGE_CHUNK,
        TORCH_BACKENDS,
        TorchLIFEngine,
        TorchUnavailableError,
        check_backend,
        require_torch,
    )

    try:
        check_backend(cfg.backend)
    except TorchUnavailableError as exc:
        raise RunError(str(exc)) from exc
    torch = require_torch(cfg.backend)
    torch.manual_seed(cfg.seed)  # seeds every device; the engines draw no random numbers
    torch.use_deterministic_algorithms(cfg.deterministic)
    if cfg.backend == "torch-cuda" and _use_fused(cfg, net):
        from flylab.neural.lif_fused import FusedCudaLIFEngine

        return FusedCudaLIFEngine(net, prep.disc, cfg.dtype, recorded)
    return TorchLIFEngine(
        net, prep.disc, cfg.dtype, recorded, device=TORCH_BACKENDS[cfg.backend],
        max_edges_per_chunk=max_edges_per_chunk or DEFAULT_EDGE_CHUNK,
    )  # fmt: skip


def _use_fused(cfg: ExperimentConfig, net: SynapticNetwork | None) -> bool:
    """Resolve `engine_impl` for `torch-cuda`. `fused` that cannot run is a RunError."""
    if cfg.engine_impl == "eager":
        return False
    from flylab.neural.lif_fused import fused_status, network_supports_fused

    status = fused_status()
    ok = bool(status["available"]) and (net is None or network_supports_fused(net))
    if cfg.engine_impl == "fused" and not ok:
        reason = status.get("reason") or "network weights are not integer synapse counts"
        raise RunError(f"engine_impl 'fused' cannot run here: {reason}")
    return ok


def _intervention_rows(
    prep: PreparedRun, pops: dict[str, NDArray[np.int32]]
) -> list[dict[str, object]]:
    cfg, g = prep.config, prep.signed_graph
    body = g.index.body_ids
    rows: list[dict[str, object]] = []
    for s in compile_stimuli(cfg.stimuli, pops, cfg.dt_ms):
        mode = None
        if s.kind == "forced_spikes":
            mode = "times" if s.times_steps is not None else "poisson"
        rows.append(
            {
                "category": "stimulus", "config_index": s.index, "kind": s.kind, "mode": mode,
                "population": s.population, "target": "neuron", "idx": s.idx,
                "body_id": body[s.idx], "onset_step": s.onset_step, "offset_step": s.offset_step,
                "amplitude_mV": s.amplitude_mV, "rate_hz": s.rate_hz,
            }
        )  # fmt: skip
    for c in compile_silences(cfg.interventions, pops, cfg.dt_ms):
        rows.append(
            {
                "category": "intervention", "config_index": c.index, "kind": "silence_neurons",
                "mode": c.mode, "population": c.population, "target": "neuron", "idx": c.idx,
                "body_id": body[c.idx], "onset_step": c.onset_step, "offset_step": c.offset_step,
            }
        )  # fmt: skip
    for k, eids in prep.removed.per_intervention:
        iv = cfg.interventions[k]
        rows.append(
            {
                "category": "intervention", "config_index": k, "kind": "remove_synapses",
                "mode": None if iv.fraction in (None, 1.0) else f"fraction={iv.fraction}",
                "population": None, "target": "edge", "idx": g.src[eids],
                "body_id": body[g.src[eids]],
                "dst_idx": g.dst[eids], "dst_body_id": body[g.dst[eids]],
                "synapse_count": g.synapse_count[eids], "onset_step": 0, "offset_step": cfg.steps,
            }
        )  # fmt: skip
    return [r for r in rows if np.asarray(r["idx"]).size]


def _run_id(cfg: ExperimentConfig, config_sha: str) -> str:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{cfg.experiment_id}-{stamp}-{config_sha[:8]}"


def _execute(
    prep: PreparedRun,
    run_id: str,
    run_dir: Path,
    data: LoadedDataset,
    pops: dict[str, NDArray[np.int32]],
    group: dict[str, object],
    baseline: RunResult | None,
    progress: Callable[[str], None] | None,
    code_state: Mapping[str, object],
    recorder_factory: RecorderFactory | None = None,
) -> RunResult:
    cfg = prep.config
    final_dir = run_dir
    # Written under a ".partial" name and renamed at the end, so an interrupted run can never be
    # mistaken for a complete one.
    run_dir = final_dir.with_name(f".{final_dir.name}.partial")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_text = cfg.to_yaml()
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    net = _build_network(prep)
    engine = _engine(cfg, prep, net, pops)
    engine.reset(cfg.seed)
    schedule = ExperimentSchedule(
        compile_stimuli(cfg.stimuli, pops, cfg.dt_ms),
        compile_silences(cfg.interventions, pops, cfg.dt_ms),
        cfg.seed,
    )
    recorded = {k: pops[k] for k in cfg.recorded_populations()}
    bin_steps = round(cfg.recording.population_rate_bin_ms / cfg.dt_ms)
    # Latency reference: earliest stimulus onset (on the dt grid, checked by the config parser).
    onsets = [s.onset_ms for s in cfg.stimuli]
    reference = min(onsets) if onsets else 0.0
    recorder = RunRecorder(
        run_dir,
        data.graph.index.body_ids,
        recorded,
        cfg.steps,
        bin_steps,
        cfg.recording.spike_events,
        reference_step=round(reference / cfg.dt_ms),
    )
    if progress:
        progress(f"[{prep.variant}] {cfg.backend} {cfg.dtype}: {cfg.steps} steps x {cfg.dt_ms} ms, "
                 f"{net.n_neurons:,} neurons, {net.n_edges:,} edges")  # fmt: skip
    engine.reset_peak_device_memory()
    observed: Recorder = recorder
    try:
        if recorder_factory is not None:
            observed = _ObservedRecorder(recorder, recorder_factory(cfg, data, pops))
        summary = engine.simulate(cfg.steps, observed, schedule)
    except BaseException:
        recorder.abort()
        raise
    device_memory = engine.device_memory_stats()
    recorder.close(run_dir, cfg.dt_ms)
    n_rows = write_interventions(run_dir / "interventions.parquet", _intervention_rows(prep, pops))

    pop_metrics = recorder.population_metrics(cfg.dt_ms, cfg.steps, reference)
    metrics: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "variant": prep.variant,
        "units": {"rate": "Hz per neuron", "time": "ms"},
        "totals": {
            "steps": cfg.steps,
            "simulated_ms": cfg.duration_ms,
            "n_spikes": summary.n_spikes,
            "n_threshold_spikes": recorder.n_threshold,
            "n_forced_spikes": summary.n_forced_spikes,
            "n_forced_dropped_refractory_or_clamped": summary.n_forced_dropped,
            "n_spikes_not_propagated": summary.n_not_propagated,
            "state_finite_at_end": summary.state_finite,
        },
        "populations": pop_metrics,
    }
    rates = recorder.mean_rates_hz(cfg.dt_ms, cfg.steps)
    if baseline is not None:
        metrics["vs_baseline"] = {
            "baseline_run_id": baseline.run_id,
            "note": "single seed, paired by construction; not a statistical test",
            "populations": {
                k: {
                    "rate_hz": r,
                    "baseline_rate_hz": baseline.rates_hz[k],
                    "difference_hz": r - baseline.rates_hz[k],
                }
                for k, r in rates.items()
                if k in baseline.rates_hz
            },
        }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "benchmark.json",
        {
            "schema_version": SCHEMA_VERSION,
            "note": "side observation of this run (one repeat, no warm-up); not a benchmark per "
            "docs/BENCHMARK_PROTOCOL.md",
            "measured": {
                "simulate_wall_time_s": summary.wall_time_s,
                "steps_per_s": cfg.steps / summary.wall_time_s if summary.wall_time_s > 0 else None,
                "process_peak_rss_bytes": peak_rss_bytes(),
                "engine_retained_bytes": engine.retained_bytes(),
                "gpu_max_memory_allocated_bytes": None
                if device_memory is None
                else device_memory["max_allocated_bytes"],
                "gpu_max_memory_reserved_bytes": None
                if device_memory is None
                else device_memory["max_reserved_bytes"],
            },
            "notes": {
                "process_peak_rss_bytes": "lifetime maximum of this process (ru_maxrss): includes "
                "dataset loading and any earlier variant of the same `flylab run`; not "
                "attributable to this run alone",
                "gpu_max_memory": "torch allocator peaks during simulate() on this device; null "
                "for host backends. Includes the edge arrays copied to the device beforehand",
            },
        },
    )
    (run_dir / "dataset_sha256.txt").write_text(
        "".join(f"{sha}  {path}\n" for path, sha in data.files), encoding="utf-8"
    )
    git = code_state
    (run_dir / "git_commit.txt").write_text(
        f"{git.get('commit') or 'unknown'}\ndirty={git.get('dirty')}\nsource={git.get('source')}\n",
        encoding="utf-8",
    )
    params = cfg.model.parameters()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_run",
        "run_id": run_id,
        "variant": prep.variant,
        "run_group": group,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "flylab_version": __version__,
        "experiment_id": cfg.experiment_id,
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "engine": engine.describe(),
        "seed": cfg.seed,
        "determinism": {
            "engine_rng": "none (inputs drawn by the schedule from SeedSequence([seed, k]))",
            "torch_deterministic_algorithms": cfg.deterministic if cfg.backend != "numpy" else None,
            "promise": "same config + same platform/versions -> same spikes; not promised "
            "across platforms or library versions",
        },
        "time": {
            "dt": labeled_dt(cfg.dt_ms).to_dict(),
            "steps": cfg.steps,
            "simulated_ms": cfg.duration_ms,
        },
        "graph_null": data.info.get("graph_null"),
        "dataset": {
            **{k: v for k, v in data.info.items() if k != "graph_null"},
            "files": [{"path": p, "sha256": s} for p, s in data.files],
            "graph_fingerprint_sha256": data.graph.fingerprint(),
            "n_neurons": data.graph.n_neurons,
            "n_edges": data.graph.n_edges,
            "total_synapses": data.graph.total_synapses,
        },
        "model": {
            "equations": EQUATIONS,
            "scheme": "docs/MODEL_ASSUMPTIONS.md section 2.1",
            "parameter_set": cfg.model.parameter_set,
            "parameters": params.to_dict(),
            "discretization": prep.disc.to_dict(),
        },
        "synapse_sign": prep.sign_report,
        "network": net.summary(),
        "populations": {
            p.name: {"selector": p.selector(), "n_neurons": int(pops[p.name].size)}
            for p in cfg.populations
        }
        | {
            ALL_POPULATION: {
                "selector": "every neuron",
                "n_neurons": int(pops[ALL_POPULATION].size),
            }
        },
        "stimuli": [{**s.to_dict(), "evidence": "ENGINEERED"} for s in cfg.stimuli],
        "interventions": [iv.to_dict() for iv in cfg.interventions],
        "interventions_parquet_rows": n_rows,
        "run_summary": summary.to_dict(),
        "outputs": OUTPUT_DESCRIPTIONS,
        "environment": _environment(code_state),
        "host": "not recorded (no host name, user or address; spec 8.4)",
        "limitations": list(LIMITATIONS)
        + ([GRAPH_NULL_LIMITATION] if cfg.graph_null is not None else []),
    }
    write_json(run_dir / "manifest.json", manifest)
    files = sorted(p for p in run_dir.iterdir() if p.is_file() and p.name != "checksums.sha256")
    write_checksums(files, run_dir / "checksums.sha256", root=run_dir)
    run_dir.rename(final_dir)
    run_dir = final_dir
    if progress:
        progress(f"[{prep.variant}] wrote {run_dir}: {summary.n_spikes:,} spikes, "
                 f"finite={summary.state_finite}")  # fmt: skip
    return RunResult(run_id, prep.variant, run_dir, metrics, rates)


def _environment(code_state: Mapping[str, object]) -> dict[str, object]:
    """Environment at write time, with `git` = the state captured when the run started (the code
    that ran). A different HEAD at write time is kept as `git_at_write` so it is never hidden."""
    env = capture_environment()
    at_write = env.get("git")
    env["git"] = dict(code_state)
    if at_write != env["git"]:
        env["git_at_write"] = at_write
        env["git_note"] = (
            "the repository changed while this run was executing; `git` is the state when the "
            "run's code was loaded (replicate groups: when the group started)"
        )
    return env


def apply_graph_null(
    cfg: ExperimentConfig, data: LoadedDataset, progress: Callable[[str], None] | None = None
) -> LoadedDataset:
    """The dataset with its graph replaced by the configured null graph (or unchanged)."""
    if cfg.graph_null is None:
        return data
    if progress:
        progress(f"graph_null: {cfg.graph_null.method} seed {cfg.graph_null.seed} "
                 f"on {data.graph.n_edges:,} edges")  # fmt: skip
    try:
        null, report = make_null_graph(data.graph, cfg.graph_null.method, cfg.graph_null.seed)
    except NullModelError as exc:
        raise RunError(f"graph_null: {exc}") from exc
    return replace(data, graph=null, info={**data.info, "graph_null": report.to_dict()})


def run_experiment(
    config: ExperimentConfig | str | Path,
    out_root: str | Path = "runs",
    run_id: str | None = None,
    with_baseline: bool = False,
    progress: Callable[[str], None] | None = print,
    validate_only: bool = False,
    data: LoadedDataset | None = None,
    code_state: Mapping[str, object] | None = None,
    recorder_factory: RecorderFactory | None = None,
) -> list[RunResult]:
    """Run the main config, its matched baseline (optional) and every sensitivity variant.

    `data` may pass a dataset already loaded from the same `dataset:` section (replicate groups
    load it once); it is refused if it came from a different section. `code_state` is the git
    state of the code that runs (default: captured now, before anything runs); replicate groups
    pass the state captured when the group started, because forked workers execute the code the
    group loaded, not whatever HEAD is when a member finishes.
    """
    code = dict(code_state) if code_state is not None else git_state()
    cfg = config if isinstance(config, ExperimentConfig) else load_config(config)
    if data is None:
        data = load_dataset(cfg.dataset)
    elif data.config != cfg.dataset:
        raise ConfigError(
            f"dataset: the preloaded dataset was loaded from {data.config!r}, "
            f"the config asks for {cfg.dataset!r}"
        )
    if not validate_only:  # the null keeps the node index, so validation can skip building it
        data = apply_graph_null(cfg, data, progress)
    pops = resolve_populations(cfg.populations, data)

    variants: list[tuple[str, ExperimentConfig]] = []
    if with_baseline:
        if not cfg.interventions:
            raise ConfigError("--with-baseline: the config has no interventions to compare against")
        variants.append(("baseline", cfg.without_interventions()))
    variants.append(("main", cfg))
    variants += [(f"sens-{v.name}", cfg.with_sign_variant(v)) for v in cfg.sensitivity]
    prepared = [_prepare(name, c, data, pops) for name, c in variants]
    if cfg.backend != "numpy":
        from flylab.neural.lif_torch import TorchUnavailableError, check_backend

        try:
            check_backend(cfg.backend)
        except TorchUnavailableError as exc:
            raise RunError(str(exc)) from exc
        if cfg.backend == "torch-cuda":
            _use_fused(cfg, None)  # an explicit `fused` that cannot run fails before any output
    if validate_only:
        return []

    config_sha = hashlib.sha256(cfg.to_yaml().encode()).hexdigest()
    base_id = run_id or _run_id(cfg, config_sha)
    if not _RUN_ID.match(base_id):
        raise RunError(f"run id {base_id!r} must match {_RUN_ID.pattern}")
    ids = {name: base_id if name == "main" else f"{base_id}--{name}" for name, _ in variants}
    out = Path(out_root)
    for rid in ids.values():
        for d in (out / rid, out / f".{rid}.partial"):
            if d.exists():
                raise RunError(f"{d} already exists; refusing to overwrite a run")
    group: dict[str, object] = {"group_id": base_id, "members": dict(ids)}
    results: list[RunResult] = []
    baseline: RunResult | None = None
    for prep in prepared:
        with _process_globals(prep.config):
            res = _execute(
                prep, ids[prep.variant], out / ids[prep.variant], data, pops, group,
                baseline if prep.variant == "main" else None, progress, code, recorder_factory,
            )  # fmt: skip
        if prep.variant == "baseline":
            baseline = res
        results.append(res)
    return results
