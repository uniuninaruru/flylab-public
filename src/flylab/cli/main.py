"""`flylab` command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa

from flylab import SCHEMA_VERSION, __version__
from flylab.data import memory
from flylab.data.csv_io import FixtureFormatError, read_edges_csv, read_neurons_csv
from flylab.data.fetch import FetchError, fetch_release, release_dir
from flylab.data.graph import DuplicateRule, GraphBuildError, build_graph
from flylab.data.inspect import inspect_dir
from flylab.data.malecns import DEFAULT_NODE_SET, NODE_SETS, RawDataError, prepare_male_cns
from flylab.data.processed import verify_processed
from flylab.data.sources import MVP_FILES, RELEASES
from flylab.provenance.environment import capture_environment

EXIT_OK = 0
EXIT_DATA_ERROR = 3
EXIT_CONFIG_ERROR = 4
EXIT_RUN_ERROR = 5
EXIT_GATE_FAILED = 6


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False))


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"flylab {__version__} (schema {SCHEMA_VERSION})")
    return EXIT_OK


def _cmd_env(_: argparse.Namespace) -> int:
    _print_json(capture_environment())
    return EXIT_OK


def _cmd_sources(args: argparse.Namespace) -> int:
    rel = RELEASES[args.dataset]
    _print_json(
        {
            "dataset_id": rel.dataset_id,
            "release_date": rel.release_date,
            "license": rel.license_spdx,
            "attribution": rel.attribution,
            "verified_on": rel.verified_on,
            "verified_by": rel.verified_by,
            "evidence": rel.evidence.value,
            "files": [
                {
                    "name": f.name,
                    "url": rel.url(f.name),
                    "size_bytes": f.size_bytes,
                    "md5_base64": f.md5_base64,
                    "needed_for_mvp": f.name in MVP_FILES,
                    "role": f.role,
                }
                for f in rel.files
            ],
        }
    )
    return EXIT_OK


def _cmd_graph_inspect(args: argparse.Namespace) -> int:
    try:
        body_ids = read_neurons_csv(args.neurons)
        rows = read_edges_csv(args.edges)
        graph, report = build_graph(
            body_ids,
            rows.pre_body_id,
            rows.post_body_id,
            rows.synapse_count,
            rows.sign,
            dataset_id=args.dataset_id,
            duplicate_rule=DuplicateRule(args.duplicates),
        )
    except (FixtureFormatError, GraphBuildError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    out = {
        "schema_version": SCHEMA_VERSION,
        "report": report.to_dict(),
        "fingerprint_sha256": graph.fingerprint(),
        "memory_estimate": memory.estimate(graph.n_neurons, graph.n_edges).to_dict(),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    _print_json(out)
    return EXIT_OK


def _cmd_memory(args: argparse.Namespace) -> int:
    try:
        _print_json(memory.estimate(args.neurons, args.edges).to_dict())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    return EXIT_OK


def _cmd_data_fetch(args: argparse.Namespace) -> int:
    release = RELEASES[args.dataset]
    names = args.files or list(MVP_FILES)
    try:
        results = fetch_release(release, args.dest, names)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except (FetchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    _print_json(
        {
            "dataset_id": release.dataset_id,
            "dir": str(release_dir(args.dest, release)),
            "files": [{"name": r.name, "status": r.status, "sha256": r.sha256} for r in results],
        }
    )
    return EXIT_OK


def _cmd_data_inspect(args: argparse.Namespace) -> int:
    try:
        report = inspect_dir(args.raw)
    except (FileNotFoundError, OSError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    files = report["files"]
    assert isinstance(files, list)
    for f in files:
        cols = ", ".join(f"{k}:{v['type']}" for k, v in f["columns"].items())
        print(f"{f['file']}: {f['num_rows']:,} rows, {f['num_record_batches']} batches; {cols}")
    print(f"peak_rss_bytes={report['peak_rss_bytes']:,}  report={args.report}")
    return EXIT_OK


def _cmd_data_prepare(args: argparse.Namespace) -> int:
    try:
        out = prepare_male_cns(
            args.raw, args.out, args.node_set, count_all_bodies=args.count_all_bodies
        )
    except (FileNotFoundError, RawDataError, GraphBuildError, OSError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    print(f"processed={out}")
    return EXIT_OK


def _cmd_data_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_processed(args.processed)
    except (OSError, ValueError, KeyError, pa.ArrowException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    _print_json(report)
    return EXIT_OK if report["ok"] else EXIT_DATA_ERROR


def _cmd_run(args: argparse.Namespace) -> int:
    # Imported here so that `flylab data ...` never loads the neural stack.
    from flylab.experiments.config import ConfigError
    from flylab.experiments.dataset import DatasetError
    from flylab.experiments.runner import RunError, run_experiment

    try:
        results = run_experiment(
            args.config,
            out_root=args.out,
            run_id=args.run_id,
            with_baseline=args.with_baseline,
            validate_only=args.validate_only,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except DatasetError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except RunError as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR
    if args.validate_only:
        print(f"config OK: {args.config}")
        return EXIT_OK
    _print_json(
        {
            "runs": [
                {"variant": r.variant, "run_id": r.run_id, "dir": str(r.run_dir)} for r in results
            ]
        }
    )
    return EXIT_OK


def _cmd_replicate_run(args: argparse.Namespace) -> int:
    from flylab.experiments.config import ConfigError
    from flylab.experiments.dataset import DatasetError
    from flylab.experiments.replicates import load_group, run_group
    from flylab.experiments.runner import RunError

    try:
        group = load_group(args.config)
        if args.validate_only:
            from flylab.experiments.replicates import expand_members

            members = expand_members(group)
            print(f"group config OK: {len(members)} members "
                  f"({len(group.conditions)} conditions x {len(group.seeds)} seeds)")  # fmt: skip
            return EXIT_OK
        res = run_group(group, out_root=args.out, jobs=args.jobs, resume=args.resume)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except DatasetError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except RunError as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR
    _print_json(
        {
            "group_dir": str(res.group_dir),
            "members": res.n_members,
            "run_now": res.n_run,
            "reused": res.n_reused,
            "stats": str(res.group_dir / "group_stats.json"),
        }
    )
    return EXIT_OK


def _cmd_replicate_stats(args: argparse.Namespace) -> int:
    from flylab.experiments.config import ConfigError
    from flylab.experiments.replicates import recompute_statistics
    from flylab.experiments.runner import RunError

    try:
        recompute_statistics(args.group_dir)
    except (ConfigError, RunError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR
    print(f"wrote {Path(args.group_dir) / 'group_stats.json'}")
    return EXIT_OK


def _measure(args: argparse.Namespace, kind: str) -> int:
    """Shared body of `flylab bench` and `flylab parity`: run, write JSON, map errors."""
    from flylab.experiments.benchmark import BenchmarkError, run_benchmark, write_record
    from flylab.experiments.config import ConfigError
    from flylab.experiments.dataset import DatasetError
    from flylab.experiments.parity import run_parity
    from flylab.experiments.runner import RunError

    common = {"steps": args.steps, "backend": args.backend, "dtype": args.dtype, "tag": args.tag,
              "max_edges_per_chunk": args.edge_chunk, "engine_impl": args.engine_impl}  # fmt: skip
    try:
        if kind == "bench":
            record = run_benchmark(
                args.config, warmup=args.warmup, repeats=args.repeats, **common
            )  # fmt: skip
        else:
            record = run_parity(
                args.config, reference=args.reference, margin_mV=args.margin_mv,
                state_every=args.state_every, seed_control=not args.no_seed_control, **common,
            )  # fmt: skip
        path = write_record(record, args.out, kind)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except DatasetError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return EXIT_DATA_ERROR
    except (RunError, BenchmarkError) as exc:
        print(f"run error: {exc}", file=sys.stderr)
        return EXIT_RUN_ERROR
    summary = {"status": record["status"], "record": str(path)}
    if kind == "bench":
        summary |= {"gates": record["gates"], "stats": record["stats"]}
    else:
        lock = record["lockstep"]
        assert isinstance(lock, dict)
        keys = ("status", "steps", "steps_identical", "steps_compared", "resyncs",
                "max_abs_v_diff_mV", "max_abs_g_diff_mV", "divergence")  # fmt: skip
        summary |= {"lockstep": {k: lock[k] for k in keys}}
    _print_json(summary)
    return EXIT_OK if record["status"] == "PASS" else EXIT_GATE_FAILED


def _cmd_bench(args: argparse.Namespace) -> int:
    return _measure(args, "bench")


def _cmd_parity(args: argparse.Namespace) -> int:
    return _measure(args, "parity")


def _add_workload_args(sp: argparse.ArgumentParser, out_default: str) -> None:
    sp.add_argument("--config", required=True, type=Path, help="experiment YAML (schema v1)")
    sp.add_argument("--steps", type=int, help="override `steps` (validated like the config)")
    sp.add_argument("--backend", choices=["numpy", "torch-cpu", "torch-cuda"],
                    help="override `backend`")  # fmt: skip
    sp.add_argument("--dtype", choices=["float32", "float64"], help="override `dtype`")
    sp.add_argument("--tag", help="short label for the output file name, e.g. B1")
    sp.add_argument("--out", type=Path, default=Path(out_default), help="output directory")
    sp.add_argument("--edge-chunk", type=int, help="torch engines: max edges per arrival chunk")
    sp.add_argument("--engine-impl", choices=["auto", "eager", "fused"],
                    help="override `engine_impl` (torch-cuda only)")  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flylab",
        description="FlyLab tools: dataset import, LIF experiment runs, benchmarks.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="print version").set_defaults(func=_cmd_version)
    sub.add_parser("env", help="print host-anonymous environment JSON").set_defaults(func=_cmd_env)

    s = sub.add_parser("sources", help="list registered upstream dataset files (no download)")
    s.add_argument("--dataset", default="male-cns:v1.0", choices=sorted(RELEASES))
    s.set_defaults(func=_cmd_sources)

    g = sub.add_parser("graph", help="graph tools for synthetic CSV fixtures")
    gsub = g.add_subparsers(dest="graph_command", required=True)
    gi = gsub.add_parser("inspect", help="build and validate a graph from neurons/edges CSV")
    gi.add_argument("--neurons", required=True, type=Path)
    gi.add_argument("--edges", required=True, type=Path)
    gi.add_argument("--dataset-id", default="synthetic:csv")
    gi.add_argument("--duplicates", choices=[r.value for r in DuplicateRule], default="error")
    gi.add_argument("--report", type=Path, help="also write the JSON report here")
    gi.set_defaults(func=_cmd_graph_inspect)

    m = sub.add_parser("memory", help="sparse vs dense storage estimate (planning, not measured)")
    m.add_argument("--neurons", required=True, type=int)
    m.add_argument("--edges", required=True, type=int)
    m.set_defaults(func=_cmd_memory)

    d = sub.add_parser("data", help="dataset import: fetch, inspect, prepare, verify")
    dsub = d.add_subparsers(dest="data_command", required=True)
    df = dsub.add_parser("fetch", help="download registered upstream files with hash checks")
    df.add_argument("--dataset", default="male-cns:v1.0", choices=sorted(RELEASES))
    df.add_argument("--dest", type=Path, default=Path("data/raw"))
    df.add_argument(
        "--files",
        nargs="+",
        help=f"registered file names (default: the {len(MVP_FILES)} MVP files, ~1.1 GB)",
    )
    df.set_defaults(func=_cmd_data_fetch)

    di = dsub.add_parser("inspect", help="audit raw Feather files: schema, counts, value ranges")
    di.add_argument("--raw", type=Path, default=Path("data/raw/male-cns-v1.0"))
    di.add_argument("--report", type=Path, default=Path("artifacts/data_audit.json"))
    di.set_defaults(func=_cmd_data_inspect)

    dp = dsub.add_parser("prepare", help="build the processed graph from the raw MaleCNS files")
    dp.add_argument("--raw", type=Path, default=Path("data/raw/male-cns-v1.0"))
    dp.add_argument("--out", type=Path, default=Path("data/processed/male-cns-v1.0"))
    dp.add_argument(
        "--node-set",
        default=DEFAULT_NODE_SET,
        choices=sorted(NODE_SETS),
        help="named rule for which bodies are nodes (written to the manifest)",
    )
    dp.add_argument(
        "--count-all-bodies",
        action="store_true",
        help="also count every distinct segment id in the weights file (slow, several GiB)",
    )
    dp.set_defaults(func=_cmd_data_prepare)

    dv = dsub.add_parser("verify", help="check a processed graph: hashes, indices, counts")
    dv.add_argument(
        "--processed",
        type=Path,
        default=Path("data/processed/male-cns-v1.0") / DEFAULT_NODE_SET,
    )
    dv.set_defaults(func=_cmd_data_verify)

    r = sub.add_parser("run", help="validate and run an experiment config (LIF engines)")
    r.add_argument("--config", required=True, type=Path, help="experiment YAML (schema v1)")
    r.add_argument("--out", type=Path, default=Path("runs"), help="parent of the run directory")
    r.add_argument("--run-id", help="run directory name (default: <experiment_id>-<UTC>-<hash>)")
    r.add_argument(
        "--with-baseline",
        action="store_true",
        help="also run the matched baseline (same config without interventions)",
    )
    r.add_argument(
        "--validate-only", action="store_true", help="check config + data, write nothing"
    )
    r.set_defaults(func=_cmd_run)

    rg = sub.add_parser(
        "replicate", help="replicate groups: seeds x conditions with paired statistics"
    )
    rgsub = rg.add_subparsers(dest="replicate_command", required=True)
    rr = rgsub.add_parser("run", help="run (or resume) a replicate group and write statistics")
    rr.add_argument("--config", required=True, type=Path, help="group YAML (kind: "
                    "flylab_replicate_group)")  # fmt: skip
    rr.add_argument("--out", type=Path, default=Path("runs"), help="parent of the group dir")
    rr.add_argument("--jobs", type=int, default=1, help="parallel member runs (fork; default 1)")
    rr.add_argument("--resume", action="store_true", help="continue an existing group directory")
    rr.add_argument("--validate-only", action="store_true", help="check the group config only")
    rr.set_defaults(func=_cmd_replicate_run)
    rs = rgsub.add_parser("stats", help="recompute group_stats.json of a finished group")
    rs.add_argument("--group-dir", required=True, type=Path)
    rs.set_defaults(func=_cmd_replicate_stats)

    b = sub.add_parser(
        "bench", help="time a config: warm-up, repeats, synchronized clock, peak memory"
    )
    _add_workload_args(b, "artifacts/bench")
    b.add_argument("--warmup", type=int, default=3, help="untimed runs first (default 3)")
    b.add_argument("--repeats", type=int, default=10, help="timed runs (default 10)")
    b.set_defaults(func=_cmd_bench)

    q = sub.add_parser(
        "parity", help="run the config's backend against the NumPy reference in lockstep"
    )
    _add_workload_args(q, "artifacts/parity")
    q.add_argument("--reference", default="numpy", choices=["numpy", "torch-cpu"])
    q.add_argument("--margin-mv", type=float, help="state/threshold margin (default by dtype)")
    q.add_argument("--state-every", type=int, default=100, help="compare states every N steps")
    q.add_argument(
        "--no-seed-control",
        action="store_true",
        help="skip the extra reference run with the next seed after a divergence",
    )
    q.set_defaults(func=_cmd_parity)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
