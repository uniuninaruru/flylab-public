"""Processed-graph directory: write, load and verify (spec section 6.2).

Layout written by `write_processed`:
    neurons.parquet    idx, body_id + source-tagged annotation columns (nullable)
    edges.parquet      src_idx, dst_idx, synapse_count, sign   (src = presynaptic)
    id_map.parquet     idx <-> body_id
    graph.npz          the same arrays for fast loading (uncompressed)
    manifest.json      provenance, rules, measured counts, column evidence labels
    checksums.sha256   SHA-256 of every file above
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from flylab import SCHEMA_VERSION
from flylab.data.graph import (
    ORIENTATION,
    SIGN_UNKNOWN,
    BuildReport,
    DirectedGraph,
    DuplicateRule,
    NeuronIndex,
)
from flylab.provenance.environment import capture_environment
from flylab.provenance.hashing import verify_checksums, write_checksums

OUTPUT_FILES = ("neurons.parquet", "edges.parquet", "id_map.parquet", "graph.npz", "manifest.json")

#: How `sign` is encoded on disk. Stored as Parquet field metadata and in graph.npz, because a
#: plain 0 is easy to misread as "zero weight" (`count * sign` would silently drop every edge).
SIGN_ENCODING = "int8: +1 excitatory, -1 inhibitory, 0 = UNKNOWN (not zero weight)"
EDGES_SCHEMA = pa.schema(
    [
        pa.field("src_idx", pa.int32()),
        pa.field("dst_idx", pa.int32()),
        pa.field("synapse_count", pa.int64()),
        pa.field("sign", pa.int8(), metadata={"flylab.sign_encoding": SIGN_ENCODING}),
    ]
)


def write_processed(
    out_dir: str | Path,
    graph: DirectedGraph,
    report: BuildReport,
    neuron_columns: Mapping[str, pa.Array],
    manifest_extra: Mapping[str, object],
) -> Path:
    """Write the processed directory. `neuron_columns` are aligned to `graph.index` order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = graph.n_neurons
    idx = pa.array(np.arange(n, dtype=np.int32))
    body = pa.array(graph.index.body_ids)

    for name, col in neuron_columns.items():
        if len(col) != n:
            raise ValueError(f"neuron column {name!r} has {len(col)} rows, graph has {n}")
    neurons = pa.table({"idx": idx, "body_id": body, **dict(neuron_columns)})
    pq.write_table(neurons, out / "neurons.parquet")
    pq.write_table(pa.table({"idx": idx, "body_id": body}), out / "id_map.parquet")
    edges = pa.Table.from_arrays(
        [
            pa.array(graph.src),
            pa.array(graph.dst),
            pa.array(graph.synapse_count),
            pa.array(graph.sign),
        ],
        schema=EDGES_SCHEMA,
    )
    pq.write_table(edges, out / "edges.parquet")
    np.savez(
        out / "graph.npz",
        body_ids=graph.index.body_ids,
        src=graph.src,
        dst=graph.dst,
        synapse_count=graph.synapse_count,
        sign=graph.sign,
        sign_encoding=np.array(SIGN_ENCODING),  # npz has no metadata slot; this is it
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "flylab_processed_graph",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "dataset_id": graph.dataset_id,
        "orientation": ORIENTATION,
        "duplicate_rule": graph.duplicate_rule.value,
        "graph_fingerprint_sha256": graph.fingerprint(),
        "counts": report.to_dict(),
        "environment": capture_environment(),
        **dict(manifest_extra),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    write_checksums([out / f for f in OUTPUT_FILES], out / "checksums.sha256", root=out)
    return out


def load_graph(processed_dir: str | Path) -> DirectedGraph:
    """Load graph.npz back into a DirectedGraph (no rebuild, no re-sorting)."""
    d = Path(processed_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    with np.load(d / "graph.npz") as z:
        arrays = {k: z[k] for k in ("body_ids", "src", "dst", "synapse_count", "sign")}
    for a in arrays.values():
        a.setflags(write=False)
    return DirectedGraph(
        index=NeuronIndex(arrays["body_ids"]),
        src=arrays["src"],
        dst=arrays["dst"],
        synapse_count=arrays["synapse_count"],
        sign=arrays["sign"],
        duplicate_rule=DuplicateRule(manifest["duplicate_rule"]),
        dataset_id=manifest["dataset_id"],
    )


def _first_mismatch(a: NDArray[Any], b: NDArray[Any]) -> int | None:
    diff = np.flatnonzero(a != b)
    return int(diff[0]) if diff.size else None


def _compare_edges_parquet(d: Path, g: DirectedGraph, problems: list[str]) -> None:
    """edges.parquet must equal graph.npz column by column (streamed by record batch)."""
    pf = pq.ParquetFile(d / "edges.parquet")
    if pf.metadata.num_rows != g.n_edges:
        problems.append(f"edges.parquet has {pf.metadata.num_rows} rows, graph.npz {g.n_edges}")
        return
    expected: dict[str, NDArray[Any]] = {
        "src_idx": g.src,
        "dst_idx": g.dst,
        "synapse_count": g.synapse_count,
        "sign": g.sign,
    }
    missing = [c for c in expected if c not in pf.schema_arrow.names]
    if missing:
        problems.append(f"edges.parquet lacks column(s) {missing}")
        return
    offset = 0
    for batch in pf.iter_batches(batch_size=1 << 20, columns=list(expected)):
        stop = offset + batch.num_rows
        for name, ref in expected.items():
            bad = _first_mismatch(batch.column(name).to_numpy(), ref[offset:stop])
            if bad is not None:
                problems.append(
                    f"edges.parquet column {name!r} differs from graph.npz at row {offset + bad}"
                )
                return
        offset = stop


def verify_processed(processed_dir: str | Path) -> dict[str, object]:
    """Check files, hashes and graph invariants. `ok` is True only if every check passes.

    Checksums only prove that files did not change after writing, so the Parquet tables are
    also compared with graph.npz content-wise (edge direction, index alignment).
    """
    d = Path(processed_dir)
    problems: list[str] = []
    missing = [f for f in (*OUTPUT_FILES, "checksums.sha256") if not (d / f).is_file()]
    if missing:
        return {"ok": False, "problems": [f"missing files: {missing}"]}

    listed = {
        ln.split(maxsplit=1)[1].strip()
        for ln in (d / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    if listed != set(OUTPUT_FILES):
        problems.append(
            f"checksums.sha256 must list exactly {sorted(OUTPUT_FILES)}; "
            f"missing {sorted(set(OUTPUT_FILES) - listed)}, "
            f"extra {sorted(listed - set(OUTPUT_FILES))}"
        )
    for m in verify_checksums(d / "checksums.sha256", d):
        problems.append(f"checksum mismatch: {m.path}")

    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    g = load_graph(d)
    n, e = g.n_neurons, g.n_edges
    ids = g.index.body_ids
    if n and not (np.diff(ids) > 0).all():
        problems.append("body_ids are not strictly increasing (index map not canonical)")
    if e:
        if int(g.src.min()) < 0 or int(g.dst.min()) < 0:
            problems.append("negative node index")
        if int(g.src.max()) >= n or int(g.dst.max()) >= n:
            problems.append("dangling edge: node index >= number of neurons")
        key = g.src.astype(np.int64) * max(n, 1) + g.dst.astype(np.int64)
        if not (np.diff(key) > 0).all():
            problems.append("edges not unique and sorted by (src, dst)")
        if int(g.synapse_count.min()) <= 0:
            problems.append("non-positive synapse_count")
    if not np.isin(g.sign, (-1, 0, 1)).all():
        problems.append("sign outside {-1, 0, +1}")
    if g.fingerprint() != manifest.get("graph_fingerprint_sha256"):
        problems.append("graph fingerprint differs from manifest")

    counts = manifest.get("counts", {})
    measured = {
        "n_neurons": n,
        "n_unique_edges": e,
        "total_synapses": g.total_synapses,
        "n_unknown_sign_edges": int((g.sign == SIGN_UNKNOWN).sum()),
        "n_self_loops": int((g.src == g.dst).sum()),
    }
    for k, v in measured.items():
        if counts.get(k) != v:
            problems.append(f"{k}: manifest {counts.get(k)} != measured {v}")

    arange = np.arange(n, dtype=np.int32)
    for table_name in ("id_map.parquet", "neurons.parquet"):
        t = pq.read_table(d / table_name, columns=["idx", "body_id"])
        if (
            t.num_rows != n
            or not np.array_equal(t["idx"].to_numpy(), arange)
            or not np.array_equal(t["body_id"].to_numpy(), ids)
        ):
            problems.append(f"{table_name} rows are not aligned with the graph.npz index")
    _compare_edges_parquet(d, g, problems)

    edge_filter = manifest.get("edge_filter", {})
    return {
        "ok": not problems,
        "problems": problems,
        "dataset_id": g.dataset_id,
        "node_set": manifest.get("node_set", {}).get("rule"),
        "measured": measured,
        "min_index": int(min(g.src.min(), g.dst.min())) if e else None,
        "max_index": int(max(g.src.max(), g.dst.max())) if e else None,
        # Spec 6.2: source file hashes and rows dropped for ids outside the node set.
        "source_files": [
            {k: i.get(k) for k in ("file", "sha256", "matches_registry", "matches_source_json")}
            for i in manifest.get("inputs", [])
        ],
        "dropped_rows": {
            k: edge_filter[k]
            for k in ("rows_total", "rows_pre_only", "rows_post_only", "rows_neither")
            if k in edge_filter
        },
    }
