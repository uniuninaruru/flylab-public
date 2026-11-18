"""MaleCNS v1.0 importer (`flylab data prepare`).

Inputs (official flat-connectome Feather files, schemas audited 2026-11-04):
    body-annotations-...: one row per annotated body; `bodyId`, `status`, `superclass`, `type`, ...
    body-neurotransmitters-...: one row per body; `body`, `predicted_nt`, `consensus_nt`, ...
    connectome-weights-...: `body_pre`, `body_post`, `weight` for every connected segment pair

The node set is a named rule (`NODE_SETS`), never an implicit filter. Edges are kept only when
both ends are in the node set; every dropped row is counted. No synapse sign is assigned here:
all edges get sign 0 (unknown). The transmitter -> sign mapping is a model decision for PR-02.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
from numpy.typing import NDArray

from flylab.data.fetch import hash_file_md5_sha256
from flylab.data.graph import DuplicateRule, NeuronIndex, build_graph_indexed
from flylab.data.inspect import peak_rss_bytes
from flylab.data.processed import write_processed
from flylab.data.sources import MALE_CNS_V1_0, DatasetRelease
from flylab.provenance.labels import Evidence

DATASET_ID = "male-cns:v1.0"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
INPUT_FILES = (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS)

# Photoreceptor type names as listed in HEREISCB/flybrain flybrain/connectome.py (MIT); used only
# by the `flybrain_compat` node set so that its 167,106-node graph can be reproduced and compared.
PHOTORECEPTOR_TYPES = (
    "R1-R6", "R7y", "R7p", "R7d", "R7_unclear", "R8y", "R8p", "R8d", "R8_unclear", "R7R8_unclear",
)  # fmt: skip

NODE_SETS: dict[str, str] = {
    "superclass_annotated": "bodies whose curated `superclass` annotation is non-null",
    "traced": "bodies with status == 'Traced'",
    "flybrain_compat": "status == 'Traced' or `type` is a photoreceptor type (HEREISCB/flybrain)",
}
DEFAULT_NODE_SET = "superclass_annotated"

ANNOTATION_COLUMNS = (
    "type", "instance", "superclass", "class", "subclass", "somaSide", "status", "statusLabel",
    "flywireType", "hemibrainType", "mancType", "dimorphism",
)  # fmt: skip
NT_COLUMNS = (
    "predicted_nt", "predicted_nt_confidence", "consensus_nt", "celltype_predicted_nt",
    "celltype_predicted_nt_confidence", "ground_truth", "total_nt_predictions",
)  # fmt: skip

Progress = Callable[[str], None] | None


class RawDataError(ValueError):
    """Raw files do not match what the importer expects."""


def _require(table: pa.Table, columns: tuple[str, ...], file: str) -> None:
    missing = [c for c in columns if c not in table.column_names]
    if missing:
        raise RawDataError(f"{file}: missing column(s) {missing}")


def _bool_mask(arr: pa.ChunkedArray) -> NDArray[np.bool_]:
    return np.asarray(pc.fill_null(arr, False).to_numpy(zero_copy_only=False), dtype=bool)


def node_set_masks(ann: pa.Table) -> dict[str, NDArray[np.bool_]]:
    """Boolean row mask over the annotation table for every rule in NODE_SETS."""
    _require(ann, ("bodyId", "status", "superclass", "type"), ANNOTATIONS)
    status = ann["status"]
    if pa.types.is_dictionary(status.type):
        status = status.cast(pa.string())
    traced = _bool_mask(pc.equal(status, "Traced"))
    photoreceptor = _bool_mask(pc.is_in(ann["type"], value_set=pa.array(PHOTORECEPTOR_TYPES)))
    return {
        "superclass_annotated": np.asarray(ann["superclass"].is_valid(), dtype=bool),
        "traced": traced,
        "flybrain_compat": traced | photoreceptor,
    }


def _lookup(
    sorted_ids: NDArray[np.int64], query: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """Positions of `query` in `sorted_ids`, and which queries were found."""
    if sorted_ids.size == 0:
        return np.zeros(query.size, dtype=np.int64), np.zeros(query.size, dtype=bool)
    pos = np.searchsorted(sorted_ids, query)
    found = sorted_ids[np.minimum(pos, sorted_ids.size - 1)] == query
    return pos, found


def stream_edges(
    weights_path: Path,
    node_ids: NDArray[np.int64],
    progress: Progress = None,
    count_all_bodies: bool = False,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64], dict[str, int | None]]:
    """Read the weights file batch by batch; keep rows whose two ends are both in `node_ids`.

    `count_all_bodies` also counts every distinct segment id in the file (88.4 M in v1.0). That
    needs a large running unique set (~21 min and several GiB on the 151.9 M-row file), so it is
    off by default; the per-node-set statistics are always computed cheaply.
    """
    srcs: list[NDArray[np.int32]] = []
    dsts: list[NDArray[np.int32]] = []
    counts: list[NDArray[np.int32]] = []
    stats = dict.fromkeys(
        ("rows_total", "synapses_total", "rows_kept", "synapses_kept", "rows_pre_only",
         "rows_post_only", "rows_neither"),
        0,
    )  # fmt: skip
    seen_parts: list[NDArray[np.int64]] = []
    seen: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
    node_seen = np.zeros(node_ids.size, dtype=bool)  # node-set bodies appearing in any row
    with pa.memory_map(str(weights_path), "r") as source:
        reader = pa.ipc.open_file(source)
        for name in ("body_pre", "body_post", "weight"):
            if reader.schema.get_field_index(name) < 0:
                raise RawDataError(f"{WEIGHTS}: missing column {name!r}")
        n_batches = reader.num_record_batches
        next_report = 0.0
        for i in range(n_batches):
            b = reader.get_batch(i)
            pre = b.column("body_pre").to_numpy().astype(np.int64, copy=False)
            post = b.column("body_post").to_numpy().astype(np.int64, copy=False)
            w = b.column("weight").to_numpy()
            pi, pin = _lookup(node_ids, pre)
            qi, qin = _lookup(node_ids, post)
            keep = pin & qin
            stats["rows_total"] += int(pre.size)
            stats["synapses_total"] += int(w.sum())
            stats["rows_kept"] += int(keep.sum())
            stats["synapses_kept"] += int(w[keep].sum())
            stats["rows_pre_only"] += int((pin & ~qin).sum())
            stats["rows_post_only"] += int((~pin & qin).sum())
            stats["rows_neither"] += int((~pin & ~qin).sum())
            srcs.append(pi[keep].astype(np.int32))
            dsts.append(qi[keep].astype(np.int32))
            counts.append(w[keep].astype(np.int32))
            node_seen[pi[pin]] = True
            node_seen[qi[qin]] = True
            if count_all_bodies:
                seen_parts.append(np.unique(np.concatenate([pre, post])))
                if len(seen_parts) >= 64:
                    seen = np.unique(np.concatenate([seen, *seen_parts]))
                    seen_parts = []
            if progress and n_batches and (i + 1) / n_batches >= next_report:
                progress(f"  weights: batch {i + 1}/{n_batches}, kept {stats['rows_kept']:,} rows")
                next_report += 0.1
    out_stats: dict[str, int | None] = dict(stats)
    if count_all_bodies:
        seen = np.unique(np.concatenate([seen, *seen_parts]))
        out_stats["bodies_in_weights"] = int(seen.size)
    else:
        out_stats["bodies_in_weights"] = None  # not measured (pass count_all_bodies=True)
    out_stats["node_set_bodies_in_weights"] = int(node_seen.sum())
    return (
        np.concatenate(srcs) if srcs else np.zeros(0, np.int32),
        np.concatenate(dsts) if dsts else np.zeros(0, np.int32),
        np.concatenate(counts).astype(np.int64) if counts else np.zeros(0, np.int64),
        out_stats,
    )


def _check_inputs(raw: Path, release: DatasetRelease | None) -> list[dict[str, object]]:
    """Hash every input; require size + MD5 to match the registered upstream object.

    The registry check does not depend on SOURCE.json, so a manual/offline import of other
    content under the official names is refused. SOURCE.json, when present, is checked too.
    `release=None` (synthetic test inputs only) skips the registry check.
    """
    source_path = raw / "SOURCE.json"
    recorded: dict[str, object] = {}
    if source_path.is_file():
        src = json.loads(source_path.read_text(encoding="utf-8"))
        recorded = {k: v.get("sha256") for k, v in src.get("files", {}).items()}
    out = []
    for name in INPUT_FILES:
        p = raw / name
        if not p.is_file():
            raise FileNotFoundError(f"missing input {p} (run `flylab data fetch` first)")
        md5_b64, digest, size = hash_file_md5_sha256(p)
        matches_registry: bool | None = None
        if release is not None:
            spec = release.file(name)
            if size != spec.size_bytes or md5_b64 != spec.md5_base64:
                raise RawDataError(
                    f"{name}: size {size} / MD5 {md5_b64} do not match the registered "
                    f"{release.dataset_id} object (size {spec.size_bytes} / MD5 "
                    f"{spec.md5_base64}); refusing to label it {release.dataset_id}"
                )
            matches_registry = True
        expected = recorded.get(name)
        if expected is not None and expected != digest:
            raise RawDataError(f"{name}: sha256 {digest} != SOURCE.json {expected}")
        out.append(
            {
                "file": name,
                "size_bytes": size,
                "sha256": digest,
                "md5_base64": md5_b64,
                "matches_registry": matches_registry,
                "matches_source_json": None if expected is None else True,
            }
        )
    return out


def prepare_male_cns(
    raw_dir: str | Path,
    out_dir: str | Path,
    node_set: str = DEFAULT_NODE_SET,
    progress: Progress = print,
    count_all_bodies: bool = False,
    release: DatasetRelease | None = MALE_CNS_V1_0,
    dataset_id: str | None = None,
) -> Path:
    """Build data/processed/<dataset>/<node_set>/ from the three raw files.

    By default the inputs must be the registered MaleCNS v1.0 objects. Tests on synthetic
    tables pass `release=None` and a `dataset_id` starting with "synthetic:"; the output is
    then labelled ENGINEERED, never EMPIRICAL.
    """
    if node_set not in NODE_SETS:
        raise ValueError(f"unknown node set {node_set!r}; choose from {sorted(NODE_SETS)}")
    if release is None:
        if not (dataset_id and dataset_id.startswith("synthetic:")):
            raise ValueError("release=None requires dataset_id='synthetic:...'")
        ds_id = dataset_id
    else:
        if dataset_id not in (None, release.dataset_id):
            raise ValueError(f"dataset_id {dataset_id!r} contradicts release {release.dataset_id}")
        ds_id = release.dataset_id
    raw = Path(raw_dir)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    inputs = _check_inputs(raw, release)
    timings["hash_inputs_s"] = time.perf_counter() - t0

    t = time.perf_counter()
    ann = feather.read_table(str(raw / ANNOTATIONS), memory_map=True)
    _require(ann, ("bodyId", *ANNOTATION_COLUMNS), ANNOTATIONS)
    body_all = ann["bodyId"].to_numpy()
    if np.unique(body_all).size != body_all.size:
        raise RawDataError(f"{ANNOTATIONS}: bodyId is not unique")
    masks = node_set_masks(ann)
    mask = masks[node_set]
    order = np.argsort(body_all[mask], kind="stable")
    rows = np.flatnonzero(mask)[order]  # annotation rows in index (sorted body id) order
    index = NeuronIndex.from_body_ids(body_all[rows])
    if progress:
        progress(f"node set {node_set!r}: {index.n:,} bodies ({NODE_SETS[node_set]})")
    timings["node_set_s"] = time.perf_counter() - t

    t = time.perf_counter()
    take = pa.array(rows)
    neuron_columns: dict[str, pa.Array] = {}
    for col in ANNOTATION_COLUMNS:
        arr = ann[col].take(take).combine_chunks()
        if pa.types.is_dictionary(arr.type):
            arr = arr.cast(arr.type.value_type)
        neuron_columns[col] = arr

    nt = feather.read_table(str(raw / NEUROTRANSMITTERS), columns=["body", *NT_COLUMNS])
    nt_body = nt["body"].to_numpy()
    nt_order = np.argsort(nt_body, kind="stable")
    nt_sorted = nt_body[nt_order]
    if nt_sorted.size and (np.diff(nt_sorted) == 0).any():
        raise RawDataError(f"{NEUROTRANSMITTERS}: body is not unique")
    pos, found = _lookup(nt_sorted, index.body_ids)
    nt_take = pa.array(np.where(found, nt_order[np.minimum(pos, max(nt_sorted.size - 1, 0))], 0),
                       mask=~found)  # fmt: skip
    for col in NT_COLUMNS:
        neuron_columns[col] = nt[col].take(nt_take).combine_chunks()
    timings["neuron_table_s"] = time.perf_counter() - t

    t = time.perf_counter()
    src, dst, counts, edge_stats = stream_edges(
        raw / WEIGHTS, index.body_ids, progress, count_all_bodies
    )
    timings["stream_weights_s"] = time.perf_counter() - t

    t = time.perf_counter()
    graph, report = build_graph_indexed(
        index, src, dst, counts, dataset_id=ds_id, duplicate_rule=DuplicateRule.ERROR
    )
    del src, dst, counts
    timings["build_graph_s"] = time.perf_counter() - t

    data_label = Evidence.EMPIRICAL if release is not None else Evidence.ENGINEERED
    synthetic_note = None if release is not None else "synthetic test input, not connectome data"
    evidence_ann = {"evidence": data_label.value, "source": ANNOTATIONS, "note": synthetic_note}
    evidence_nt = {
        "evidence": data_label.value,
        "source": NEUROTRANSMITTERS,
        "note": synthetic_note
        or "dataset-published neurotransmitter prediction, not ground truth; null if absent",
    }
    evidence_index = {
        "evidence": Evidence.ENGINEERED.value,
        "source": "flylab index convention: idx = position of body_id in ascending order; "
        "src_idx = presynaptic, dst_idx = postsynaptic (flylab.data.graph.ORIENTATION)",
    }
    manifest_extra: dict[str, object] = {
        "release": None
        if release is None
        else {
            "dataset_id": release.dataset_id,
            "release_date": release.release_date,
            "license": release.license_spdx,
            "registry_verified_on": release.verified_on,
        },
        "inputs": inputs,
        "node_set": {
            "rule": node_set,
            "description": NODE_SETS[node_set],
            "n_selected": index.n,
            "all_rules_measured": {k: int(m.sum()) for k, m in masks.items()},
            "annotation_rows": int(body_all.size),
        },
        "edge_filter": {
            "rule": "keep a weights row only if body_pre and body_post are both in the node set",
            **edge_stats,
        },
        "sign_policy": (
            "not assigned: every edge has sign 0 (unknown); see docs/MODEL_ASSUMPTIONS.md"
        ),
        "columns": {
            "idx": evidence_index,
            "src_idx": evidence_index,
            "dst_idx": evidence_index,
            "body_id": evidence_ann,
            **{c: evidence_ann for c in ANNOTATION_COLUMNS},
            **{c: evidence_nt for c in NT_COLUMNS},
            "ground_truth": {
                **evidence_nt,
                "note": synthetic_note or "dataset-provided ground-truth transmitter",
            },
            "synapse_count": {
                "evidence": data_label.value,
                "source": f"{WEIGHTS} column `weight` (synapse confidence >= 0.5)",
                "note": synthetic_note,
            },
            "sign": {
                "evidence": Evidence.UNKNOWN.value,
                "source": None,
                "note": "int8 on disk: 0 = UNKNOWN (not zero weight); see sign field metadata",
            },
        },
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "peak_rss_bytes": peak_rss_bytes(),
    }
    out = Path(out_dir) / node_set
    write_processed(out, graph, report, neuron_columns, manifest_extra)
    if progress:
        progress(
            f"wrote {out}: {report.n_neurons:,} neurons, {report.n_unique_edges:,} edges, "
            f"{report.total_synapses:,} synapses; dropped rows: pre-only "
            f"{edge_stats['rows_pre_only']:,}, post-only {edge_stats['rows_post_only']:,}, "
            f"neither {edge_stats['rows_neither']:,}"
        )
    return out
