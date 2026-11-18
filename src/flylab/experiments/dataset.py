"""Load the graph and neuron table an experiment runs on, with file hashes for provenance.

Three sources (config `dataset.kind`):
- `processed`: a PR-01 processed directory (`graph.npz`, `neurons.parquet`, ...). Read only;
  file hashes are checked against the directory's own `checksums.sha256`.
- `csv_fixture`: synthetic CSV fixtures (tests/fixtures). Not biological data.
- `synthetic_random`: `flylab.data.synthetic.random_graph`. Not biological data.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from flylab.data.csv_io import read_edges_csv
from flylab.data.graph import DirectedGraph, DuplicateRule, GraphBuildError, build_graph
from flylab.data.processed import load_graph
from flylab.data.synthetic import random_graph
from flylab.experiments.config import DatasetConfig
from flylab.provenance.hashing import sha256_file, verify_checksums

PROCESSED_FILES_READ = ("graph.npz", "neurons.parquet", "manifest.json")


class DatasetError(ValueError):
    """The dataset is missing, inconsistent or does not match its recorded checksums."""


@dataclass(frozen=True, eq=False)
class NeuronTable:
    """Per-neuron annotations aligned to the graph index. Columns load lazily."""

    body_ids: NDArray[np.int64]
    columns: tuple[str, ...]
    _loader: Callable[[str], NDArray[np.object_]]
    _cache: dict[str, NDArray[np.object_]] = field(default_factory=dict)

    def column(self, name: str) -> NDArray[np.object_]:
        """Values as Python objects (str or None for null), length N, index order."""
        if name not in self.columns:
            raise KeyError(name)
        if name not in self._cache:
            col = self._loader(name)
            if col.shape != self.body_ids.shape:
                raise DatasetError(f"neuron column {name!r} has {col.size} rows")
            self._cache[name] = col
        return self._cache[name]


@dataclass(frozen=True, eq=False)
class LoadedDataset:
    graph: DirectedGraph
    neurons: NeuronTable
    files: tuple[tuple[str, str], ...]  # (path as given in the config, sha256)
    info: dict[str, object]
    config: DatasetConfig | None = None  # the config section this was loaded from


def _objects(arr: pa.ChunkedArray | pa.Array) -> NDArray[np.object_]:
    out = np.empty(len(arr), dtype=object)
    out[:] = arr.to_pylist()
    return out


def _load_processed(cfg: DatasetConfig) -> LoadedDataset:
    assert cfg.path is not None
    root = Path(cfg.path)
    missing = [f for f in (*PROCESSED_FILES_READ, "checksums.sha256") if not (root / f).is_file()]
    if missing:
        raise DatasetError(f"{cfg.path}: missing {missing} (run `flylab data prepare`)")
    listed = {
        ln.split(maxsplit=1)[1].strip(): ln.split(maxsplit=1)[0]
        for ln in (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    unlisted = [n for n in PROCESSED_FILES_READ if n not in listed]
    if unlisted:
        raise DatasetError(f"{cfg.path}: checksums.sha256 does not list {unlisted}")
    bad = [m.path for m in verify_checksums(root / "checksums.sha256", root)]
    if bad:
        raise DatasetError(
            f"{cfg.path}: content of {bad} does not match checksums.sha256; "
            "run `flylab data verify`"
        )
    files = [(f"{cfg.path}/{name}", listed[name]) for name in PROCESSED_FILES_READ]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    graph = load_graph(root)
    if cfg.expected_dataset_id is not None and graph.dataset_id != cfg.expected_dataset_id:
        raise DatasetError(
            f"{cfg.path}: dataset_id {graph.dataset_id!r} != expected {cfg.expected_dataset_id!r}"
        )
    pf = pq.ParquetFile(root / "neurons.parquet")
    columns = tuple(n for n in pf.schema_arrow.names if n not in ("idx", "body_id"))
    body = pq.read_table(root / "neurons.parquet", columns=["body_id"])["body_id"].to_numpy()
    if not np.array_equal(body, graph.index.body_ids):
        raise DatasetError(f"{cfg.path}: neurons.parquet body_id order differs from graph.npz")

    def loader(name: str) -> NDArray[np.object_]:
        return _objects(pq.read_table(root / "neurons.parquet", columns=[name])[name])

    node_set = manifest.get("node_set", {})
    # Synthetic processed directories exist (PR-01 `prepare_male_cns(release=None, ...)`,
    # `write_processed` on a random graph): biological only if the importer verified a
    # registered release, or (older manifests without the key) the id is not synthetic.
    release = manifest.get("release", "not recorded")
    biological = release is not None and not str(graph.dataset_id).startswith("synthetic:")
    return LoadedDataset(
        graph=graph,
        neurons=NeuronTable(graph.index.body_ids, columns, loader),
        files=tuple(files),
        info={
            "kind": "processed",
            "path": cfg.path,
            "dataset_id": graph.dataset_id,
            "node_set": node_set.get("rule") if isinstance(node_set, dict) else None,
            "processed_manifest_fingerprint": manifest.get("graph_fingerprint_sha256"),
            "release": release,
            "biological_data": biological,
        },
    )


def _read_neurons_table(path: Path) -> tuple[NDArray[np.int64], dict[str, NDArray[np.object_]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    if not reader.fieldnames or "body_id" not in reader.fieldnames:
        raise DatasetError(f"{path.name}: needs a body_id column")
    rows = list(reader)
    try:
        ids = np.array([int(r["body_id"]) for r in rows], dtype=np.int64)
    except ValueError as exc:
        raise DatasetError(f"{path.name}: body_id must be integers") from exc
    cols: dict[str, NDArray[np.object_]] = {}
    for name in reader.fieldnames:
        if name == "body_id":
            continue
        col = np.empty(len(rows), dtype=object)
        col[:] = [r[name] if r[name] not in ("", None) else None for r in rows]
        cols[name] = col
    return ids, cols


def _load_csv(cfg: DatasetConfig) -> LoadedDataset:
    if not (cfg.neurons and cfg.edges and cfg.duplicate_rule and cfg.dataset_id):
        raise DatasetError("csv_fixture needs neurons, edges, duplicate_rule and dataset_id")
    npath, epath = Path(cfg.neurons), Path(cfg.edges)
    for p in (npath, epath):
        if not p.is_file():
            raise DatasetError(f"missing fixture file {p}")
    ids, cols = _read_neurons_table(npath)
    rows = read_edges_csv(epath)
    try:
        graph, _ = build_graph(
            ids,
            rows.pre_body_id,
            rows.post_body_id,
            rows.synapse_count,
            rows.sign,
            dataset_id=cfg.dataset_id,
            duplicate_rule=DuplicateRule(cfg.duplicate_rule),
        )
    except GraphBuildError as exc:
        raise DatasetError(f"{cfg.edges}: {exc}") from exc
    order = np.argsort(ids)  # graph index = body ids ascending
    aligned = {k: v[order] for k, v in cols.items()}
    return LoadedDataset(
        graph=graph,
        neurons=NeuronTable(graph.index.body_ids, tuple(aligned), aligned.__getitem__),
        files=((cfg.neurons, sha256_file(npath)), (cfg.edges, sha256_file(epath))),
        info={"kind": "csv_fixture", "dataset_id": cfg.dataset_id, "biological_data": False},
    )


def _load_synthetic(cfg: DatasetConfig) -> LoadedDataset:
    n, k, seed, inh = cfg.n_neurons, cfg.mean_out_degree, cfg.graph_seed, cfg.inhibitory_fraction
    if n is None or k is None or seed is None or inh is None:
        raise DatasetError("synthetic_random needs n_neurons, mean_out_degree, graph_seed")
    graph, _ = random_graph(n, k, seed, inh)

    def no_columns(name: str) -> NDArray[np.object_]:
        raise KeyError(name)

    return LoadedDataset(
        graph=graph,
        neurons=NeuronTable(graph.index.body_ids, (), no_columns),
        files=(),
        info={
            "kind": "synthetic_random",
            "dataset_id": graph.dataset_id,
            "biological_data": False,
            "note": "generated in memory; reproducible from the config (no files)",
        },
    )


def load_dataset(cfg: DatasetConfig) -> LoadedDataset:
    if cfg.kind == "processed":
        data = _load_processed(cfg)
    elif cfg.kind == "csv_fixture":
        data = _load_csv(cfg)
    else:
        data = _load_synthetic(cfg)
    return replace(data, config=cfg)
