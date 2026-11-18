"""Small CSV reader for synthetic fixtures. Not for MaleCNS bulk data (that is Feather/Parquet).

neurons.csv columns: body_id[, label]
edges.csv columns:   pre_body_id, post_body_id, synapse_count[, sign]
Lines starting with '#' are comments.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

NEURON_REQUIRED = ("body_id",)
EDGE_REQUIRED = ("pre_body_id", "post_body_id", "synapse_count")


class FixtureFormatError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class EdgeRows:
    pre_body_id: NDArray[np.int64]
    post_body_id: NDArray[np.int64]
    synapse_count: NDArray[np.int64]
    sign: NDArray[np.int8] | None


def _read_rows(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    header = tuple(reader.fieldnames or ())
    missing = [c for c in required if c not in header]
    if missing:
        raise FixtureFormatError(f"{path.name}: missing column(s) {missing}; found {list(header)}")
    return list(reader)


def _int_column(rows: list[dict[str, str]], name: str, path: Path) -> NDArray[np.int64]:
    try:
        return np.array([int(r[name]) for r in rows], dtype=np.int64)
    except (TypeError, ValueError) as exc:
        raise FixtureFormatError(f"{path.name}: column {name!r} must be integers") from exc


def read_neurons_csv(path: str | Path) -> NDArray[np.int64]:
    p = Path(path)
    return _int_column(_read_rows(p, NEURON_REQUIRED), "body_id", p)


def read_edges_csv(path: str | Path) -> EdgeRows:
    p = Path(path)
    rows = _read_rows(p, EDGE_REQUIRED)
    has_sign = bool(rows) and "sign" in rows[0]
    return EdgeRows(
        pre_body_id=_int_column(rows, "pre_body_id", p),
        post_body_id=_int_column(rows, "post_body_id", p),
        synapse_count=_int_column(rows, "synapse_count", p),
        sign=_int_column(rows, "sign", p).astype(np.int8) if has_sign else None,
    )
