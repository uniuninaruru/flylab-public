"""Audit findings on evidence labels and the on-disk encoding of unknown synapse signs."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from flylab.data.malecns import prepare_male_cns

ISSUE_LABELS = "https://github.com/uniuninaruru/flylab/issues/9"
ISSUE_SIGN = "https://github.com/uniuninaruru/flylab/issues/10"


def test_every_output_column_has_an_evidence_label(male_cns_raw: Path, tmp_path: Path) -> None:
    # Spec 1.3: every dataset field carries exactly one label. manifest["columns"] labels the
    # annotation, transmitter, synapse_count and sign columns, but not `idx`, `src_idx`,
    # `dst_idx` (all ENGINEERED: the deterministic index convention).
    d = prepare_male_cns(
        male_cns_raw, tmp_path / "proc", progress=None, release=None, dataset_id="synthetic:audit"
    )
    labels = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["columns"]
    columns = set(pq.read_schema(d / "neurons.parquet").names)
    columns |= set(pq.read_schema(d / "edges.parquet").names)
    unlabeled = sorted(c for c in columns if c not in labels)
    assert unlabeled == []


def test_unknown_sign_is_not_a_plain_zero_in_edges_parquet(
    male_cns_raw: Path, tmp_path: Path
) -> None:
    # Spec 6.1 declares `sign: int8?` (nullable) and "never silently assign +/-". PR-01 writes
    # unknown as a non-null 0 with no schema-level marker, so the obvious consumer arithmetic
    # `synapse_count * sign` turns every unknown edge into a zero-weight edge (today: all of
    # them) without any error. Accept either fix: nulls, or Parquet field metadata that names
    # the sentinel.
    d = prepare_male_cns(
        male_cns_raw, tmp_path / "proc", progress=None, release=None, dataset_id="synthetic:audit"
    )
    edges = pq.read_table(d / "edges.parquet")
    sign = edges["sign"]
    unknown_is_null = sign.null_count == edges.num_rows
    meta = edges.schema.field("sign").metadata or {}
    sentinel_declared = any(b"unknown" in v.lower() for v in meta.values())
    assert edges.num_rows > 0
    assert unknown_is_null or sentinel_declared
