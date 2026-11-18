"""Audit finding: `verify_processed` passes processed directories whose files disagree.

`verify_processed` re-hashes files listed in checksums.sha256 and checks graph.npz invariants,
but it never compares the contents of neurons.parquet or edges.parquet with graph.npz, and it
accepts a checksums.sha256 that omits files. Checksums prove only that files did not change
after writing; they cannot catch a writer that wrote inconsistent files, because the
checksums are computed from those files. Spec section 6.2 also asks `verify` to report source
file hashes and unknown IDs, which the report does not contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flylab.data.malecns import prepare_male_cns
from flylab.data.processed import OUTPUT_FILES, verify_processed, write_processed
from flylab.data.synthetic import random_graph
from flylab.provenance.hashing import write_checksums

ISSUE = "https://github.com/uniuninaruru/flylab/issues/4"


def _processed(tmp_path: Path) -> Path:
    g, r = random_graph(200, 4, seed=11)
    labels = pa.array([f"type_{i}" for i in range(g.n_neurons)])
    return write_processed(tmp_path / "proc", g, r, {"type": labels}, {"node_set": {"rule": "t"}})


def _rehash(d: Path) -> None:
    """What a buggy writer produces: inconsistent files with matching checksums."""
    write_checksums([d / f for f in OUTPUT_FILES], d / "checksums.sha256", root=d)


def test_control_untouched_directory_verifies(tmp_path: Path) -> None:
    """Control (not a finding): the unmodified directory passes."""
    assert verify_processed(_processed(tmp_path))["ok"] is True


def test_edges_parquet_with_reversed_direction_is_detected(tmp_path: Path) -> None:
    # edges.parquet is the spec's canonical edge table; here every edge points the wrong way
    # (dst -> src) while graph.npz is correct. A consumer of the Parquet file would simulate
    # the reversed connectome.
    d = _processed(tmp_path)
    e = pq.read_table(d / "edges.parquet")
    flipped = pa.table(
        {
            "src_idx": e["dst_idx"],
            "dst_idx": e["src_idx"],
            "synapse_count": e["synapse_count"],
            "sign": e["sign"],
        }
    )
    pq.write_table(flipped, d / "edges.parquet")
    _rehash(d)
    assert verify_processed(d)["ok"] is False


def test_neurons_parquet_misaligned_with_index_is_detected(tmp_path: Path) -> None:
    # Reversing the rows attaches every annotation to the wrong neuron index.
    d = _processed(tmp_path)
    t = pq.read_table(d / "neurons.parquet")
    pq.write_table(t.take(pa.array(np.arange(t.num_rows)[::-1])), d / "neurons.parquet")
    _rehash(d)
    assert verify_processed(d)["ok"] is False


def test_checksum_file_that_omits_files_is_rejected(tmp_path: Path) -> None:
    # An empty (or truncated) checksums.sha256 verifies nothing, yet the directory passes.
    d = _processed(tmp_path)
    (d / "checksums.sha256").write_text("", encoding="utf-8")
    assert verify_processed(d)["ok"] is False


def test_report_includes_source_file_hashes(male_cns_raw: Path, tmp_path: Path) -> None:
    # Spec 6.2: "verify reports ... source file hashes".
    d = prepare_male_cns(
        male_cns_raw, tmp_path / "proc", progress=None, release=None, dataset_id="synthetic:audit"
    )
    inputs = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["inputs"]
    report = json.dumps(verify_processed(d))
    assert all(i["sha256"] in report for i in inputs)


def test_report_includes_unknown_id_counts(male_cns_raw: Path, tmp_path: Path) -> None:
    # Spec 6.2: "verify reports ... unknown IDs". The fixture has one weights row whose
    # presynaptic body is outside the node set (counted as `rows_post_only`: only the post
    # end is in the set); the manifest counts it, the verify report does not.
    d = prepare_male_cns(
        male_cns_raw, tmp_path / "proc", progress=None, release=None, dataset_id="synthetic:audit"
    )
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["edge_filter"]["rows_post_only"] == 1  # precondition
    report = json.dumps(verify_processed(d))
    assert any(k in report for k in ("rows_pre_only", "rows_post_only", "unknown_id"))
