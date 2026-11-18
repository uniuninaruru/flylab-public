"""Importer on miniature SYNTHETIC files that copy the real MaleCNS v1.0 column names.

Bodies (hand-chosen):
  101 Traced  cb_intrinsic  A      102 Traced  cb_intrinsic  B      103 Traced  -  C
  104 (null)  ol_sensory    R7p    105 Orphan  -  -                 106 Glia    -  -
  107 Anchor  vnc_motor     MN
Node sets: superclass_annotated = {101,102,104,107}; traced = {101,102,103};
           flybrain_compat = traced + photoreceptor 104 = {101,102,103,104}
Weights rows (pre, post, weight):
  101>102 5 | 102>101 3 | 101>104 2 | 104>107 1 | 101>103 4 | 105>101 6 | 999>998 7 | 107>107 2
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

from flylab.cli.main import main
from flylab.data.malecns import (
    ANNOTATIONS,
    NEUROTRANSMITTERS,
    WEIGHTS,
    RawDataError,
    prepare_male_cns,
)
from flylab.data.processed import load_graph, verify_processed

# Synthetic inputs are not the registered MaleCNS objects: opt out of the registry check.
SYN: dict[str, Any] = {"release": None, "dataset_id": "synthetic:malecns-test"}

WEIGHT_ROWS = [
    (101, 102, 5), (102, 101, 3), (101, 104, 2), (104, 107, 1),
    (101, 103, 4), (105, 101, 6), (999, 998, 7), (107, 107, 2),
]  # fmt: skip


def _ann() -> pa.Table:
    ids = [101, 102, 103, 104, 105, 106, 107]
    status = ["Traced", "Traced", "Traced", None, "Orphan", "Glia", "Anchor"]
    sup = ["cb_intrinsic", "cb_intrinsic", None, "ol_sensory", None, None, "vnc_motor"]
    typ = ["A", "B", "C", "R7p", None, None, "MN"]
    n = len(ids)
    cols: dict[str, pa.Array] = {
        "bodyId": pa.array(ids, pa.int64()),
        "type": pa.array(typ),
        "instance": pa.array([f"{t}_R" if t else None for t in typ]),
        "superclass": pa.array(sup),
        "class": pa.array([None] * n, pa.string()),
        "subclass": pa.array([None] * n, pa.string()),
        "somaSide": pa.array(["R", "L", "R", None, None, None, "M"]),
        "status": pa.array(status),
        "statusLabel": pa.array(
            ["Reviewed", "Reviewed", "Roughly traced", None, "Orphan", "Glia", "Anchor"]
        ).dictionary_encode(),
        "flywireType": pa.array([None] * n, pa.string()),
        "hemibrainType": pa.array([None] * n, pa.string()),
        "mancType": pa.array([None] * n, pa.string()),
        "dimorphism": pa.array([None, "male-specific", None, None, None, None, None]),
    }
    return pa.table(cols)


def _nt() -> pa.Table:
    return pa.table(
        {
            "body": pa.array([101, 102, 104, 999], pa.int64()),
            "predicted_nt": ["acetylcholine", "gaba", "unclear", "unclear"],
            "predicted_nt_confidence": [0.9, 0.8, 0.3, None],
            "consensus_nt": ["acetylcholine", "gaba", "histamine", "unclear"],
            "celltype_predicted_nt": ["acetylcholine", "gaba", "unclear", "unclear"],
            "celltype_predicted_nt_confidence": [0.9, 0.8, None, None],
            "ground_truth": [None, "gaba", None, None],
            "total_nt_predictions": pa.array([120, 80, 3, 0], pa.int32()),
        }
    )


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    d = tmp_path / "raw" / "male-cns-v1.0"
    d.mkdir(parents=True)
    feather.write_feather(_ann(), str(d / ANNOTATIONS))
    feather.write_feather(_nt(), str(d / NEUROTRANSMITTERS))
    pre, post, w = (list(x) for x in zip(*WEIGHT_ROWS, strict=True))
    weights = pa.table(
        {"body_pre": pa.array(pre, pa.int64()), "body_post": pa.array(post, pa.int64()),
         "weight": pa.array(w, pa.int64())}
    )  # fmt: skip
    feather.write_feather(weights, str(d / WEIGHTS), chunksize=3)  # 3 record batches
    return d


def test_default_node_set_matches_hand_computation(raw: Path, tmp_path: Path) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)
    assert out.name == "superclass_annotated"
    g = load_graph(out)
    assert g.index.body_ids.tolist() == [101, 102, 104, 107]
    got = list(zip(g.src.tolist(), g.dst.tolist(), g.synapse_count.tolist(), strict=True))
    assert got == [(0, 1, 5), (0, 2, 2), (1, 0, 3), (2, 3, 1), (3, 3, 2)]
    assert set(g.sign.tolist()) == {0}  # no sign assigned in PR-01

    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["node_set"]["all_rules_measured"] == {
        "superclass_annotated": 4,
        "traced": 3,
        "flybrain_compat": 4,
    }
    ef = m["edge_filter"]
    assert (ef["rows_total"], ef["synapses_total"]) == (8, 30)
    assert (ef["rows_kept"], ef["synapses_kept"]) == (5, 13)
    assert (ef["rows_pre_only"], ef["rows_post_only"], ef["rows_neither"]) == (1, 1, 1)
    assert ef["bodies_in_weights"] is None  # not measured by default (expensive at scale)
    assert ef["node_set_bodies_in_weights"] == 4
    assert m["counts"]["n_self_loops"] == 1
    assert m["columns"]["sign"]["evidence"] == "UNKNOWN"
    assert all(i["matches_source_json"] is None for i in m["inputs"])  # no SOURCE.json here
    assert verify_processed(out)["ok"] is True


def test_neuron_table_joins_annotations_and_transmitters(raw: Path, tmp_path: Path) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)
    t = pq.read_table(out / "neurons.parquet").to_pydict()
    assert t["body_id"] == [101, 102, 104, 107]
    assert t["type"] == ["A", "B", "R7p", "MN"]
    assert t["statusLabel"] == ["Reviewed", "Reviewed", None, "Anchor"]  # dictionary decoded
    assert t["predicted_nt"] == ["acetylcholine", "gaba", "unclear", None]  # 107 has no row
    assert t["ground_truth"] == [None, "gaba", None, None]
    assert t["predicted_nt_confidence"][3] is None


@pytest.mark.parametrize(
    ("rule", "ids", "n_edges"),
    [("traced", [101, 102, 103], 3), ("flybrain_compat", [101, 102, 103, 104], 4)],
)
def test_other_node_sets(
    raw: Path, tmp_path: Path, rule: str, ids: list[int], n_edges: int
) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", node_set=rule, progress=None, **SYN)
    g = load_graph(out)
    assert g.index.body_ids.tolist() == ids
    assert g.n_edges == n_edges


def test_source_json_mismatch_is_refused(raw: Path, tmp_path: Path) -> None:
    (raw / "SOURCE.json").write_text(
        json.dumps({"files": {WEIGHTS: {"sha256": "0" * 64}}}), encoding="utf-8"
    )
    with pytest.raises(RawDataError, match="sha256"):
        prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)


def test_missing_input_and_unknown_rule(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown node set"):
        prepare_male_cns(raw, tmp_path / "proc", node_set="everything", progress=None, **SYN)
    (raw / NEUROTRANSMITTERS).unlink()
    with pytest.raises(FileNotFoundError):
        prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)


def test_duplicate_weight_rows_are_refused(raw: Path, tmp_path: Path) -> None:
    dup = pa.table({"body_pre": [101, 101], "body_post": [102, 102], "weight": [1, 2]})
    feather.write_feather(dup, str(raw / WEIGHTS))
    with pytest.raises(ValueError, match="repeated"):
        prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)


def test_cli_refuses_unregistered_inputs_and_verifies(raw: Path, tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    # The CLI always checks the registry: synthetic tables under official names are refused.
    assert main(["data", "prepare", "--raw", str(raw), "--out", str(proc)]) == 3
    out = prepare_male_cns(raw, proc, progress=None, **SYN)
    assert main(["data", "verify", "--processed", str(out)]) == 0
    np.testing.assert_array_equal(load_graph(out).index.body_ids, [101, 102, 104, 107])


def test_synthetic_output_is_labelled_engineered(raw: Path, tmp_path: Path) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", progress=None, **SYN)
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["dataset_id"] == "synthetic:malecns-test"
    assert m["release"] is None
    assert m["columns"]["type"]["evidence"] == "ENGINEERED"
    assert m["columns"]["synapse_count"]["evidence"] == "ENGINEERED"
    assert all(i["matches_registry"] is None for i in m["inputs"])


def test_synthetic_opt_out_needs_a_synthetic_dataset_id(raw: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="synthetic"):
        prepare_male_cns(
            raw, tmp_path / "proc", progress=None, release=None, dataset_id="male-cns:v1.0"
        )


def test_count_all_bodies_option(raw: Path, tmp_path: Path) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", progress=None, count_all_bodies=True, **SYN)
    ef = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["edge_filter"]
    assert ef["bodies_in_weights"] == 8  # 101,102,103,104,105,107,998,999
    assert ef["node_set_bodies_in_weights"] == 4


def test_node_set_body_absent_from_weights_is_counted(raw: Path, tmp_path: Path) -> None:
    out = prepare_male_cns(raw, tmp_path / "proc", node_set="traced", progress=None, **SYN)
    ef = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["edge_filter"]
    assert ef["node_set_bodies_in_weights"] == 3  # 101, 102, 103 all appear in some row
