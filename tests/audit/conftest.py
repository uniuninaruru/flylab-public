"""Fixtures for the independent audit tests (docs/audit/AUDIT_2026-11.md, and
docs/audit/PR-03-review.md for the test_audit_pr03_*.py modules).

Every test in tests/audit/ demonstrates one filed finding and is marked
`xfail(strict=True)`: it fails today, and a fix makes it pass, which strict mode turns
into a CI failure until the marker is removed together with the fix.

The raw files below are SYNTHETIC: tiny tables that copy the real MaleCNS v1.0 column names
(checked against the public files on 2026-11-04), not connectome data.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flylab.data.malecns import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS

# Bodies 1..4 carry a superclass (the default node set); 5 is an unannotated fragment.
_IDS = [1, 2, 3, 4, 5]
# (pre, post, weight). 5 -> 1 is dropped by the node-set filter (pre not in the node set).
WEIGHT_ROWS = [(1, 2, 5), (2, 1, 3), (2, 3, 2), (3, 4, 1), (5, 1, 6)]


def _annotations() -> pa.Table:
    n = len(_IDS)
    nulls = pa.array([None] * n, pa.string())
    return pa.table(
        {
            "bodyId": pa.array(_IDS, pa.int64()),
            "type": pa.array(["A", "B", "C", "D", None]),
            "instance": pa.array(["A_R", "B_L", "C_R", "D_R", None]),
            "superclass": pa.array(
                ["cb_intrinsic", "cb_intrinsic", "ol_sensory", "vnc_motor", None]
            ),
            "class": nulls,
            "subclass": nulls,
            "somaSide": pa.array(["R", "L", "R", "R", None]),
            "status": pa.array(["Traced", "Traced", "Traced", "Traced", "Orphan"]),
            "statusLabel": pa.array(["Traced"] * 4 + ["Orphan"]).dictionary_encode(),
            "flywireType": nulls,
            "hemibrainType": nulls,
            "mancType": nulls,
            "dimorphism": nulls,
        }
    )


def _neurotransmitters() -> pa.Table:
    return pa.table(
        {
            "body": pa.array([1, 2, 3], pa.int64()),
            "predicted_nt": ["acetylcholine", "gaba", "unclear"],
            "predicted_nt_confidence": [0.9, 0.8, 0.3],
            "consensus_nt": ["acetylcholine", "gaba", "unclear"],
            "celltype_predicted_nt": ["acetylcholine", "gaba", "unclear"],
            "celltype_predicted_nt_confidence": [0.9, 0.8, None],
            "ground_truth": pa.array([None, None, None], pa.string()),
            "total_nt_predictions": pa.array([10, 8, 1], pa.int32()),
        }
    )


@pytest.fixture
def male_cns_raw(tmp_path: Path) -> Path:
    """A raw directory with the three official file names, holding SYNTHETIC tables."""
    d = tmp_path / "raw" / "male-cns-v1.0"
    d.mkdir(parents=True)
    feather.write_feather(_annotations(), str(d / ANNOTATIONS))
    feather.write_feather(_neurotransmitters(), str(d / NEUROTRANSMITTERS))
    pre, post, w = (list(x) for x in zip(*WEIGHT_ROWS, strict=True))
    weights = pa.table(
        {
            "body_pre": pa.array(pre, pa.int64()),
            "body_post": pa.array(post, pa.int64()),
            "weight": pa.array(w, pa.int64()),
        }
    )
    feather.write_feather(weights, str(d / WEIGHTS))
    return d
