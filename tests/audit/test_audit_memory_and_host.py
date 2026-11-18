"""Audit findings on importer memory and on container limits for the shared GPU host."""

from __future__ import annotations

import re
import tracemalloc
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flylab.data.malecns import stream_edges

ISSUE_MEMORY = "https://github.com/uniuninaruru/flylab/issues/11"
ISSUE_LIMITS = "https://github.com/uniuninaruru/flylab/issues/12"

REPO = Path(__file__).resolve().parents[2]


def test_stream_edges_memory_does_not_grow_when_nothing_is_kept(tmp_path: Path) -> None:
    # Regression test for ISSUE_MEMORY (fixed in c29d70c: the all-segment set is opt-in).
    # Every row joins two fragments outside the node set, so nothing is kept. A streaming
    # filter should then need memory for a few record batches only. At 67dc887 stream_edges
    # kept the set of every body id it had seen (for the `bodies_in_weights` statistic), so its
    # peak grew with the number of distinct bodies: 88.4 M in the real file; the owner measured
    # 7.73 GiB peak RSS for the whole import at 67dc887, 2.22 GiB after the fix.
    rows = 1_200_000  # 2.4 M distinct synthetic fragment ids
    ids = 9_000_000_000_000 + np.arange(2 * rows, dtype=np.int64)
    table = pa.table(
        {
            "body_pre": pa.array(ids[0::2]),
            "body_post": pa.array(ids[1::2]),
            "weight": pa.array(np.ones(rows, dtype=np.int64)),
        }
    )
    path = tmp_path / "weights.feather"
    feather.write_feather(table, str(path), compression="uncompressed", chunksize=65_536)
    del table, ids
    node_ids = np.array([1, 2, 3], dtype=np.int64)

    tracemalloc.start()
    try:
        _, _, _, stats = stream_edges(path, node_ids)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert stats["rows_kept"] == 0  # precondition
    one_batch = 65_536 * 8 * 16  # generous: 16 int64 temporaries of one record batch (~8 MB)
    assert peak <= one_batch, f"peak {peak / 1e6:.0f} MB with 0 rows kept ({ISSUE_MEMORY})"


@pytest.mark.skipif(
    not (REPO / "compose.yaml").is_file(), reason="compose.yaml is not copied into the image"
)
def test_compose_services_have_memory_and_pid_limits() -> None:
    # The GPU host is a shared WSL VM (15 GiB visible, 16 other containers). FlyLab is a guest
    # there, but no service sets a memory or PID limit, so an import or a runaway job competes
    # with the host's services for RAM without bound.
    text = (REPO / "compose.yaml").read_text(encoding="utf-8")
    assert re.search(r"^\s*(mem_limit|memory)\s*:", text, re.MULTILINE)
    assert re.search(r"^\s*pids_limit\s*:", text, re.MULTILINE)
