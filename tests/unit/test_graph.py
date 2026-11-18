"""Graph contract tests. Expected numbers: tests/fixtures/tiny_graph/README.md."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flylab.data.csv_io import FixtureFormatError, read_edges_csv, read_neurons_csv
from flylab.data.graph import (
    ORIENTATION,
    BuildReport,
    DanglingIdError,
    DirectedGraph,
    DuplicateEdgeError,
    DuplicateRule,
    GraphBuildError,
    NeuronIndex,
    SignConflictError,
    UnknownSignError,
    UnknownSignPolicy,
    build_graph,
)
from flylab.data.memory import DenseMatrixRefused
from flylab.data.synthetic import random_graph


def _tiny(
    tiny_dir: Path, rule: DuplicateRule = DuplicateRule.SUM
) -> tuple[DirectedGraph, BuildReport]:
    rows = read_edges_csv(tiny_dir / "edges.csv")
    return build_graph(
        read_neurons_csv(tiny_dir / "neurons.csv"),
        rows.pre_body_id,
        rows.post_body_id,
        rows.synapse_count,
        rows.sign,
        dataset_id="synthetic:tiny:v1",
        duplicate_rule=rule,
    )


# Unique edges sorted by (src, dst): (src, dst, synapse_count, sign).
TINY_EDGES = [
    (0, 1, 5, 1),
    (0, 2, 2, 1),
    (1, 3, 5, -1),  # 3 + 2 merged
    (2, 3, 4, 1),
    (3, 4, 1, 1),
    (4, 4, 1, 1),  # self-loop
    (5, 0, 7, 0),  # unknown sign
]


def test_tiny_report_matches_hand_computation(tiny_dir: Path) -> None:
    _, r = _tiny(tiny_dir)
    assert r.orientation == ORIENTATION
    assert r.n_neurons == 7
    assert r.n_input_rows == 8
    assert r.n_unique_edges == 7
    assert r.n_duplicate_rows_merged == 1
    assert r.total_synapses == 25
    assert r.n_self_loops == 1
    assert r.n_unknown_sign_edges == 1
    assert r.n_isolated_neurons == 1
    assert (r.min_index, r.max_index) == (0, 5)
    assert r.duplicate_rule == "sum"


def test_tiny_edges_sorted_and_aggregated(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    got = list(
        zip(g.src.tolist(), g.dst.tolist(), g.synapse_count.tolist(), g.sign.tolist(), strict=True)
    )
    assert got == TINY_EDGES
    assert g.src.dtype == np.int32
    assert g.dst.dtype == np.int32
    assert g.synapse_count.dtype == np.int64
    assert g.sign.dtype == np.int8


def test_index_map_sorted_ascending(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    assert g.index.body_ids.tolist() == list(range(900001, 900008))
    assert g.index.indices_of([900006, 900001]).tolist() == [5, 0]


def test_input_row_order_does_not_change_graph(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    ids = read_neurons_csv(tiny_dir / "neurons.csv")
    rows = read_edges_csv(tiny_dir / "edges.csv")
    assert rows.sign is not None
    rng = np.random.default_rng(0)
    perm_n = rng.permutation(ids.size)
    perm_e = rng.permutation(rows.pre_body_id.size)
    g2, _ = build_graph(
        ids[perm_n],
        rows.pre_body_id[perm_e],
        rows.post_body_id[perm_e],
        rows.synapse_count[perm_e],
        rows.sign[perm_e],
        dataset_id="synthetic:tiny:v1",
        duplicate_rule=DuplicateRule.SUM,
    )
    assert g2.fingerprint() == g.fingerprint()


def test_row_ptr_is_csr_by_source(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    ptr = g.row_ptr()
    assert ptr.tolist() == [0, 2, 3, 4, 5, 6, 7, 7]
    lo, hi = int(ptr[1]), int(ptr[2])
    assert g.dst[lo:hi].tolist() == [3]  # neuron 1's only outgoing edge goes to 3


def test_duplicates_refused_by_default(tiny_dir: Path) -> None:
    with pytest.raises(DuplicateEdgeError, match=r"\(900002, 900004\)"):
        _tiny(tiny_dir, DuplicateRule.ERROR)


def test_duplicate_rows_with_conflicting_signs_refused() -> None:
    with pytest.raises(SignConflictError):
        build_graph(
            [1, 2],
            [1, 1],
            [2, 2],
            [1, 1],
            [1, -1],
            dataset_id="t",
            duplicate_rule=DuplicateRule.SUM,
        )


def test_dangling_edge_id_refused() -> None:
    with pytest.raises(DanglingIdError, match="123"):
        build_graph([1, 2], [1], [123], [4], dataset_id="t")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"body_ids": [1, 1]}, "duplicate body_id"),
        ({"body_ids": [-1, 2], "pre": [2], "post": [2]}, "non-negative"),
        ({"counts": [0]}, "> 0"),
        ({"counts": [1.5]}, "integers"),
        ({"signs": [2]}, "-1"),
        ({"post": [2, 1]}, "same length"),
    ],
)
def test_invalid_inputs_refused(kwargs: dict[str, list[float]], match: str) -> None:
    args: dict[str, list[float]] = {
        "body_ids": [1, 2],
        "pre": [1],
        "post": [2],
        "counts": [3],
        "signs": [1],
    }
    args.update(kwargs)
    with pytest.raises(GraphBuildError, match=match):
        build_graph(
            args["body_ids"],
            args["pre"],
            args["post"],
            args["counts"],
            args["signs"],
            dataset_id="t",
        )


def test_missing_signs_default_to_unknown() -> None:
    g, r = build_graph([1, 2], [1], [2], [3], dataset_id="t")
    assert g.sign.tolist() == [0]
    assert r.n_unknown_sign_edges == 1
    with pytest.raises(UnknownSignError):
        g.signed_edges()


def test_empty_graph() -> None:
    g, r = build_graph([5, 6], [], [], [], dataset_id="t")
    assert g.n_edges == 0
    assert r.n_isolated_neurons == 2
    assert r.min_index is None
    assert g.row_ptr().tolist() == [0, 0, 0]


def test_unknown_sign_policies(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    with pytest.raises(UnknownSignError, match="1 edge"):
        g.signed_edges()
    excl = g.signed_edges(UnknownSignPolicy.EXCLUDE)
    assert excl.src.size == 6
    assert excl.n_unknown_resolved == 1
    exc = g.signed_edges(UnknownSignPolicy.AS_EXCITATORY)
    inh = g.signed_edges(UnknownSignPolicy.AS_INHIBITORY)
    assert exc.signed_count[-1] == 7.0
    assert inh.signed_count[-1] == -7.0
    assert exc.signed_count[2] == -5.0  # known signs are untouched


def test_dense_matrix_hand_check(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    w = g.to_dense_for_tests(UnknownSignPolicy.EXCLUDE)  # w[post, pre]
    expected = np.zeros((7, 7), dtype=np.float32)
    expected[1, 0], expected[2, 0], expected[3, 1] = 5, 2, -5
    expected[3, 2], expected[4, 3], expected[4, 4] = 4, 1, 1
    np.testing.assert_array_equal(w, expected)
    spikes = np.zeros(7, dtype=np.float32)
    spikes[0] = 1.0  # neuron 0 (body 900001) fires
    np.testing.assert_array_equal(w @ spikes, [0, 5, 2, 0, 0, 0, 0])


def test_sparse_scatter_matches_dense_on_random_graph() -> None:
    g, _ = random_graph(n_neurons=300, mean_out_degree=8, seed=1)
    e = g.signed_edges()
    spikes = (np.random.default_rng(2).random(300) < 0.1).astype(np.float32)
    sparse_in = np.zeros(300, dtype=np.float64)
    np.add.at(sparse_in, e.dst, e.signed_count * spikes[e.src])
    dense_in = g.to_dense_for_tests().astype(np.float64) @ spikes
    np.testing.assert_allclose(sparse_in, dense_in, rtol=0, atol=1e-9)


def test_dense_refused_above_limit() -> None:
    g, _ = random_graph(n_neurons=5000, mean_out_degree=0, seed=0)
    with pytest.raises(DenseMatrixRefused, match="5000x5000"):
        g.to_dense_for_tests()


def test_graph_arrays_are_read_only(tiny_dir: Path) -> None:
    g, _ = _tiny(tiny_dir)
    with pytest.raises(ValueError, match="read-only"):
        g.src[0] = 3


def test_random_graph_is_deterministic() -> None:
    a, _ = random_graph(200, 5, seed=7)
    b, _ = random_graph(200, 5, seed=7)
    c, _ = random_graph(200, 5, seed=8)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()
    assert a.dataset_id.startswith("synthetic:")


def test_neuron_index_rejects_2d() -> None:
    with pytest.raises(GraphBuildError, match="1-D"):
        NeuronIndex.from_body_ids([[1, 2]])


def test_csv_missing_column(tmp_path: Path) -> None:
    p = tmp_path / "edges.csv"
    p.write_text("pre_body_id,post_body_id\n1,2\n", encoding="utf-8")
    with pytest.raises(FixtureFormatError, match="synapse_count"):
        read_edges_csv(p)
