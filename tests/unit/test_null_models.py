"""Degree-preserving null graph (flylab.data.null_models). Synthetic graphs only."""

from __future__ import annotations

import numpy as np
import pytest

from flylab.data.graph import DirectedGraph, NeuronIndex, build_graph_indexed
from flylab.data.null_models import NullModelError, degree_preserving_swap, make_null_graph
from flylab.data.synthetic import random_graph


def _deg(a: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(a, minlength=n)


def _strength(a: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(a, weights=w, minlength=n)


@pytest.fixture
def graph() -> DirectedGraph:
    g, _ = random_graph(300, 12, seed=5, inhibitory_fraction=0.3)
    return g


def test_preserves_degrees_out_strength_and_presynaptic_sign(graph: DirectedGraph) -> None:
    null, rep = degree_preserving_swap(graph, seed=1)
    n = graph.n_neurons
    assert null.n_edges == graph.n_edges
    assert np.array_equal(_deg(null.src, n), _deg(graph.src, n))
    assert np.array_equal(_deg(null.dst, n), _deg(graph.dst, n))
    assert np.array_equal(
        _strength(null.src, null.synapse_count, n), _strength(graph.src, graph.synapse_count, n)
    )
    # sign is a presynaptic property in random_graph: every neuron keeps one output sign
    for g in (graph, null):
        lo = np.full(n, 2, dtype=np.int8)
        hi = np.full(n, -2, dtype=np.int8)
        np.minimum.at(lo, g.src, g.sign)
        np.maximum.at(hi, g.src, g.sign)
        has = _deg(g.src, n) > 0
        assert np.array_equal(lo[has], hi[has])
    sign_of = {int(s): int(v) for s, v in zip(graph.src, graph.sign, strict=True)}
    assert all(sign_of[int(s)] == int(v) for s, v in zip(null.src, null.sign, strict=True))
    assert rep.in_degree_preserved
    assert rep.out_degree_preserved
    assert rep.out_strength_preserved


def test_no_self_loops_or_repeated_pairs(graph: DirectedGraph) -> None:
    null, _ = degree_preserving_swap(graph, seed=2)
    assert not (null.src == null.dst).any()
    keys = null.src.astype(np.int64) * graph.n_neurons + null.dst
    assert np.unique(keys).size == keys.size


def test_actually_shuffles_and_is_reproducible(graph: DirectedGraph) -> None:
    a, ra = degree_preserving_swap(graph, seed=3)
    b, _ = degree_preserving_swap(graph, seed=3)
    c, _ = degree_preserving_swap(graph, seed=4)
    assert a.fingerprint() == b.fingerprint() == ra.null_fingerprint_sha256
    assert a.fingerprint() != c.fingerprint()
    assert a.fingerprint() != graph.fingerprint()
    assert ra.fraction_pairs_kept < 0.2  # a random graph of density 0.04 keeps few pairs
    assert ra.original_fingerprint_sha256 == graph.fingerprint()
    assert "null:degree_preserving_swap:seed=3" in a.dataset_id
    d = ra.to_dict()
    assert d["evidence"] == "ENGINEERED"
    not_kept = d["does_not_preserve"]
    assert isinstance(not_kept, list)
    assert "in-strength" in not_kept


def test_unrepairable_graph_raises() -> None:
    # complete digraph on 4 nodes without self-loops: every swap of targets that keeps degrees
    # creates a self-loop or a repeated pair unless it reproduces the graph exactly; with more
    # bad edges than good ones the repair refuses instead of looping forever
    idx = NeuronIndex.from_body_ids(np.arange(4))
    src, dst = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    keep = src != dst
    g, _ = build_graph_indexed(
        idx, src[keep], dst[keep], np.ones(keep.sum(), dtype=np.int64), dataset_id="synthetic:k4"
    )
    with pytest.raises(NullModelError):
        degree_preserving_swap(g, seed=0, max_rounds=5)


def test_make_null_graph_rejects_unknown_method(graph: DirectedGraph) -> None:
    with pytest.raises(ValueError, match="unknown null model"):
        make_null_graph(graph, "erdos_renyi", 1)
    with pytest.raises(ValueError, match="seed"):
        degree_preserving_swap(graph, seed=-1)
