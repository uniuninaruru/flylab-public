from __future__ import annotations

import pytest

from flylab.data import memory

# Neuron count from the MaleCNS Cell paper abstract (EMPIRICAL, publication). Used here only
# as a size for arithmetic; the importer will measure its own node count (PR-01).
PUBLISHED_NEURONS = 166_691


def test_coo_is_12_bytes_per_edge() -> None:
    assert memory.coo_bytes(1_000) == 12_000


def test_csr_is_8_bytes_per_edge_plus_row_ptr() -> None:
    assert memory.csr_bytes(n_neurons=10, n_edges=1_000) == 8_000 + 11 * 4


def test_neuron_vector_size_matches_spec() -> None:
    assert memory.neuron_vector_bytes(PUBLISHED_NEURONS) == 666_764  # about 0.67 MB


def test_dense_matrix_size_matches_spec() -> None:
    b = memory.dense_bytes(PUBLISHED_NEURONS)
    assert b == PUBLISHED_NEURONS**2 * 4
    assert round(b / 1e9, 1) == 111.1  # "roughly 111 GB decimal - prohibited"


def test_dense_guard_boundary() -> None:
    memory.require_dense_allowed(memory.MAX_DENSE_NEURONS)
    with pytest.raises(memory.DenseMatrixRefused):
        memory.require_dense_allowed(memory.MAX_DENSE_NEURONS + 1)


def test_estimate_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        memory.estimate(-1, 0)


def test_estimate_is_labelled_as_not_measured() -> None:
    est = memory.estimate(10, 20).to_dict()
    assert "Not a measurement" in str(est["note"])
