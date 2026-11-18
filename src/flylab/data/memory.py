"""Back-of-envelope storage estimates for graph layouts (spec section 7.3).

These are planning numbers *before* allocator, temporary and scatter buffers. They are not
measurements: real device memory must come from torch/NVML on the target GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Largest neuron count for which a dense float32 N x N matrix may be built (64 MiB).
#: Dense matrices exist only for hand-checking tiny fixture graphs.
MAX_DENSE_NEURONS = 4096


class DenseMatrixRefused(MemoryError):
    """A dense N x N matrix was requested for a graph larger than MAX_DENSE_NEURONS."""


def dense_bytes(n_neurons: int, dtype_bytes: int = 4) -> int:
    """Bytes for a dense N x N matrix."""
    return n_neurons * n_neurons * dtype_bytes


def require_dense_allowed(n_neurons: int) -> None:
    if n_neurons > MAX_DENSE_NEURONS:
        raise DenseMatrixRefused(
            f"refusing dense {n_neurons}x{n_neurons} matrix "
            f"({dense_bytes(n_neurons) / 1e9:.1f} GB float32); use the sparse edge arrays"
        )


def coo_bytes(n_edges: int, index_bytes: int = 4, weight_bytes: int = 4) -> int:
    """src + dst + weight per edge (12 bytes/edge with int32 indices and float32 weights)."""
    return n_edges * (2 * index_bytes + weight_bytes)


def csr_bytes(n_neurons: int, n_edges: int, index_bytes: int = 4, weight_bytes: int = 4) -> int:
    """col_indices + weight per edge, plus row_ptr of length N+1.

    Note: an int32 row_ptr only works while n_edges < 2**31; FlyLab's own row_ptr is int64.
    """
    return n_edges * (index_bytes + weight_bytes) + (n_neurons + 1) * index_bytes


def neuron_vector_bytes(n_neurons: int, dtype_bytes: int = 4) -> int:
    """One per-neuron state vector (membrane potential, input current, ...)."""
    return n_neurons * dtype_bytes


@dataclass(frozen=True)
class MemoryEstimate:
    n_neurons: int
    n_edges: int
    coo_bytes: int
    csr_bytes: int
    neuron_vector_bytes: int
    dense_bytes: int
    note: str = (
        "Planning estimate only. Excludes allocator overhead, temporaries, scatter buffers, "
        "delays and recording. Not a measurement."
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate(n_neurons: int, n_edges: int) -> MemoryEstimate:
    if n_neurons < 0 or n_edges < 0:
        raise ValueError("counts must be non-negative")
    return MemoryEstimate(
        n_neurons=n_neurons,
        n_edges=n_edges,
        coo_bytes=coo_bytes(n_edges),
        csr_bytes=csr_bytes(n_neurons, n_edges),
        neuron_vector_bytes=neuron_vector_bytes(n_neurons),
        dense_bytes=dense_bytes(n_neurons),
    )
