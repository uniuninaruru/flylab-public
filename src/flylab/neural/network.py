"""Sparse synaptic network the engines run on (no dense N x N matrix, ever).

Edges stay sorted by (src, dst) as in `DirectedGraph`, so `row_ptr` slices give each
presynaptic neuron's outgoing edges (CSR by source). Weights are float64 mV:
`w = w_syn * sign * synapse_count` after the unknown-sign policy and any synapse removal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from flylab.data import memory
from flylab.data.graph import DirectedGraph, UnknownSignPolicy


def _readonly[S: np.generic](arr: NDArray[S]) -> NDArray[S]:
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class SynapticNetwork:
    n_neurons: int
    body_ids: NDArray[np.int64]
    src: NDArray[np.int32]
    dst: NDArray[np.int32]
    weight_mV: NDArray[np.float64]
    row_ptr: NDArray[np.int64]
    dataset_id: str
    unknown_sign_policy: UnknownSignPolicy
    n_unknown_sign_edges: int
    n_removed_edges: int
    #: Integer form of the weights when every weight is `weight_scale_mV * signed_count`
    #: exactly (true for `build_network`). Lets an engine sum arrivals as exact integers.
    signed_count: NDArray[np.int32] | None = None
    weight_scale_mV: float | None = None

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    def fingerprint(self) -> str:
        """SHA-256 over ids, edge endpoints and weights."""
        h = hashlib.sha256()
        for arr in (self.body_ids, self.src, self.dst, self.weight_mV):
            h.update(str(arr.dtype).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        return h.hexdigest()

    def summary(self) -> dict[str, object]:
        w = self.weight_mV
        return {
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_excitatory_edges": int((w > 0).sum()),
            "n_inhibitory_edges": int((w < 0).sum()),
            "n_zero_weight_edges": int((w == 0).sum()),
            "unknown_sign_policy": self.unknown_sign_policy.value,
            "n_unknown_sign_edges_before_policy": self.n_unknown_sign_edges,
            "n_removed_edges": self.n_removed_edges,
            "network_fingerprint_sha256": self.fingerprint(),
            "memory_estimate": memory.estimate(self.n_neurons, self.n_edges).to_dict(),
        }

    def dense_for_tests(self) -> NDArray[np.float64]:
        """W[post, pre] in mV, for hand-checking tiny graphs only (refused above 4,096)."""
        memory.require_dense_allowed(self.n_neurons)
        w = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float64)
        np.add.at(w, (self.dst, self.src), self.weight_mV)
        return w


def build_network(
    graph: DirectedGraph,
    policy: UnknownSignPolicy,
    w_syn_mV: float,
    remove_edges: NDArray[np.bool_] | None = None,
) -> SynapticNetwork:
    """Apply synapse removal (mask aligned to `graph` edges), then the unknown-sign policy."""
    n_removed = 0
    g = graph
    if remove_edges is not None:
        if remove_edges.shape != (graph.n_edges,) or remove_edges.dtype != np.bool_:
            raise ValueError("remove_edges must be a boolean mask over the graph's edges")
        n_removed = int(remove_edges.sum())
        keep = ~remove_edges
        g = replace(
            graph,
            src=graph.src[keep],
            dst=graph.dst[keep],
            synapse_count=graph.synapse_count[keep],
            sign=graph.sign[keep],
        )
    e = g.signed_edges(policy)  # raises UnknownSignError for policy ERROR
    counts = e.signed_count.astype(np.int32)  # integer-valued float32 (synapse count x sign)
    if not np.array_equal(counts, e.signed_count):
        raise ValueError("signed synapse counts must be integers")
    weight = float(w_syn_mV) * e.signed_count.astype(np.float64)
    ptr = np.zeros(g.n_neurons + 1, dtype=np.int64)
    np.cumsum(np.bincount(e.src, minlength=g.n_neurons), out=ptr[1:])
    return SynapticNetwork(
        n_neurons=g.n_neurons,
        body_ids=g.index.body_ids,
        src=_readonly(np.ascontiguousarray(e.src, dtype=np.int32)),
        dst=_readonly(np.ascontiguousarray(e.dst, dtype=np.int32)),
        weight_mV=_readonly(weight),
        row_ptr=_readonly(ptr),
        dataset_id=g.dataset_id,
        unknown_sign_policy=policy,
        n_unknown_sign_edges=e.n_unknown_resolved,
        n_removed_edges=n_removed,
        signed_count=_readonly(np.ascontiguousarray(counts)),
        weight_scale_mV=float(w_syn_mV),
    )
