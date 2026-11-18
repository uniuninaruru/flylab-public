"""Canonical directed sparse graph and deterministic body_id -> index map (spec section 6.1).

Orientation: edge k points from `src[k]` (presynaptic neuron) to `dst[k]` (postsynaptic
neuron). Nothing in this module builds a dense N x N matrix except
`DirectedGraph.to_dense_for_tests`, which refuses graphs above `MAX_DENSE_NEURONS`.

What is stored here is only what the dataset provides: identities, directed synapse counts,
and a sign that may be UNKNOWN. Turning counts into synaptic efficacy is a model decision
and belongs to the neural engine, where it must be labelled (see docs/MODEL_ASSUMPTIONS.md).
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flylab.data.memory import require_dense_allowed

SIGN_INHIBITORY = -1
SIGN_UNKNOWN = 0  # Not "zero weight". Must be resolved by an explicit UnknownSignPolicy.
SIGN_EXCITATORY = 1
_VALID_SIGNS = (SIGN_INHIBITORY, SIGN_UNKNOWN, SIGN_EXCITATORY)

ORIENTATION = "src=presynaptic -> dst=postsynaptic"


class DuplicateRule(enum.Enum):
    """What to do when the input has more than one row for the same (pre, post) pair."""

    ERROR = "error"  # Refuse. Default, so aggregation is never silent.
    SUM = "sum"  # Sum synapse counts; the rows' signs must agree. Direction is kept.


class UnknownSignPolicy(enum.Enum):
    """How a consumer resolves SIGN_UNKNOWN edges. Every choice must be recorded in run output."""

    ERROR = "error"  # Refuse if any edge has an unknown sign. Default.
    EXCLUDE = "exclude"  # Drop unknown-sign edges.
    AS_EXCITATORY = "as_excitatory"  # Sensitivity analysis only.
    AS_INHIBITORY = "as_inhibitory"  # Sensitivity analysis only.


class GraphBuildError(ValueError):
    """Input rows cannot form a valid graph."""


class DanglingIdError(GraphBuildError):
    """An edge refers to a body_id that is not in the neuron table."""


class DuplicateEdgeError(GraphBuildError):
    """Repeated (pre, post) rows while DuplicateRule.ERROR is in force."""


class SignConflictError(GraphBuildError):
    """Repeated (pre, post) rows disagree on sign."""


class UnknownSignError(ValueError):
    """Unknown-sign edges present while UnknownSignPolicy.ERROR is in force."""


def _readonly[S: np.generic](arr: NDArray[S]) -> NDArray[S]:
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class NeuronIndex:
    """Deterministic dense index: neuron `i` is `body_ids[i]`, with body_ids sorted ascending.

    Sorting makes the map independent of input row order, so the same neuron set always
    gets the same indices.
    """

    body_ids: NDArray[np.int64]

    @classmethod
    def from_body_ids(cls, body_ids: ArrayLike) -> NeuronIndex:
        ids = np.asarray(body_ids)
        if ids.ndim != 1:
            raise GraphBuildError(f"body_ids must be 1-D, got shape {ids.shape}")
        if ids.size and not np.issubdtype(ids.dtype, np.integer):
            raise GraphBuildError(f"body_ids must be integers, got dtype {ids.dtype}")
        ids = ids.astype(np.int64, copy=False)
        if ids.size and ids.min() < 0:
            raise GraphBuildError("body_ids must be non-negative")
        uniq, counts = np.unique(ids, return_counts=True)
        if (counts > 1).any():
            dupes = uniq[counts > 1][:10].tolist()
            raise GraphBuildError(f"duplicate body_id(s) in neuron table, e.g. {dupes}")
        if uniq.size > np.iinfo(np.int32).max:
            raise GraphBuildError("more neurons than an int32 index can address")
        return cls(_readonly(uniq))

    @property
    def n(self) -> int:
        return int(self.body_ids.size)

    def indices_of(self, body_ids: ArrayLike) -> NDArray[np.int32]:
        """Map body_ids to dense indices. Raises DanglingIdError on any unknown id."""
        q = np.asarray(body_ids, dtype=np.int64)
        if self.n == 0:
            if q.size:
                raise DanglingIdError(f"{q.size} reference(s) to an empty neuron table")
            return np.zeros(0, dtype=np.int32)
        pos = np.searchsorted(self.body_ids, q)
        found = self.body_ids[np.minimum(pos, self.n - 1)] == q
        if not found.all():
            missing = np.unique(q[~found])
            raise DanglingIdError(
                f"{missing.size} body_id(s) not in neuron table "
                f"({int((~found).sum())} rows), e.g. {missing[:10].tolist()}"
            )
        return pos.astype(np.int32)


@dataclass(frozen=True)
class BuildReport:
    """Counts produced while building a graph. Every field is measured, never assumed."""

    dataset_id: str
    orientation: str
    duplicate_rule: str
    n_neurons: int
    n_input_rows: int
    n_unique_edges: int
    n_duplicate_rows_merged: int
    total_synapses: int
    n_self_loops: int
    n_unknown_sign_edges: int
    n_isolated_neurons: int
    min_index: int | None
    max_index: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, eq=False)
class SignedEdges:
    """Edges with signed synapse counts after an explicit unknown-sign policy was applied."""

    src: NDArray[np.int32]
    dst: NDArray[np.int32]
    signed_count: NDArray[np.float32]
    policy: UnknownSignPolicy
    n_unknown_resolved: int


@dataclass(frozen=True, eq=False)
class DirectedGraph:
    """Unique directed edges sorted by (src, dst); `row_ptr` gives CSR slices by source."""

    index: NeuronIndex
    src: NDArray[np.int32]
    dst: NDArray[np.int32]
    synapse_count: NDArray[np.int64]
    sign: NDArray[np.int8]
    duplicate_rule: DuplicateRule
    dataset_id: str

    @property
    def n_neurons(self) -> int:
        return self.index.n

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    @property
    def total_synapses(self) -> int:
        return int(self.synapse_count.sum())

    def row_ptr(self) -> NDArray[np.int64]:
        """CSR row pointer (length N+1): neuron i's outgoing edges are [ptr[i], ptr[i+1])."""
        ptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.src, minlength=self.n_neurons), out=ptr[1:])
        return ptr

    def signed_edges(self, policy: UnknownSignPolicy = UnknownSignPolicy.ERROR) -> SignedEdges:
        """Synapse count x sign, after resolving unknown signs by an explicit policy."""
        unknown = self.sign == SIGN_UNKNOWN
        n_unknown = int(unknown.sum())
        sign = self.sign.astype(np.float32)
        src, dst, count = self.src, self.dst, self.synapse_count
        if n_unknown and policy is UnknownSignPolicy.ERROR:
            raise UnknownSignError(
                f"{n_unknown} edge(s) have unknown sign; choose an UnknownSignPolicy explicitly"
            )
        if policy is UnknownSignPolicy.EXCLUDE:
            keep = ~unknown
            src, dst, count, sign = src[keep], dst[keep], count[keep], sign[keep]
        elif policy is UnknownSignPolicy.AS_EXCITATORY:
            sign[unknown] = SIGN_EXCITATORY
        elif policy is UnknownSignPolicy.AS_INHIBITORY:
            sign[unknown] = SIGN_INHIBITORY
        return SignedEdges(
            src=src,
            dst=dst,
            signed_count=(count.astype(np.float32) * sign).astype(np.float32),
            policy=policy,
            n_unknown_resolved=n_unknown,
        )

    def to_dense_for_tests(
        self, policy: UnknownSignPolicy = UnknownSignPolicy.ERROR
    ) -> NDArray[np.float32]:
        """Dense W[dst, src] of signed counts, for hand-checking tiny graphs only.

        Indexed [post, pre] so that `W @ spikes` gives the input each neuron receives.
        """
        require_dense_allowed(self.n_neurons)
        e = self.signed_edges(policy)
        w = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float32)
        np.add.at(w, (e.dst, e.src), e.signed_count)
        return w

    def fingerprint(self) -> str:
        """SHA-256 over ids, edges, counts and signs: equal graphs give equal fingerprints."""
        h = hashlib.sha256()
        for arr in (self.index.body_ids, self.src, self.dst, self.synapse_count, self.sign):
            h.update(str(arr.dtype).encode())
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(self.duplicate_rule.value.encode())
        return h.hexdigest()


def build_graph(
    body_ids: ArrayLike,
    pre_body_ids: ArrayLike,
    post_body_ids: ArrayLike,
    synapse_counts: ArrayLike,
    signs: ArrayLike | None = None,
    *,
    dataset_id: str,
    duplicate_rule: DuplicateRule = DuplicateRule.ERROR,
) -> tuple[DirectedGraph, BuildReport]:
    """Build the canonical graph from a neuron table and directed connection rows.

    Args:
        body_ids: Every neuron in the graph (the node set). Edges must refer only to these.
        pre_body_ids: Presynaptic body_id of each connection row.
        post_body_ids: Postsynaptic body_id of each connection row.
        synapse_counts: Positive integer synapse count of each row.
        signs: Optional per-row sign in {-1, 0, +1}; 0 means unknown. Defaults to all unknown.
        dataset_id: Where the rows came from, e.g. "male-cns:v1.0" or "synthetic:tiny:v1".
        duplicate_rule: How to treat repeated (pre, post) rows.
    """
    index = NeuronIndex.from_body_ids(body_ids)
    pre_ids = np.asarray(pre_body_ids)
    post_ids = np.asarray(post_body_ids)
    n_rows = int(pre_ids.size)
    if (
        pre_ids.shape != (n_rows,)
        or post_ids.shape != (n_rows,)
        or np.shape(synapse_counts) != (n_rows,)
    ):
        raise GraphBuildError("pre, post and synapse_counts must be 1-D and the same length")
    _check_counts(np.asarray(synapse_counts))
    return build_graph_indexed(
        index,
        index.indices_of(pre_ids),
        index.indices_of(post_ids),
        synapse_counts,
        signs,
        dataset_id=dataset_id,
        duplicate_rule=duplicate_rule,
    )


def _check_counts(counts: NDArray[Any]) -> None:
    if counts.size and not np.issubdtype(counts.dtype, np.integer):
        raise GraphBuildError(f"synapse_counts must be integers, got dtype {counts.dtype}")
    if counts.size and counts.min() <= 0:
        raise GraphBuildError("synapse_counts must be > 0 (a zero-count row is not a connection)")


def build_graph_indexed(
    index: NeuronIndex,
    src_idx: ArrayLike,
    dst_idx: ArrayLike,
    synapse_counts: ArrayLike,
    signs: ArrayLike | None = None,
    *,
    dataset_id: str,
    duplicate_rule: DuplicateRule = DuplicateRule.ERROR,
) -> tuple[DirectedGraph, BuildReport]:
    """Like `build_graph`, for rows already mapped to indices of `index` (saves a second lookup
    on large inputs). Indices are range-checked; everything else is identical."""
    src = np.asarray(src_idx)
    dst = np.asarray(dst_idx)
    counts = np.asarray(synapse_counts)
    n_rows = int(src.size)
    if src.shape != (n_rows,) or dst.shape != (n_rows,) or counts.shape != (n_rows,):
        raise GraphBuildError("src, dst and synapse_counts must be 1-D and the same length")
    if n_rows:
        if not (np.issubdtype(src.dtype, np.integer) and np.issubdtype(dst.dtype, np.integer)):
            raise GraphBuildError("src/dst indices must be integers")
        lo, hi = min(int(src.min()), int(dst.min())), max(int(src.max()), int(dst.max()))
        if lo < 0 or hi >= index.n:
            raise DanglingIdError(f"edge index out of range [0, {index.n}): min {lo}, max {hi}")
    _check_counts(counts)
    counts = counts.astype(np.int64, copy=False)

    if signs is None:
        sign = np.full(n_rows, SIGN_UNKNOWN, dtype=np.int8)
    else:
        sign_in = np.asarray(signs)
        if sign_in.shape != (n_rows,):
            raise GraphBuildError("signs must be 1-D and the same length as the edge rows")
        if n_rows and not np.isin(sign_in, _VALID_SIGNS).all():
            raise GraphBuildError("signs must be -1 (inhibitory), 0 (unknown) or +1 (excitatory)")
        sign = sign_in.astype(np.int8)

    src = src.astype(np.int32, copy=False)
    dst = dst.astype(np.int32, copy=False)
    order = np.lexsort((dst, src))  # primary key src, secondary dst
    src, dst, counts, sign = src[order], dst[order], counts[order], sign[order]

    # starts_mask[i] is True where row i begins a new (src, dst) pair.
    starts_mask = np.ones(n_rows, dtype=bool)
    starts_mask[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    starts = np.flatnonzero(starts_mask)
    n_dup = n_rows - int(starts.size)

    if n_dup:
        if duplicate_rule is DuplicateRule.ERROR:
            dup_rows = np.flatnonzero(~starts_mask)[:5]
            examples = [
                (int(index.body_ids[src[i]]), int(index.body_ids[dst[i]])) for i in dup_rows
            ]
            raise DuplicateEdgeError(
                f"{n_dup} repeated (pre, post) row(s), e.g. {examples}; "
                "pass DuplicateRule.SUM to aggregate explicitly"
            )
        sign_lo = np.minimum.reduceat(sign, starts)
        sign_hi = np.maximum.reduceat(sign, starts)
        if (sign_lo != sign_hi).any():
            raise SignConflictError("repeated (pre, post) rows disagree on sign")
        counts = np.add.reduceat(counts, starts)
        src, dst, sign = src[starts], dst[starts], sign_lo

    graph = DirectedGraph(
        index=index,
        src=_readonly(np.ascontiguousarray(src, dtype=np.int32)),
        dst=_readonly(np.ascontiguousarray(dst, dtype=np.int32)),
        synapse_count=_readonly(np.ascontiguousarray(counts, dtype=np.int64)),
        sign=_readonly(np.ascontiguousarray(sign, dtype=np.int8)),
        duplicate_rule=duplicate_rule,
        dataset_id=dataset_id,
    )
    degree = np.bincount(graph.src, minlength=index.n) + np.bincount(graph.dst, minlength=index.n)
    report = BuildReport(
        dataset_id=dataset_id,
        orientation=ORIENTATION,
        duplicate_rule=duplicate_rule.value,
        n_neurons=index.n,
        n_input_rows=n_rows,
        n_unique_edges=graph.n_edges,
        n_duplicate_rows_merged=n_dup,
        total_synapses=graph.total_synapses,
        n_self_loops=int((graph.src == graph.dst).sum()),
        n_unknown_sign_edges=int((graph.sign == SIGN_UNKNOWN).sum()),
        n_isolated_neurons=int((degree == 0).sum()),
        min_index=int(min(graph.src.min(), graph.dst.min())) if graph.n_edges else None,
        max_index=int(max(graph.src.max(), graph.dst.max())) if graph.n_edges else None,
    )
    return graph, report
