"""Null-model graphs for controls (PR-04). NOT biological data.

`degree_preserving_swap` keeps every edge's presynaptic neuron, synapse count and sign, and
permutes the postsynaptic ends among all edges. Consequences, checked by tests and recorded in
the report:

- out-degree, out-strength (synapses per presynaptic neuron) and the sign of every neuron's
  output are **exactly** preserved (sign is a property of the presynaptic neuron, section 4 of
  docs/MODEL_ASSUMPTIONS.md);
- in-degree is **exactly** preserved (a permutation of the `dst` column);
- in-strength (synapses per postsynaptic neuron) and who-connects-to-whom are **not** preserved;
- the result has no self-loops and no repeated (pre, post) pairs.

Algorithm: a uniform random permutation of `dst`, then repair rounds. In each round every
self-loop or repeated pair is swapped (its `dst`) with a uniformly chosen edge outside the bad
set; swaps keep both degree sequences. The loop stops when nothing is bad, or raises after
`max_rounds`. This is a configuration-model-style sample with rejection repair; it is not proven
to sample uniformly from all graphs with these degrees (Maslov-Sneppen chains would need ~10 E
swaps, too slow for 25 M edges in NumPy). The choice is ENGINEERED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flylab.data.graph import DirectedGraph, DuplicateRule, build_graph_indexed

NULL_METHODS = ("degree_preserving_swap",)
DEFAULT_MAX_ROUNDS = 200


class NullModelError(RuntimeError):
    """The null graph could not be built without self-loops or repeated pairs."""


@dataclass(frozen=True)
class NullModelReport:
    method: str
    seed: int
    n_edges: int
    rounds: int
    n_repair_swaps: int
    original_fingerprint_sha256: str
    null_fingerprint_sha256: str
    original_self_loops: int
    fraction_pairs_kept: float  # share of (pre, post) pairs that also exist in the original
    in_degree_preserved: bool
    out_degree_preserved: bool
    out_strength_preserved: bool
    in_strength_preserved: bool
    in_strength_pearson_r: float | None
    evidence: str = "ENGINEERED"

    def to_dict(self) -> dict[str, object]:
        d = dict(self.__dict__)
        d["preserves"] = ["in-degree", "out-degree", "out-strength", "presynaptic sign"]
        d["does_not_preserve"] = ["in-strength", "pair identity", "self-loops (none in null)"]
        d["rationale"] = (
            "control graph: permute postsynaptic ends with collision repair "
            "(flylab.data.null_models); not a uniform sample proof"
        )
        return d


def _pair_keys(src: NDArray[np.int32], dst: NDArray[np.int32], n: int) -> NDArray[np.int64]:
    return src.astype(np.int64) * np.int64(n) + dst.astype(np.int64)


def _bad_edges(src: NDArray[np.int32], dst: NDArray[np.int32], n: int) -> NDArray[np.int64]:
    """Self-loops plus every repeated (pre, post) pair except its first occurrence."""
    keys = _pair_keys(src, dst, n)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    dup = np.zeros(keys.size, dtype=bool)
    if keys.size > 1:
        dup[order[1:][sk[1:] == sk[:-1]]] = True
    return np.flatnonzero(dup | (src == dst)).astype(np.int64)


def degree_preserving_swap(
    graph: DirectedGraph, seed: int, max_rounds: int = DEFAULT_MAX_ROUNDS
) -> tuple[DirectedGraph, NullModelReport]:
    """Shuffled control graph with the same in/out-degree sequences (see module docstring)."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    n, e = graph.n_neurons, graph.n_edges
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6E756C6C]))  # "null"
    src = graph.src
    dst = graph.dst[rng.permutation(e)].astype(np.int32)
    swaps = 0
    rounds = 0
    while True:
        bad = _bad_edges(src, dst, n)
        if bad.size == 0:
            break
        if rounds >= max_rounds:
            raise NullModelError(
                f"{bad.size} self-loop/repeated edge(s) left after {max_rounds} repair rounds"
            )
        if e - bad.size < bad.size:
            raise NullModelError("graph too dense for collision repair (bad edges > half)")
        rounds += 1
        ok = np.ones(e, dtype=bool)
        ok[bad] = False
        pool = np.flatnonzero(ok)
        partners = pool[rng.choice(pool.size, size=bad.size, replace=False)]
        tmp = dst[bad].copy()
        dst[bad] = dst[partners]
        dst[partners] = tmp
        swaps += int(bad.size)

    null, _ = build_graph_indexed(
        graph.index,
        src,
        dst,
        graph.synapse_count,
        graph.sign,
        dataset_id=f"{graph.dataset_id}+null:degree_preserving_swap:seed={seed}",
        duplicate_rule=DuplicateRule.ERROR,
    )
    orig_keys = np.sort(_pair_keys(graph.src, graph.dst, n))
    null_keys = _pair_keys(null.src, null.dst, n)
    kept = float(np.isin(null_keys, orig_keys, assume_unique=True).mean()) if e else 0.0

    def deg(a: NDArray[np.int32]) -> NDArray[np.int64]:
        return np.bincount(a, minlength=n)

    def strength(a: NDArray[np.int32], g: DirectedGraph) -> NDArray[np.int64]:
        return np.bincount(a, weights=g.synapse_count, minlength=n).astype(np.int64)

    in_o, in_n = strength(graph.dst, graph), strength(null.dst, null)
    r = None
    if e and in_o.std() > 0 and in_n.std() > 0:
        r = float(np.corrcoef(in_o, in_n)[0, 1])
    report = NullModelReport(
        method="degree_preserving_swap",
        seed=seed,
        n_edges=e,
        rounds=rounds,
        n_repair_swaps=swaps,
        original_fingerprint_sha256=graph.fingerprint(),
        null_fingerprint_sha256=null.fingerprint(),
        original_self_loops=int((graph.src == graph.dst).sum()),
        fraction_pairs_kept=kept,
        in_degree_preserved=bool(np.array_equal(deg(graph.dst), deg(null.dst))),
        out_degree_preserved=bool(np.array_equal(deg(graph.src), deg(null.src))),
        out_strength_preserved=bool(
            np.array_equal(strength(graph.src, graph), strength(null.src, null))
        ),
        in_strength_preserved=bool(np.array_equal(in_o, in_n)),
        in_strength_pearson_r=r,
    )
    return null, report


def make_null_graph(
    graph: DirectedGraph, method: str, seed: int
) -> tuple[DirectedGraph, NullModelReport]:
    if method not in NULL_METHODS:
        raise ValueError(f"unknown null model {method!r}; known: {list(NULL_METHODS)}")
    return degree_preserving_swap(graph, seed)
