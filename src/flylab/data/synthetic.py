"""Deterministic synthetic graphs for tests and benchmarks. NOT biological data.

Body ids start at SYNTHETIC_ID_BASE so they are easy to tell apart from real MaleCNS ids.
The per-neuron sign assignment is an ENGINEERED test device, not a transmitter prediction.
"""

from __future__ import annotations

import numpy as np

from flylab.data.graph import (
    SIGN_EXCITATORY,
    SIGN_INHIBITORY,
    BuildReport,
    DirectedGraph,
    DuplicateRule,
    build_graph,
)

SYNTHETIC_ID_BASE = 9_000_000_000_000


def random_graph(
    n_neurons: int,
    mean_out_degree: float,
    seed: int,
    inhibitory_fraction: float = 0.2,
) -> tuple[DirectedGraph, BuildReport]:
    """Uniform random directed graph with a fixed sign per presynaptic neuron.

    Same arguments always give the same graph (and the same `fingerprint()`).
    Repeated (pre, post) draws are summed (DuplicateRule.SUM); their signs agree because
    sign is a property of the presynaptic neuron here.
    """
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if mean_out_degree < 0:
        raise ValueError("mean_out_degree must be non-negative")
    if not 0.0 <= inhibitory_fraction <= 1.0:
        raise ValueError("inhibitory_fraction must be in [0, 1]")
    rng = np.random.default_rng(seed)
    n_rows = round(n_neurons * mean_out_degree)
    body_ids = SYNTHETIC_ID_BASE + np.arange(n_neurons, dtype=np.int64)
    pre = rng.integers(0, n_neurons, size=n_rows)
    post = rng.integers(0, n_neurons, size=n_rows)
    counts = rng.integers(1, 11, size=n_rows)
    neuron_sign = np.where(
        rng.random(n_neurons) < inhibitory_fraction, SIGN_INHIBITORY, SIGN_EXCITATORY
    ).astype(np.int8)
    return build_graph(
        body_ids,
        body_ids[pre],
        body_ids[post],
        counts,
        neuron_sign[pre],
        dataset_id=f"synthetic:random:v1:n={n_neurons}:k={mean_out_degree}:seed={seed}",
        duplicate_rule=DuplicateRule.SUM,
    )
