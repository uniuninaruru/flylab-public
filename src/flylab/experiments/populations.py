"""Resolve named populations against the loaded dataset (second validation stage).

A population that does not exist in the data is an error, never an empty selection
(spec 8.1 warning): unknown body ids, unknown columns and values that match no neuron all fail.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from flylab.data.graph import DanglingIdError
from flylab.experiments.config import ALL_POPULATION, ConfigError, PopulationConfig
from flylab.experiments.dataset import LoadedDataset


def resolve_population(pop: PopulationConfig, data: LoadedDataset) -> NDArray[np.int32]:
    q = f"populations.{pop.name}"
    if pop.body_ids is not None:
        try:
            idx = data.graph.index.indices_of(np.array(pop.body_ids, dtype=np.int64))
        except DanglingIdError as exc:
            raise ConfigError(f"{q}.body_ids: {exc}") from exc
        return np.sort(idx).astype(np.int32)
    if pop.column is None or pop.values is None:
        raise ConfigError(f"{q}: needs body_ids, or column and values")
    if pop.column not in data.neurons.columns:
        raise ConfigError(
            f"{q}.column: {pop.column!r} is not a neuron-table column; "
            f"available: {list(data.neurons.columns)}"
        )
    col = data.neurons.column(pop.column)
    as_text = np.array([None if v is None else str(v) for v in col], dtype=object)
    selected = np.zeros(col.size, dtype=bool)
    for value in pop.values:
        hit = as_text == value
        if not hit.any():
            raise ConfigError(
                f"{q}.values: {value!r} matches no neuron in column {pop.column!r} "
                "(placeholders and unverified names are rejected)"
            )
        selected |= hit
    return np.flatnonzero(selected).astype(np.int32)


def resolve_populations(
    pops: Sequence[PopulationConfig], data: LoadedDataset
) -> dict[str, NDArray[np.int32]]:
    """Index arrays (ascending) for every population, plus `__all__`."""
    out = {p.name: resolve_population(p, data) for p in pops}
    out[ALL_POPULATION] = np.arange(data.graph.n_neurons, dtype=np.int32)
    return out
