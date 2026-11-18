"""Transmitter -> synapse sign, applied per presynaptic neuron (docs/MODEL_ASSUMPTIONS.md 4).

The processed graph from PR-01 stores sign 0 (unknown) for every MaleCNS edge. Signs are
assigned here, in memory, from an explicit named mapping; the processed files are never changed.
Values that the mapping does not list are refused instead of guessed.
"""

from __future__ import annotations

import enum
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from flylab.data.graph import (
    SIGN_EXCITATORY,
    SIGN_INHIBITORY,
    SIGN_UNKNOWN,
    DirectedGraph,
    UnknownSignPolicy,
)
from flylab.provenance.labels import Evidence

#: PR-01 transmitter columns in neurons.parquet (any other existing column is also accepted).
TRANSMITTER_COLUMNS = ("predicted_nt", "consensus_nt", "celltype_predicted_nt", "ground_truth")

#: Key used for missing (null) values in mappings, counts and manifests.
NULL_KEY = "<null>"

_VALID = (SIGN_INHIBITORY, SIGN_UNKNOWN, SIGN_EXCITATORY)


class SignSource(enum.Enum):
    GRAPH = "graph"  # keep the signs stored in the graph (fixtures, synthetic graphs)
    TRANSMITTER = "transmitter"  # map a neuron-table transmitter column


class SignMappingError(ValueError):
    """The mapping is malformed, or the data has a transmitter value the mapping does not list."""


@dataclass(frozen=True)
class SignMapping:
    """A transmitter -> sign table. `evidence` labels the table as a whole (its weakest row);
    `row_evidence` labels individual rows where they differ."""

    name: str
    signs: Mapping[str, int]
    rationale: Mapping[str, str]
    evidence: Evidence = Evidence.ASSUMED
    row_evidence: Mapping[str, Evidence] = field(default_factory=dict)
    source: str | None = None

    def label_of(self, value: str) -> Evidence:
        return self.row_evidence.get(value, self.evidence)


#: Every row is ASSUMED; see the rationale table in docs/MODEL_ASSUMPTIONS.md section 4.
FAST_IONOTROPIC_V1 = SignMapping(
    name="fast_ionotropic_v1",
    signs={
        "acetylcholine": SIGN_EXCITATORY,
        "gaba": SIGN_INHIBITORY,
        "glutamate": SIGN_INHIBITORY,
        "histamine": SIGN_INHIBITORY,
        "dopamine": SIGN_UNKNOWN,
        "serotonin": SIGN_UNKNOWN,
        "octopamine": SIGN_UNKNOWN,
        "unclear": SIGN_UNKNOWN,
        NULL_KEY: SIGN_UNKNOWN,
    },
    rationale={
        "acetylcholine": "main fast excitatory transmitter of the fly CNS (nicotinic receptors)",
        "gaba": "fast inhibition through GABA-A-type chloride channels",
        "glutamate": (
            "largely inhibitory in the fly CNS via GluCl-alpha chloride channels; excitatory at "
            "the NMJ and some central synapses -> run a glutamate=+1 sensitivity variant"
        ),
        "histamine": "acts on histamine-gated chloride channels (photoreceptor output)",
        "dopamine": "modulatory, mainly metabotropic; sign depends on receptor",
        "serotonin": "modulatory, mainly metabotropic; sign depends on receptor",
        "octopamine": "modulatory, mainly metabotropic; sign depends on receptor",
        "unclear": "no usable prediction",
        NULL_KEY: "no prediction available",
    },
)

_SHIU_METHODS = (
    "Shiu et al. 2024, Nature 634:210-219, doi:10.1038/s41586-024-07763-9 (CC BY 4.0), Methods "
    "'Neurotransmitter predictions'"
)
_LD = Evidence.LITERATURE_DERIVED

#: The rule stated in the Shiu et al. (2024) Methods, applied to a neuron-level transmitter
#: value. Rows the paper does not cover (histamine, unclear, missing) are ASSUMED unknown.
SHIU2024_METHODS_V1 = SignMapping(
    name="shiu2024_methods_v1",
    signs={
        "acetylcholine": SIGN_EXCITATORY,
        "gaba": SIGN_INHIBITORY,
        "glutamate": SIGN_INHIBITORY,
        "dopamine": SIGN_EXCITATORY,
        "serotonin": SIGN_EXCITATORY,
        "octopamine": SIGN_EXCITATORY,
        "histamine": SIGN_UNKNOWN,
        "unclear": SIGN_UNKNOWN,
        NULL_KEY: SIGN_UNKNOWN,
    },
    rationale={
        "acetylcholine": "Methods: only GABA/Glut neurons are inhibitory; all others excitatory",
        "gaba": "Methods: 'We assume GABAergic and glutamatergic neurons are inhibitory'",
        "glutamate": "Methods: 'We assume GABAergic and glutamatergic neurons are inhibitory'",
        "dopamine": (
            "Methods: 'Neurons predicted to be dopaminergic, octopaminergic or serotonergic are "
            "assigned to the excitatory category'"
        ),
        "serotonin": "Methods: assigned to the excitatory category (see dopamine)",
        "octopamine": "Methods: assigned to the excitatory category (see dopamine)",
        "histamine": (
            "not covered: the Methods list only ACh, Glu, GABA, DA, OA and 5-HT (55% + 24% + "
            "14% + 7%); FlyLab leaves histamine unknown"
        ),
        "unclear": "not covered by the paper; FlyLab leaves it unknown",
        NULL_KEY: "not covered by the paper; FlyLab leaves it unknown",
    },
    evidence=Evidence.ASSUMED,  # weakest row
    row_evidence={
        k: _LD
        for k in ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine")
    },
    source=_SHIU_METHODS,
)

SIGN_MAPPINGS: dict[str, SignMapping] = {
    m.name: m for m in (FAST_IONOTROPIC_V1, SHIU2024_METHODS_V1)
}


@dataclass(frozen=True)
class SignSpec:
    """What the run config says about synapse signs. `unknown_sign_policy` has no default."""

    source: SignSource
    unknown_sign_policy: UnknownSignPolicy
    transmitter_column: str | None = None
    mapping: str | None = None
    overrides: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source is SignSource.TRANSMITTER:
            if not self.transmitter_column:
                raise SignMappingError("source 'transmitter' needs transmitter_column")
            if self.mapping not in SIGN_MAPPINGS:
                raise SignMappingError(
                    f"unknown mapping {self.mapping!r}; choose from {sorted(SIGN_MAPPINGS)}"
                )
        elif self.transmitter_column or self.mapping or self.overrides:
            raise SignMappingError(
                "transmitter_column, mapping and overrides only apply to source 'transmitter'"
            )
        for k, v in self.overrides.items():
            if isinstance(v, bool) or v not in _VALID:
                raise SignMappingError(f"override {k!r}: sign must be -1, 0 or +1, got {v!r}")

    def final_mapping(self) -> dict[str, int]:
        if self.source is not SignSource.TRANSMITTER or self.mapping is None:
            return {}
        return {**SIGN_MAPPINGS[self.mapping].signs, **self.overrides}

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "source": self.source.value,
            "unknown_sign_policy": self.unknown_sign_policy.value,
        }
        if self.source is SignSource.TRANSMITTER:
            out |= {
                "transmitter_column": self.transmitter_column,
                "mapping": self.mapping,
                "overrides": dict(self.overrides),
            }
        return out


def neuron_signs(
    values: Sequence[str | None] | NDArray[np.object_], mapping: Mapping[str, int]
) -> tuple[NDArray[np.int8], dict[str, int]]:
    """Sign of every neuron from its transmitter value; counts of neurons per value.

    Raises SignMappingError if a value is not in `mapping` (nothing is guessed).
    """
    keys = [NULL_KEY if v is None else str(v) for v in values]
    counts = Counter(keys)
    unmapped = sorted(k for k in counts if k not in mapping)
    if unmapped:
        examples = {k: counts[k] for k in unmapped[:10]}
        raise SignMappingError(
            f"transmitter value(s) not in the mapping (neurons per value): {examples}; "
            "add them to `overrides` explicitly"
        )
    lut = {k: np.int8(mapping[k]) for k in counts}
    signs = np.fromiter((lut[k] for k in keys), dtype=np.int8, count=len(keys))
    return signs, dict(sorted(counts.items()))


def _edge_sign_counts(sign: NDArray[np.int8]) -> dict[str, int]:
    return {
        "excitatory": int((sign == SIGN_EXCITATORY).sum()),
        "inhibitory": int((sign == SIGN_INHIBITORY).sum()),
        "unknown": int((sign == SIGN_UNKNOWN).sum()),
    }


def resolve_signs(
    graph: DirectedGraph,
    spec: SignSpec,
    transmitter_values: Sequence[str | None] | NDArray[np.object_] | None = None,
) -> tuple[DirectedGraph, dict[str, object]]:
    """Graph with signs assigned per `spec`, plus a JSON-ready report for the run manifest.

    Unknown-sign edges are left as SIGN_UNKNOWN here; `spec.unknown_sign_policy` is applied when
    the synaptic network is built (`flylab.neural.network.build_network`).
    """
    report: dict[str, object] = {**spec.to_dict()}
    if spec.source is SignSource.GRAPH:
        signed = graph
        report["note"] = "signs taken from the graph as stored"
    else:
        if transmitter_values is None:
            raise SignMappingError("source 'transmitter' needs the transmitter column values")
        if len(transmitter_values) != graph.n_neurons:
            raise SignMappingError(
                f"{len(transmitter_values)} transmitter values for {graph.n_neurons} neurons"
            )
        mapping = spec.final_mapping()
        per_neuron, neuron_counts = neuron_signs(transmitter_values, mapping)
        signed = replace(graph, sign=per_neuron[graph.src])
        signed.sign.setflags(write=False)
        assert spec.mapping is not None
        preset = SIGN_MAPPINGS[spec.mapping]
        report |= {
            "rule": "edge sign = sign of its presynaptic neuron (one sign per neuron, ASSUMED)",
            "mapping_evidence": preset.evidence.value,
            "mapping_source": preset.source,
            "row_evidence": {
                k: Evidence.ASSUMED.value + " (override)"
                if k in spec.overrides
                else preset.label_of(k).value
                for k in mapping
            },
            "preset_table": dict(preset.signs),
            "final_table": mapping,
            "rationale": dict(preset.rationale),
            "neurons_per_value": neuron_counts,
            "neurons_per_sign": {
                "excitatory": int((per_neuron == SIGN_EXCITATORY).sum()),
                "inhibitory": int((per_neuron == SIGN_INHIBITORY).sum()),
                "unknown": int((per_neuron == SIGN_UNKNOWN).sum()),
            },
        }
    report["edges_per_sign"] = _edge_sign_counts(signed.sign)
    report["graph_fingerprint_sha256_signed"] = signed.fingerprint()
    return signed, report
