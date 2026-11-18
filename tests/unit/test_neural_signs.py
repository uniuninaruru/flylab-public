"""Transmitter -> sign mapping and unknown-sign policies (docs/MODEL_ASSUMPTIONS.md section 4)."""

from __future__ import annotations

import numpy as np
import pytest

from flylab.data.graph import (
    SIGN_UNKNOWN,
    DirectedGraph,
    UnknownSignError,
    UnknownSignPolicy,
    build_graph,
)
from flylab.neural.network import build_network
from flylab.neural.signs import (
    FAST_IONOTROPIC_V1,
    NULL_KEY,
    SHIU2024_METHODS_V1,
    SignMappingError,
    SignSource,
    SignSpec,
    neuron_signs,
    resolve_signs,
)
from flylab.provenance.labels import Evidence

# Synthetic neurons 1..6 with one transmitter each (made-up values for the test).
NT = ["acetylcholine", "gaba", "glutamate", "dopamine", None, "unclear"]


def _graph() -> DirectedGraph:
    ids = [1, 2, 3, 4, 5, 6]
    pre = [1, 2, 3, 4, 5, 6, 1]
    post = [2, 3, 4, 5, 6, 1, 3]
    g, _ = build_graph(ids, pre, post, [3, 2, 4, 1, 5, 2, 1], dataset_id="test:nt")
    return g


def _table(report: dict[str, object], key: str) -> dict[str, int]:
    t = report[key]
    assert isinstance(t, dict)
    return t


def _spec(
    unknown_sign_policy: UnknownSignPolicy = UnknownSignPolicy.EXCLUDE,
    transmitter_column: str | None = "predicted_nt",
    mapping: str | None = "fast_ionotropic_v1",
    overrides: dict[str, int] | None = None,
) -> SignSpec:
    return SignSpec(
        SignSource.TRANSMITTER, unknown_sign_policy, transmitter_column, mapping, overrides or {}
    )


def test_preset_table_is_explicit_and_assumed() -> None:
    m = FAST_IONOTROPIC_V1
    assert m.evidence is Evidence.ASSUMED
    assert m.signs["acetylcholine"] == 1
    assert m.signs["gaba"] == m.signs["glutamate"] == m.signs["histamine"] == -1
    for modulatory in ("dopamine", "serotonin", "octopamine", "unclear", NULL_KEY):
        assert m.signs[modulatory] == SIGN_UNKNOWN
    assert set(m.rationale) == set(m.signs)


def test_neuron_signs_and_counts() -> None:
    signs, counts = neuron_signs(NT, FAST_IONOTROPIC_V1.signs)
    assert signs.tolist() == [1, -1, -1, 0, 0, 0]
    assert counts == {
        NULL_KEY: 1,
        "acetylcholine": 1,
        "dopamine": 1,
        "gaba": 1,
        "glutamate": 1,
        "unclear": 1,
    }


def test_unmapped_value_is_refused_not_guessed() -> None:
    with pytest.raises(SignMappingError, match="tyramine"):
        neuron_signs(["acetylcholine", "tyramine"], FAST_IONOTROPIC_V1.signs)


def test_edge_sign_is_presynaptic_sign_and_graph_is_untouched() -> None:
    g = _graph()
    signed, report = resolve_signs(g, _spec(), np.array(NT, dtype=object))
    pre_sign = {1: 1, 2: -1, 3: -1, 4: 0, 5: 0, 6: 0}
    for s, src in zip(signed.sign, signed.src, strict=True):
        assert s == pre_sign[int(signed.index.body_ids[src])]
    assert (g.sign == SIGN_UNKNOWN).all()
    assert report["edges_per_sign"] == {"excitatory": 2, "inhibitory": 2, "unknown": 3}
    assert _table(report, "final_table")["glutamate"] == -1
    assert report["unknown_sign_policy"] == "exclude"
    assert report["transmitter_column"] == "predicted_nt"


def test_sensitivity_override() -> None:
    _, report = resolve_signs(
        _graph(),
        _spec(overrides={"glutamate": 1, "dopamine": 1}),
        np.array(NT, dtype=object),
    )
    assert report["edges_per_sign"] == {"excitatory": 4, "inhibitory": 1, "unknown": 2}
    assert _table(report, "preset_table")["glutamate"] == -1
    assert _table(report, "final_table")["glutamate"] == 1


@pytest.mark.parametrize(
    ("policy", "n_edges", "n_unknown_left"),
    [
        (UnknownSignPolicy.EXCLUDE, 4, 0),
        (UnknownSignPolicy.AS_EXCITATORY, 7, 0),
        (UnknownSignPolicy.AS_INHIBITORY, 7, 0),
    ],
)
def test_policies_when_building_the_network(
    policy: UnknownSignPolicy, n_edges: int, n_unknown_left: int
) -> None:
    signed, _ = resolve_signs(
        _graph(), _spec(unknown_sign_policy=policy), np.array(NT, dtype=object)
    )
    net = build_network(signed, policy, 0.5)
    assert net.n_edges == n_edges
    assert int((net.weight_mV == 0).sum()) == n_unknown_left
    assert net.n_unknown_sign_edges == 3
    if policy is UnknownSignPolicy.AS_INHIBITORY:
        assert int((net.weight_mV < 0).sum()) == 5


def test_error_policy_refuses_unknown_signs() -> None:
    signed, _ = resolve_signs(_graph(), _spec(), np.array(NT, dtype=object))
    with pytest.raises(UnknownSignError):
        build_network(signed, UnknownSignPolicy.ERROR, 0.5)


def test_spec_validation() -> None:
    with pytest.raises(SignMappingError, match="transmitter_column"):
        _spec(transmitter_column=None)
    with pytest.raises(SignMappingError, match="unknown mapping"):
        _spec(mapping="nope")
    with pytest.raises(SignMappingError, match="only apply"):
        SignSpec(SignSource.GRAPH, UnknownSignPolicy.ERROR, transmitter_column="x")
    with pytest.raises(SignMappingError, match="-1, 0 or \\+1"):
        _spec(overrides={"gaba": 2})


def test_graph_source_keeps_stored_signs() -> None:
    ids = [1, 2]
    g, _ = build_graph(ids, [1], [2], [4], [-1], dataset_id="t")
    signed, report = resolve_signs(g, SignSpec(SignSource.GRAPH, UnknownSignPolicy.ERROR))
    assert signed is g
    assert report["edges_per_sign"] == {"excitatory": 0, "inhibitory": 1, "unknown": 0}


def test_shiu_methods_preset_follows_the_paper() -> None:
    m = SHIU2024_METHODS_V1
    # Methods: GABA/Glut inhibitory; DA, OA, 5-HT "assigned to the excitatory category".
    assert m.signs["gaba"] == m.signs["glutamate"] == -1
    for t in ("acetylcholine", "dopamine", "serotonin", "octopamine"):
        assert m.signs[t] == 1
        assert m.label_of(t) is Evidence.LITERATURE_DERIVED
    for t in ("histamine", "unclear", NULL_KEY):  # not covered by the paper
        assert m.signs[t] == SIGN_UNKNOWN
        assert m.label_of(t) is Evidence.ASSUMED
    assert m.evidence is Evidence.ASSUMED  # the table as a whole: its weakest row
    assert m.source is not None
    assert "Methods" in m.source


def test_report_labels_rows_and_overrides() -> None:
    _, report = resolve_signs(
        _graph(),
        _spec(mapping="shiu2024_methods_v1", overrides={"histamine": -1}),
        np.array(NT, dtype=object),
    )
    rows = report["row_evidence"]
    assert isinstance(rows, dict)
    assert rows["dopamine"] == "LITERATURE_DERIVED"
    assert rows["histamine"] == "ASSUMED (override)"
    assert rows["unclear"] == "ASSUMED"
    # dopamine neuron 4 is excitatory under this preset: 4 -> 5 edge is now +1.
    assert report["edges_per_sign"] == {"excitatory": 3, "inhibitory": 2, "unknown": 2}
