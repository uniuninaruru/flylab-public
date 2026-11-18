from __future__ import annotations

import dataclasses

import pytest

from flylab.provenance.labels import Evidence, Labeled

SPEC_LABELS = {"EMPIRICAL", "LITERATURE_DERIVED", "ASSUMED", "ENGINEERED", "LEARNED", "UNKNOWN"}


def test_exactly_the_six_spec_labels() -> None:
    assert {e.value for e in Evidence} == SPEC_LABELS
    assert all(e.description for e in Evidence)


@pytest.mark.parametrize(
    "label", [Evidence.EMPIRICAL, Evidence.LITERATURE_DERIVED, Evidence.LEARNED]
)
def test_sourced_labels_need_source(label: Evidence) -> None:
    with pytest.raises(ValueError, match="source"):
        Labeled(1.0, label)
    Labeled(1.0, label, source="doi:10.0000/example")


@pytest.mark.parametrize("label", [Evidence.ASSUMED, Evidence.ENGINEERED])
def test_chosen_labels_need_rationale(label: Evidence) -> None:
    with pytest.raises(ValueError, match="rationale"):
        Labeled(1.0, label, rationale="   ")
    Labeled(1.0, label, rationale="placeholder until PR-02 verifies the paper value")


def test_unknown_needs_nothing() -> None:
    assert Labeled(None, Evidence.UNKNOWN).to_dict()["evidence"] == "UNKNOWN"


def test_string_label_rejected() -> None:
    with pytest.raises(TypeError):
        Labeled(1.0, "EMPIRICAL", source="x")  # type: ignore[arg-type]


def test_labeled_values_are_frozen() -> None:
    v = Labeled(1.0, Evidence.ASSUMED, rationale="test", units="ms")
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.value = 2.0  # type: ignore[misc]
    assert v.to_dict() == {
        "value": 1.0,
        "evidence": "ASSUMED",
        "source": None,
        "rationale": "test",
        "units": "ms",
    }
