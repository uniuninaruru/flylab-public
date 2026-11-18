"""Evidence labels (spec section 1.3).

Every dataset field, model parameter, mapping and claim carries exactly one label.
`Labeled` is frozen: a value can be replaced only by building a new `Labeled` with its own
label and justification, never by silently mutating it from a UI or a convenience constant.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Evidence(enum.Enum):
    """The six evidence labels from spec section 1.3. Do not add members without a spec change."""

    EMPIRICAL = "EMPIRICAL"
    LITERATURE_DERIVED = "LITERATURE_DERIVED"
    ASSUMED = "ASSUMED"
    ENGINEERED = "ENGINEERED"
    LEARNED = "LEARNED"
    UNKNOWN = "UNKNOWN"

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS: dict[Evidence, str] = {
    Evidence.EMPIRICAL: "Directly from a measured/published dataset",
    Evidence.LITERATURE_DERIVED: "Adopted from a published method, with citation and version",
    Evidence.ASSUMED: "Chosen without direct measurement for the target neuron",
    Evidence.ENGINEERED: "Software interface or arbitrary transform",
    Evidence.LEARNED: "Fit by a specified optimization on specified data",
    Evidence.UNKNOWN: "Not established",
}

# Labels that only make sense when we can say where the value came from.
_NEEDS_SOURCE = frozenset({Evidence.EMPIRICAL, Evidence.LITERATURE_DERIVED, Evidence.LEARNED})
# Labels that only make sense when we can say why the value was chosen.
_NEEDS_RATIONALE = frozenset({Evidence.ASSUMED, Evidence.ENGINEERED})


@dataclass(frozen=True)
class Labeled[T]:
    """A value plus its single evidence label.

    Attributes:
        value: The value itself.
        evidence: Exactly one `Evidence` label.
        source: Citation, dataset id + file, or fitting run id. Required for EMPIRICAL,
            LITERATURE_DERIVED and LEARNED.
        rationale: Why the value was chosen. Required for ASSUMED and ENGINEERED.
        units: Physical units (for example "ms", "mV"), or None if dimensionless.
    """

    value: T
    evidence: Evidence
    source: str | None = None
    rationale: str | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, Evidence):
            raise TypeError(f"evidence must be an Evidence member, got {self.evidence!r}")
        if self.evidence in _NEEDS_SOURCE and not (self.source and self.source.strip()):
            raise ValueError(f"{self.evidence.value} values need a non-empty `source`")
        if self.evidence in _NEEDS_RATIONALE and not (self.rationale and self.rationale.strip()):
            raise ValueError(f"{self.evidence.value} values need a non-empty `rationale`")

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly form for manifests and API payloads."""
        return {
            "value": self.value,
            "evidence": self.evidence.value,
            "source": self.source,
            "rationale": self.rationale,
            "units": self.units,
        }
