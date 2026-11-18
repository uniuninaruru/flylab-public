"""Motor decoders (spec 9.2). Only a placeholder exists so far.

`ZeroDecoder` ignores its inputs and returns the all-zero action. It is ENGINEERED, carries no
information from the neural simulation, and is **not** a motor map. Its uses:

- a test fixture for the decoder contract;
- later, the "no neural input" ablation that a closed-loop demo must be compared against
  (spec 9.3, item 5).

Heuristic, ridge (LEARNED) and policy decoders come after the neural and body gates pass
(spec 9.3); none is implemented here.
"""

from __future__ import annotations

import math

from flylab.embodiment.interface import (
    Action,
    ActionSpec,
    BodyObservation,
    NeuralFeatures,
    check_decoder_provenance,
)
from flylab.provenance.labels import Evidence, Labeled

ZERO_DECODER_NAME = "zero-v1"


class ZeroDecoder:
    """Returns the spec's zero action for every input. ENGINEERED placeholder, not a motor map."""

    def __init__(self, spec: ActionSpec) -> None:
        self._spec = spec
        self._zeros = spec.validate(spec.zeros())  # raises if 0 is outside any bound
        self._provenance = check_decoder_provenance(
            Labeled(
                value=ZERO_DECODER_NAME,
                evidence=Evidence.ENGINEERED,
                rationale=(
                    "Placeholder and null control: ignores neural features and observations and "
                    "returns the all-zero control vector, which lies inside every bound of the "
                    "runtime action spec. Not fitted to data; not a biological motor mapping."
                ),
            )
        )

    @property
    def provenance(self) -> Labeled[str]:
        return self._provenance

    @property
    def action_spec(self) -> ActionSpec:
        return self._spec

    def decode(
        self, neural_features: NeuralFeatures, observation: BodyObservation, time_ms: float
    ) -> Action:
        del neural_features, observation  # deliberately unused: this is the null decoder
        if not math.isfinite(time_ms):
            raise ValueError("time_ms must be finite")
        return Action(
            values=self._zeros,
            time_ms=time_ms,
            decoder=self._provenance.value,
            evidence=self._provenance.evidence,
        )
