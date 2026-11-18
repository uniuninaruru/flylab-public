"""Neural engines: NumPy reference LIF (`lif_numpy`) and PyTorch LIF on CPU or CUDA
(`lif_torch`), plus the lockstep comparison of two engines (`parity`).

The numerical scheme is specified in docs/MODEL_ASSUMPTIONS.md section 2.1; what may differ on
CUDA and how that is checked is section 2.3.

Boundary: may import flylab.data and flylab.provenance; must not import flylab.api,
flylab.embodiment, web frameworks, SSH libraries or MuJoCo. torch is imported only by
`lif_torch`, and only if installed.
"""

from flylab.neural.engine import (
    NeuralEngine,
    NeuralStep,
    RunSummary,
    StepDrive,
    StepInterventions,
)
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import SynapticNetwork, build_network
from flylab.neural.params import PARAMETER_SETS, SHIU2024_CODE_V1, LIFParameters, discretize

__all__ = [
    "PARAMETER_SETS",
    "SHIU2024_CODE_V1",
    "LIFParameters",
    "NeuralEngine",
    "NeuralStep",
    "NumpyLIFEngine",
    "RunSummary",
    "StepDrive",
    "StepInterventions",
    "SynapticNetwork",
    "build_network",
    "discretize",
]
