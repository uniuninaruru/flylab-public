"""Reproduce the parity pass-rate experiments of docs/audit/PR-03-review.md (findings 1 and 2).

Run from the repository root:  uv run python docs/audit/repro/pr03_parity_experiments.py

Two NumPy engines run in lockstep (flylab.neural.parity.lockstep, default margins) on 40
synthetic random graphs (300 neurons, mean out-degree 30, w = 0.6 mV, seeds 0..39) with noisy
injected drive, 300 steps of 0.1 ms. The reference is the real NumpyLIFEngine; the "test"
engine carries one deliberate bug. Printed: how many of the 40 runs come out ok=True.

  A  threshold error: the test engine's V_th is higher by `shift` mV (drive U(4, 14) mV)
  B  drive error: the test engine multiplies the injected drive by `gain` (drive U(14, 24) mV,
     so that the first threshold crossings come before the first state check at step 100)

SYNTHETIC graphs only; the numbers characterise the comparison policy, not MaleCNS.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import StepDrive, StepInterventions, _Advanced
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize
from flylab.neural.parity import ParityReport, lockstep

DISC = discretize(SHIU2024_CODE_V1, 0.1)
SEEDS = range(40)
STEPS = 300


class _Schedule:
    def __init__(self, n: int, seed: int, lo: float, hi: float) -> None:
        rng = np.random.default_rng(seed)
        base = rng.uniform(lo, hi, n)
        idx = np.arange(n, dtype=np.int32)
        self._steps = [StepDrive(idx, base + rng.normal(0, 2, n)) for _ in range(STEPS)]

    def inputs(self, step: int) -> tuple[StepDrive | None, StepInterventions | None]:
        return self._steps[step], None


class _DriveGain(NumpyLIFEngine):
    gain = 1.0

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        return super()._advance(
            n, current_idx, current_mV * self.gain, forced_idx, block_idx, clamp_idx
        )


def _run(dtype: str, seed: int, *, shift: float = 0.0, gain: float = 1.0,
         drive: tuple[float, float] = (4.0, 14.0)) -> ParityReport:  # fmt: skip
    g, _ = random_graph(300, 30, seed=seed)
    net = build_network(g, UnknownSignPolicy.ERROR, 0.6)
    ref = NumpyLIFEngine(net, DISC, dtype)
    disc = dataclasses.replace(DISC, v_threshold=DISC.v_threshold + shift)
    _DriveGain.gain = gain
    test = _DriveGain(net, disc, dtype)
    ref.reset(1)
    test.reset(1)
    return lockstep(ref, test, STEPS, _Schedule(net.n_neurons, seed, *drive))


def _line(label: str, reps: list[ParityReport]) -> str:
    ok = [r for r in reps if r.ok]
    flips = sorted(r.divergence.step for r in ok if r.divergence is not None)
    checks = sorted({r.state_checks for r in ok})
    statuses = ", ".join(sorted({r.status for r in reps}))
    span = f"{flips[0]}-{flips[-1]}" if flips else "-"
    return (f"{label:38s} ok={len(ok):2d}/{len(reps)}  first flip of passing runs at step {span:9s}"
            f" state checks of passing runs {checks}  statuses: {statuses}")  # fmt: skip


def main() -> None:
    print(f"lockstep, default margins, {STEPS} steps x {DISC.dt_ms} ms, seeds 0..{SEEDS[-1]}")
    for dtype, shifts in (("float32", (0.003, 0.02, 0.05, 0.2)),
                          ("float64", (3e-7, 2e-6, 1e-5, 1e-4))):  # fmt: skip
        for s in shifts:
            reps = [_run(dtype, seed, shift=s) for seed in SEEDS]
            print(_line(f"A {dtype} V_th + {s:g} mV", reps))
    for gain in (1.002, 1.01, 1.05):
        reps = [_run("float32", seed, gain=gain, drive=(14.0, 24.0)) for seed in SEEDS]
        print(_line(f"B float32 drive x {gain:g}", reps))


if __name__ == "__main__":
    main()
