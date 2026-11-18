"""NumPy reference LIF engine. Written for reading, not speed.

Implements docs/MODEL_ASSUMPTIONS.md section 2.1 line by line; the phase numbers in the
comments refer to that section.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from flylab.neural.engine import NEVER, EngineState, LIFEngineBase, _Advanced
from flylab.neural.network import SynapticNetwork
from flylab.neural.params import Discretization


def outgoing_edges(row_ptr: NDArray[np.int64], pre: NDArray[np.int32]) -> NDArray[np.int64]:
    """Edge ids of every outgoing edge of the neurons in `pre` (CSR row gather)."""
    if pre.size == 0:
        return np.zeros(0, dtype=np.int64)
    starts = row_ptr[pre]
    counts = row_ptr[pre + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    # Offset of each edge inside its row, added to the row start.
    first_out = np.cumsum(counts) - counts
    return np.repeat(starts - first_out, counts) + np.arange(total, dtype=np.int64)


class NumpyLIFEngine(LIFEngineBase):
    backend = "numpy"

    def __init__(
        self,
        network: SynapticNetwork,
        disc: Discretization,
        dtype: str = "float64",
        populations: Mapping[str, NDArray[np.integer]] | None = None,
    ) -> None:
        super().__init__(network, disc, dtype, populations)
        self._dt = np.dtype(dtype)
        t = self._dt.type
        # Propagator constants rounded once to the state dtype (computed in float64).
        self._a, self._b, self._c, self._d = t(disc.a), t(disc.b), t(disc.c), t(disc.d)
        self._v_rest, self._v_reset = t(disc.v_rest), t(disc.v_reset)
        self._v_th = t(disc.v_threshold)
        self._zero = t(0.0)
        self._v = np.empty(0, dtype=self._dt)
        self._g = np.empty(0, dtype=self._dt)
        self._last = np.empty(0, dtype=np.int64)
        self._ring: list[NDArray[np.int32]] = []
        self._candidate_v: NDArray[np.floating] | None = None

    def _reset_state(self) -> None:
        self._candidate_v = None
        self._v = np.full(self.n, self._v_rest, dtype=self._dt)
        self._g = np.zeros(self.n, dtype=self._dt)
        self._last = np.full(self.n, NEVER, dtype=np.int64)
        self._ring = [np.zeros(0, dtype=np.int32) for _ in range(self.disc.delay_steps)]

    def _mask(self, idx: NDArray[np.int32]) -> NDArray[np.bool_]:
        m = np.zeros(self.n, dtype=bool)
        m[idx] = True
        return m

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        net, v, g = self.network, self._v, self._g
        clamp = self._mask(clamp_idx)

        # 1. Refractory status at the start of the step.
        refractory = (n - self._last) < self.disc.refractory_steps
        active = ~refractory & ~clamp

        # 2. Exact subthreshold update for active neurons; inactive neurons are frozen.
        v_new = self._v_rest + (v - self._v_rest) * self._a + g * self._b
        if current_idx.size:  # the e*c term is skipped (not added as zeros) when there is no drive
            e = np.zeros(self.n, dtype=np.float64)
            np.add.at(e, current_idx, current_mV)
            v_new = v_new + e.astype(self._dt) * self._c
        self._candidate_v = v_new
        g_new = g * self._d
        np.copyto(v, v_new, where=active)
        np.copyto(g, g_new, where=active)

        # 3. Spikes: threshold crossings plus accepted forced spikes.
        forced_req = self._mask(forced_idx)
        crossed = active & (v > self._v_th)
        forced_ok = active & forced_req
        spiking = crossed | forced_ok
        spike_idx = np.flatnonzero(spiking).astype(np.int32)
        n_dropped = int((forced_req & ~active).sum())
        self._last[spike_idx] = n

        # 4. Arrivals of spikes emitted D steps ago, summed in float64, rounded once.
        slot = n % self.disc.delay_steps
        arriving = self._ring[slot]
        if arriving.size:
            eid = outgoing_edges(net.row_ptr, arriving)
            if eid.size:
                incoming = np.bincount(
                    net.dst[eid], weights=net.weight_mV[eid], minlength=self.n
                ).astype(self._dt)
                receives = ~refractory & ~spiking & ~clamp
                np.copyto(g, g + incoming, where=receives)

        # 5. Reset of spiking neurons; clamped neurons held at rest.
        v[spike_idx] = self._v_reset
        g[spike_idx] = self._zero
        v[clamp] = self._v_rest
        g[clamp] = self._zero

        # 6. Queue spikes whose output is not blocked; they arrive at step n + D.
        propagated = ~self._mask(block_idx)[spike_idx]
        self._ring[slot] = spike_idx[propagated]
        return _Advanced(
            spike_idx=spike_idx,
            forced=forced_ok[spike_idx],
            propagated=propagated,
            n_forced_dropped=n_dropped,
        )

    def state(self) -> EngineState:
        return EngineState(
            step=self._step_index,
            v=self._v.copy(),
            g=self._g.copy(),
            last_spike=self._last.copy(),
            in_flight=tuple(a.copy() for a in self._ring),
        )

    def candidate_potentials(self, neurons: NDArray[np.int64]) -> NDArray[np.float64]:
        if self._candidate_v is None:
            raise RuntimeError("candidate potentials require a completed step")
        return self._candidate_v[neurons].astype(np.float64, copy=True)

    def _load_state(self, state: EngineState) -> None:
        if state.v.shape != (self.n,) or state.g.shape != (self.n,):
            raise ValueError("state vectors do not match the network size")
        self._v = state.v.astype(self._dt, copy=True)
        self._g = state.g.astype(self._dt, copy=True)
        self._last = state.last_spike.astype(np.int64, copy=True)
        self._ring = [a.astype(np.int32, copy=True) for a in state.in_flight]
        self._candidate_v = None

    def _state_finite(self) -> bool:
        return bool(
            np.isfinite(self._v).all()
            and np.isfinite(self._g).all()
            and (self._candidate_v is None or np.isfinite(self._candidate_v).all())
        )

    def retained_bytes(self) -> dict[str, int]:
        return {
            "edges": 0,  # reads the network's arrays in place
            "state": int(
                self._v.nbytes
                + self._g.nbytes
                + self._last.nbytes
                + (0 if self._candidate_v is None else self._candidate_v.nbytes)
            ),
            "in_flight": int(sum(a.nbytes for a in self._ring)),
        }
