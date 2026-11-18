"""PyTorch LIF engine on CPU (`torch-cpu`) or CUDA (`torch-cuda`): the math of `lif_numpy`,
phase by phase.

Propagation is a gather of the spiking neurons' CSR rows plus `index_add_` (spec 7.3 option 2).
Arrivals are summed in float64 and rounded once to the state dtype, as in the NumPy reference.
On CUDA the float64 additions inside `index_add_` happen in a different order than on the CPU,
so CUDA states can differ from the reference by a few float64 rounding errors (see
docs/MODEL_ASSUMPTIONS.md section 2.3); every other operation is the same IEEE-754 operation.

Memory is bounded: the edge arrays live on the device once (int32 targets + float64 weights,
12 bytes/edge, plus int64 row pointers), state is O(N), and the per-step edge temporaries are
processed in chunks of at most `max_edges_per_chunk` edges plus one neuron's out-degree.
Nothing is N x N and nothing grows with the number of steps.

CUDA results count as verified only from logs of the owner's RTX host (docs/acceptance/M3.md).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flylab.neural.engine import NEVER, EngineState, LIFEngineBase, _Advanced
from flylab.neural.network import SynapticNetwork
from flylab.neural.params import Discretization

_torch: Any
try:  # torch is optional: the `gpu` extra (CPU-only wheel on macOS); see pyproject.toml
    import torch as _torch_module

    _torch = _torch_module
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _torch = None

TORCH_BACKENDS = {"torch-cpu": "cpu", "torch-cuda": "cuda"}
#: Edge temporaries per chunk are about 40 bytes/edge (ids, gathered targets and weights),
#: so the default bounds them near 170 MB. Results do not depend on it (tests).
DEFAULT_EDGE_CHUNK = 1 << 22


class TorchUnavailableError(RuntimeError):
    """A torch backend was requested but cannot run in this process."""


class CudaUnavailableError(TorchUnavailableError):
    """`torch-cuda` was requested but torch sees no usable CUDA device."""


def torch_available() -> bool:
    return _torch is not None


def require_torch(backend: str = "torch-cpu") -> Any:
    if _torch is None:
        raise TorchUnavailableError(
            f"backend {backend!r} needs PyTorch: `uv sync --extra gpu` (on macOS this is the "
            "CPU-only wheel; see pyproject.toml)"
        )
    return _torch


def cuda_status() -> dict[str, object]:
    """Whether this process can run `torch-cuda`, with the reason when it cannot."""
    if _torch is None:
        return {"available": False, "reason": "torch is not installed"}
    t = _torch
    build = t.version.cuda
    if not t.cuda.is_available():
        return {
            "available": False,
            "torch_version": str(t.__version__),
            "torch_cuda_build": build,
            "reason": "CPU-only torch build" if build is None else "torch sees no CUDA device",
        }
    return {
        "available": True,
        "torch_version": str(t.__version__),
        "torch_cuda_build": build,
        "device_count": int(t.cuda.device_count()),
    }


def check_backend(backend: str) -> None:
    """Raise TorchUnavailableError if `backend` cannot run here (no-op for `numpy`)."""
    if backend not in TORCH_BACKENDS:
        return
    if not torch_available():
        require_torch(backend)  # raises with the install hint
        raise TorchUnavailableError(f"backend {backend!r} needs PyTorch, which is unavailable")
    if backend == "torch-cuda":
        status = cuda_status()
        if not status["available"]:
            raise CudaUnavailableError(
                f"backend 'torch-cuda' needs a CUDA device: {status['reason']} "
                f"(torch {status.get('torch_version')}, CUDA build "
                f"{status.get('torch_cuda_build')}). On the GPU host run it in the `gpu` "
                'Compose service; see README.md, "GPU host"'
            )


def torch_settings(device: Any = None) -> dict[str, object]:
    """Torch settings that affect reproducibility (recorded in the run manifest)."""
    t = require_torch()
    out: dict[str, object] = {
        "torch_version": str(t.__version__),
        "deterministic_algorithms": bool(t.are_deterministic_algorithms_enabled()),
        "num_threads": int(t.get_num_threads()),
    }
    if device is not None and t.device(device).type == "cuda":
        props = t.cuda.get_device_properties(device)
        out |= {
            "torch_cuda_build": t.version.cuda,
            "cuda_device_name": str(props.name),
            "cuda_capability": f"{props.major}.{props.minor}",
            "cuda_total_memory_bytes": int(props.total_memory),
            "cuda_arch_list": list(t.cuda.get_arch_list()),
        }
    return out


def _resolve_device(t: Any, device: str) -> Any:
    dev = t.device(device)
    if dev.type == "cpu":
        return dev
    if dev.type != "cuda":
        raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}")
    check_backend("torch-cuda")
    if dev.index is None:
        return t.device("cuda", t.cuda.current_device())
    if dev.index >= t.cuda.device_count():
        raise ValueError(f"device {device!r}: only {t.cuda.device_count()} CUDA device(s)")
    return dev


class TorchLIFEngine(LIFEngineBase):
    backend = "torch-cpu"  # instances on CUDA report "torch-cuda"

    def __init__(
        self,
        network: SynapticNetwork,
        disc: Discretization,
        dtype: str = "float64",
        populations: Mapping[str, NDArray[np.integer]] | None = None,
        device: str = "cpu",
        max_edges_per_chunk: int = DEFAULT_EDGE_CHUNK,
    ) -> None:
        t = require_torch()
        dev = _resolve_device(t, device)
        if int(max_edges_per_chunk) < 1:
            raise ValueError("max_edges_per_chunk must be >= 1")
        super().__init__(network, disc, dtype, populations)
        self.backend = "torch-cuda" if dev.type == "cuda" else "torch-cpu"
        self._t = t
        self._device = dev
        self._cuda = dev.type == "cuda"
        self._chunk = int(max_edges_per_chunk)
        self._tdtype = t.float32 if dtype == "float32" else t.float64

        def const(x: float) -> Any:
            return t.tensor(x, dtype=self._tdtype, device=dev)

        self._a, self._b, self._c, self._d = (const(x) for x in (disc.a, disc.b, disc.c, disc.d))
        self._v_rest, self._v_reset = const(disc.v_rest), const(disc.v_reset)
        self._v_th = const(disc.v_threshold)
        self._zero = const(0.0)
        # Edge arrays: int32 targets + float64 weights (12 bytes/edge); indices are widened to
        # int64 only for the gathered edges of each chunk. The network arrays are read-only
        # NumPy views; torch gets its own copies (on the device).
        self._row_ptr = t.tensor(network.row_ptr, dtype=t.int64, device=dev)
        self._dst = t.tensor(network.dst, dtype=t.int32, device=dev)
        self._w = t.tensor(network.weight_mV, dtype=t.float64, device=dev)
        # Shared all-False mask for steps without forced/blocked/clamped neurons (never mutated).
        self._none = t.zeros(self.n, dtype=t.bool, device=dev)
        self._v: Any = None
        self._g: Any = None
        self._last: Any = None
        self._ring: list[Any] = []
        self._candidate_v: Any = None

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "impl": "eager",
            "device": str(self._device),
            "max_edges_per_chunk": self._chunk,
            **torch_settings(self._device),
        }

    # --- memory and timing ---------------------------------------------------------------
    def synchronize(self) -> None:
        if self._cuda:
            self._t.cuda.synchronize(self._device)

    def retained_bytes(self) -> dict[str, int]:
        def nbytes(*xs: Any) -> int:
            return int(sum(x.element_size() * x.numel() for x in xs if x is not None))

        return {
            "edges": nbytes(self._row_ptr, self._dst, self._w),
            "state": nbytes(self._v, self._g, self._last, self._none, self._candidate_v),
            "in_flight": nbytes(*self._ring),
        }

    def device_memory_stats(self) -> dict[str, int] | None:
        if not self._cuda:
            return None
        c, d = self._t.cuda, self._device
        return {
            "allocated_bytes": int(c.memory_allocated(d)),
            "reserved_bytes": int(c.memory_reserved(d)),
            "max_allocated_bytes": int(c.max_memory_allocated(d)),
            "max_reserved_bytes": int(c.max_memory_reserved(d)),
        }

    def reset_peak_device_memory(self) -> None:
        if self._cuda:
            self._t.cuda.reset_peak_memory_stats(self._device)

    # --- helpers -------------------------------------------------------------------------
    def _idx(self, a: NDArray[np.int32]) -> Any:
        return self._t.tensor(a, dtype=self._t.int64, device=self._device)

    def _mask(self, idx: NDArray[np.int32]) -> Any:
        """Boolean mask of `idx`. Returns the shared all-False mask when `idx` is empty, so
        callers must not modify the result in place."""
        if idx.size == 0:
            return self._none
        m = self._t.zeros(self.n, dtype=self._t.bool, device=self._device)
        m[self._idx(idx)] = True
        return m

    def _reset_state(self) -> None:
        self._candidate_v = None
        t = self._t
        self._v = self._v_rest.expand(self.n).clone()
        self._g = t.zeros(self.n, dtype=self._tdtype, device=self._device)
        self._last = t.full((self.n,), int(NEVER), dtype=t.int64, device=self._device)
        self._ring = [
            t.zeros(0, dtype=t.int64, device=self._device) for _ in range(self.disc.delay_steps)
        ]

    def _add_rows(self, incoming: Any, starts: Any, counts: Any, n_edges: int) -> None:
        """incoming[dst[e]] += w[e] for every edge e of the given CSR rows (`n_edges` edges)."""
        t = self._t
        first_out = t.cumsum(counts, 0) - counts
        eid = t.repeat_interleave(starts - first_out, counts, output_size=n_edges) + t.arange(
            n_edges, dtype=t.int64, device=self._device
        )
        incoming.index_add_(0, self._dst[eid].to(t.int64), self._w[eid])

    def _incoming(self, arriving: Any) -> Any | None:
        """Float64 sum of the weights of every edge leaving `arriving`, per target neuron, or
        None if those neurons have no outgoing edges.

        Rows are processed in input order. When there are more than `max_edges_per_chunk`
        edges, neurons are grouped by the chunk window their first edge falls in, so a group
        holds at most `max_edges_per_chunk` + (largest out-degree) edges. On the CPU the
        additions per target happen in the same order either way (tests check bitwise
        equality); on CUDA `index_add_` orders them itself."""
        t = self._t
        starts = self._row_ptr[arriving]
        counts = self._row_ptr[arriving + 1] - starts
        ends = t.cumsum(counts, 0)
        total = int(ends[-1])  # host sync: the edge count sizes the temporaries
        if total == 0:
            return None
        incoming = t.zeros(self.n, dtype=t.float64, device=self._device)
        if total <= self._chunk:
            self._add_rows(incoming, starts, counts, total)
            return incoming
        ends_h = ends.cpu().numpy()
        first_h = ends_h - counts.cpu().numpy()
        window = first_h // self._chunk
        cuts = [0, *(np.flatnonzero(np.diff(window)) + 1).tolist(), int(window.size)]
        for lo, hi in itertools.pairwise(cuts):
            n_edges = int(ends_h[hi - 1] - first_h[lo])
            if n_edges:
                self._add_rows(incoming, starts[lo:hi], counts[lo:hi], n_edges)
        return incoming

    # --- the step ------------------------------------------------------------------------
    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        t = self._t
        v, g = self._v, self._g
        clamp = self._mask(clamp_idx)

        # 1. Refractory status at the start of the step.
        refractory = (n - self._last) < self.disc.refractory_steps
        active = ~refractory & ~clamp

        # 2. Exact subthreshold update for active neurons; inactive neurons are frozen.
        v_new = self._v_rest + (v - self._v_rest) * self._a + g * self._b
        if current_idx.size:  # same rule as the NumPy reference: no e*c term without drive
            e = t.zeros(self.n, dtype=t.float64, device=self._device)
            mv = t.tensor(current_mV, dtype=t.float64, device=self._device)
            e.index_add_(0, self._idx(current_idx), mv)
            v_new = v_new + e.to(self._tdtype) * self._c
        self._candidate_v = v_new
        g_new = g * self._d
        v = t.where(active, v_new, v)
        g = t.where(active, g_new, g)

        # 3. Spikes: threshold crossings plus accepted forced spikes.
        forced_req = self._mask(forced_idx)
        crossed = active & (v > self._v_th)
        forced_ok = active & forced_req
        spiking = crossed | forced_ok
        spike_idx = t.nonzero(spiking).flatten()  # ascending
        n_dropped = int((forced_req & ~active).sum())
        self._last[spike_idx] = n

        # 4. Arrivals of spikes emitted D steps ago, summed in float64, rounded once.
        slot = n % self.disc.delay_steps
        arriving = self._ring[slot]
        if arriving.numel():
            incoming = self._incoming(arriving)
            if incoming is not None:
                receives = ~refractory & ~spiking & ~clamp
                g = t.where(receives, g + incoming.to(self._tdtype), g)

        # 5. Reset of spiking neurons; clamped neurons held at rest.
        v[spike_idx] = self._v_reset
        g[spike_idx] = 0.0
        v = t.where(clamp, self._v_rest, v)
        g = t.where(clamp, self._zero, g)

        # 6. Queue spikes whose output is not blocked; they arrive at step n + D.
        propagated = (
            t.ones(spike_idx.numel(), dtype=t.bool, device=self._device)
            if block_idx.size == 0
            else ~self._mask(block_idx)[spike_idx]
        )
        self._ring[slot] = spike_idx[propagated]
        self._v, self._g = v, g
        return _Advanced(
            spike_idx=spike_idx.cpu().numpy().astype(np.int32),
            forced=forced_ok[spike_idx].cpu().numpy(),
            propagated=propagated.cpu().numpy(),
            n_forced_dropped=n_dropped,
        )

    def state(self) -> EngineState:
        return EngineState(
            step=self._step_index,
            v=self._v.cpu().numpy().copy(),
            g=self._g.cpu().numpy().copy(),
            last_spike=self._last.cpu().numpy().copy(),
            in_flight=tuple(a.cpu().numpy().astype(np.int32) for a in self._ring),
        )

    def candidate_potentials(self, neurons: NDArray[np.int64]) -> NDArray[np.float64]:
        if self._candidate_v is None:
            raise RuntimeError("candidate potentials require a completed step")
        idx = self._t.tensor(neurons, dtype=self._t.int64, device=self._device)
        return self._candidate_v[idx].to(self._t.float64).cpu().numpy().copy()  # type: ignore[no-any-return]

    def _load_state(self, state: EngineState) -> None:
        t = self._t
        if state.v.shape != (self.n,) or state.g.shape != (self.n,):
            raise ValueError("state vectors do not match the network size")
        self._v = t.tensor(state.v, dtype=self._tdtype, device=self._device)
        self._g = t.tensor(state.g, dtype=self._tdtype, device=self._device)
        self._last = t.tensor(state.last_spike, dtype=t.int64, device=self._device)
        self._ring = [self._idx(a) for a in state.in_flight]
        self._candidate_v = None

    def _state_finite(self) -> bool:
        t = self._t
        return bool(
            t.isfinite(self._v).all().item()
            and t.isfinite(self._g).all().item()
            and (self._candidate_v is None or t.isfinite(self._candidate_v).all().item())
        )
