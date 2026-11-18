"""Fused CUDA LIF engine (`torch-cuda`, impl `fused`): the scheme of MODEL_ASSUMPTIONS 2.1 in four
Triton kernels per step, with no host synchronization inside a batch of steps.

Why: the eager torch engine spends ~3 ms per 0.1 ms step on ~110 kernel launches and 8 host
synchronizations while the GPU works for well under 0.3 ms (docs/acceptance/M3.md). Here a step is

1. `_inputs`    stamps this step's drive / forced / block / clamp inputs into per-neuron arrays
                (stamp == step number means "applies now"; no clearing pass is needed) and resets
                this step's counters;
2. `_integrate` phases 1-3 and the bookkeeping of 5-6 for every neuron: refractory status,
                exact subthreshold update, threshold, forced spikes, spike list and in-flight ring;
3. `_arrivals`  phase 4 for the spikes emitted `delay` steps ago: integer synapse counts are added
                with atomics into a per-neuron int64 accumulator;
4. `_finish`    applies the arrivals (count x w_syn, rounded once to the state dtype) to neurons
                that receive, then resets spiking and clamped neurons, and clears the accumulator.

Spike lists are copied to the host once per batch of `batch_steps` steps.

Numerics versus the NumPy reference:
- every per-neuron operation is the same IEEE-754 operation in the same order, with FMA
  contraction disabled (`enable_fp_fusion=False`);
- arrivals are summed as exact integers (synapse count x sign) and scaled once by `w_syn`, while
  the reference adds `w_syn x count` terms in float64 in edge order; the two differ by float64
  rounding only when a neuron receives several arrivals in one step, and the result is
  deterministic regardless of the order in which the GPU adds them;
- spike lists come back unordered from the GPU and are sorted on the host.

The comparison policy is `flylab.neural.parity` (MODEL_ASSUMPTIONS 2.3). Triton needs a C compiler
at first use of a kernel (the Docker image provides gcc); kernels compile once per process
(cached in `TRITON_CACHE_DIR`).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from flylab.neural.engine import NEVER, EngineState, LIFEngineBase, NeuralStep, Schedule, _Advanced
from flylab.neural.lif_torch import check_backend, require_torch, torch_settings
from flylab.neural.network import SynapticNetwork
from flylab.neural.params import Discretization

_triton: Any = None
_k: Any = None
try:  # Triton ships with the Linux CUDA torch wheels only
    import triton as _triton_module

    from flylab.neural import _fused_kernels as _kernels_module

    _triton, _k = _triton_module, _kernels_module
except ImportError:  # pragma: no cover - macOS / CPU-only environments
    pass

DEFAULT_BATCH_STEPS = 32
_BLOCK = 1024  # neurons per program in the dense kernels
_EDGE_BLOCK = 256  # edges per vector iteration in the arrivals kernel
_ARRIVAL_PROGRAMS = 512
# Input record kinds, packed into one int64 array per step.
_KINDS = ("drive", "forced", "block", "clamp")


class FusedUnavailableError(RuntimeError):
    """The fused engine cannot run here (no CUDA, no Triton, or no C compiler)."""


def fused_status() -> dict[str, object]:
    """Whether the fused engine can run in this process, and why not."""
    if _triton is None:
        return {"available": False, "reason": "Triton is not installed (Linux CUDA wheels only)"}
    try:
        check_backend("torch-cuda")
    except RuntimeError as exc:
        return {"available": False, "reason": str(exc)}
    import os

    if not (os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")):
        return {"available": False, "reason": "Triton needs a C compiler (gcc) at run time"}
    return {"available": True, "triton_version": str(_triton.__version__)}


def network_supports_fused(network: SynapticNetwork) -> bool:
    """True if every weight is exactly `weight_scale_mV * signed_count` (integer arrivals)."""
    if network.signed_count is None or network.weight_scale_mV is None:
        return False
    rebuilt = float(network.weight_scale_mV) * network.signed_count.astype(np.float64)
    return bool(np.array_equal(rebuilt, network.weight_mV))


# --- host-side helpers (pure NumPy, unit-tested without a GPU) ---------------------------------


def aggregate_drive(
    idx: NDArray[np.int32], mv: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Unique neuron indices and their summed drive, added in input order exactly like the
    reference's `np.add.at(e, idx, mv)` on a zero vector (so the float64 sums are identical)."""
    if idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    uniq, inv = np.unique(idx, return_inverse=True)
    total = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(total, inv, mv)
    return uniq.astype(np.int64), total


def pack_inputs(
    current_idx: NDArray[np.int32],
    current_mv: NDArray[np.float64],
    forced_idx: NDArray[np.int32],
    block_idx: NDArray[np.int32],
    clamp_idx: NDArray[np.int32],
) -> tuple[NDArray[np.int64], tuple[int, int, int, int]]:
    """One int64 array per step: drive indices, drive values (float64 bits), forced, block and
    clamp indices; plus the four lengths."""
    d_idx, d_val = aggregate_drive(current_idx, current_mv)
    parts = [
        d_idx,
        d_val.view(np.int64),
        forced_idx.astype(np.int64),
        block_idx.astype(np.int64),
        clamp_idx.astype(np.int64),
    ]
    lengths = (int(d_idx.size), int(forced_idx.size), int(block_idx.size), int(clamp_idx.size))
    return np.concatenate(parts) if any(lengths) else np.zeros(0, dtype=np.int64), lengths


def ring_to_state(
    step: int, delay: int, slots: list[NDArray[np.int32]]
) -> tuple[NDArray[np.int32], ...]:
    """Convert the engine's `delay + 1` ring slots (slot of a spike emitted at step m is
    m mod (delay + 1)) to the reference layout: `delay` slots, slot k holding the still-in-flight
    spikes of the step m = k mod delay, ascending."""
    out = [np.zeros(0, dtype=np.int32) for _ in range(delay)]
    for m in range(max(0, step - delay), step):
        out[m % delay] = np.sort(slots[m % (delay + 1)]).astype(np.int32)
    return tuple(out)


def state_to_ring(
    step: int, delay: int, in_flight: tuple[NDArray[np.int32], ...]
) -> list[NDArray[np.int32]]:
    """Inverse of `ring_to_state`."""
    slots = [np.zeros(0, dtype=np.int32) for _ in range(delay + 1)]
    for m in range(max(0, step - delay), step):
        slots[m % (delay + 1)] = np.asarray(in_flight[m % delay], dtype=np.int32)
    return slots


# --- kernels (src/flylab/neural/_fused_kernels.py) -----------------------------------------


class FusedCudaLIFEngine(LIFEngineBase):
    """`torch-cuda` engine with fused Triton kernels (see the module docstring)."""

    backend = "torch-cuda"
    impl = "fused"

    def __init__(
        self,
        network: SynapticNetwork,
        disc: Discretization,
        dtype: str = "float64",
        populations: Mapping[str, NDArray[np.integer]] | None = None,
        device: str = "cuda",
        batch_steps: int = DEFAULT_BATCH_STEPS,
    ) -> None:
        status = fused_status()
        if not status["available"]:
            raise FusedUnavailableError(f"fused CUDA engine unavailable: {status['reason']}")
        if not network_supports_fused(network):
            raise FusedUnavailableError(
                "fused CUDA engine needs weights = weight_scale_mV x integer counts "
                "(networks from build_network); use the eager torch engine"
            )
        if int(batch_steps) < 1:
            raise ValueError("batch_steps must be >= 1")
        t = require_torch("torch-cuda")
        super().__init__(network, disc, dtype, populations)
        dev = t.device(device)
        if dev.type != "cuda":
            raise ValueError("the fused engine runs on CUDA only")
        if dev.index is None:
            dev = t.device("cuda", t.cuda.current_device())
        self._t = t
        self._device = dev
        self._batch = int(batch_steps)
        self._tdtype = t.float32 if dtype == "float32" else t.float64
        self._np_dtype = np.dtype(dtype)
        self._D = disc.delay_steps
        self._cap = max(1, self.n)
        rd = np.dtype(dtype).type
        # a, b, c, d, v_rest, v_reset, v_threshold: rounded once to the state dtype
        raw = (disc.a, disc.b, disc.c, disc.d, disc.v_rest, disc.v_reset, disc.v_threshold)
        consts = np.array([rd(x) for x in raw], dtype=self._np_dtype)
        self._consts = t.tensor(consts, device=dev)
        assert network.weight_scale_mV is not None
        assert network.signed_count is not None
        self._scale = t.tensor([float(network.weight_scale_mV)], dtype=t.float64, device=dev)
        self._row_ptr = t.tensor(network.row_ptr, dtype=t.int64, device=dev)
        self._dst = t.tensor(network.dst, dtype=t.int32, device=dev)
        self._count = t.tensor(network.signed_count, dtype=t.int32, device=dev)
        n, cap, nslots = self.n, self._cap, self._D + 1

        def full(shape: Any, value: int, dtype: Any) -> Any:
            return t.full(shape, value, dtype=dtype, device=dev)

        self._drive_stamp = full((n,), -1, t.int64)
        self._drive_val = t.zeros(n, dtype=t.float64, device=dev)
        self._forced_stamp = full((n,), -1, t.int64)
        self._block_stamp = full((n,), -1, t.int64)
        self._clamp_stamp = full((n,), -1, t.int64)
        self._acc = t.zeros(n, dtype=t.int64, device=dev)
        self._ring = t.zeros(nslots * cap, dtype=t.int32, device=dev)
        # counters: per output slot [n_spikes, n_dropped], then ring counts at offset 2*4096
        if self._batch > 4096:
            raise ValueError("batch_steps must be <= 4096")
        self._counters = t.zeros(2 * 4096 + nslots, dtype=t.int32, device=dev)
        self._ring_cnt = self._counters[2 * 4096 :]
        self._out_idx = t.zeros(self._batch * cap, dtype=t.int32, device=dev)
        self._out_flags = t.zeros(self._batch * cap, dtype=t.int8, device=dev)
        # Per-step inputs are staged in pinned host memory (one row per step of a batch) and
        # copied asynchronously; a pageable copy would make the host wait for the GPU each step.
        self._stage_cap = 1024
        self._staging = t.empty((self._batch, self._stage_cap), dtype=t.int64, pin_memory=True)
        self._inbuf = t.empty(self._stage_cap, dtype=t.int64, device=dev)
        self._cand = t.zeros(self.n, dtype=self._tdtype, device=dev)
        self._cand_valid = False
        self._v: Any = None
        self._g: Any = None
        self._last: Any = None

    # --- bookkeeping ---------------------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "impl": self.impl,
            "device": str(self._device),
            "batch_steps": self._batch,
            "arrival_accumulation": "exact integer synapse counts x w_syn, one float64 rounding",
            "triton_version": str(_triton.__version__),
            "fp_fusion": False,
            **torch_settings(self._device),
        }

    def synchronize(self) -> None:
        self._t.cuda.synchronize(self._device)

    def retained_bytes(self) -> dict[str, int]:
        def nb(*xs: Any) -> int:
            return int(sum(x.element_size() * x.numel() for x in xs if x is not None))

        return {
            "edges": nb(self._row_ptr, self._dst, self._count),
            "state": nb(
                self._v,
                self._g,
                self._last,
                self._drive_stamp,
                self._drive_val,
                self._forced_stamp,
                self._block_stamp,
                self._clamp_stamp,
                self._acc,
                self._cand,
            ),
            "in_flight": nb(self._ring, self._counters, self._out_idx, self._out_flags),
        }

    def device_memory_stats(self) -> dict[str, int] | None:
        c, d = self._t.cuda, self._device
        return {
            "allocated_bytes": int(c.memory_allocated(d)),
            "reserved_bytes": int(c.memory_reserved(d)),
            "max_allocated_bytes": int(c.max_memory_allocated(d)),
            "max_reserved_bytes": int(c.max_memory_reserved(d)),
        }

    def reset_peak_device_memory(self) -> None:
        self._t.cuda.reset_peak_memory_stats(self._device)

    def _reset_state(self) -> None:
        t = self._t
        self._v = self._consts[4].expand(self.n).clone()  # exact state-dtype V_rest
        self._g = t.zeros(self.n, dtype=self._tdtype, device=self._device)
        self._last = t.full((self.n,), int(NEVER), dtype=t.int64, device=self._device)
        for stamp in (self._drive_stamp, self._forced_stamp, self._block_stamp, self._clamp_stamp):
            stamp.fill_(-1)
        self._acc.zero_()
        self._counters.zero_()
        self._cand_valid = False

    # --- one step on the device (no host synchronization) -------------------------------------
    def _enqueue(
        self,
        n: int,
        oslot: int,
        current_idx: NDArray[np.int32],
        current_mv: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> None:
        t = self._t
        packed, (nd, nf, nb, nc) = pack_inputs(
            current_idx, current_mv, forced_idx, block_idx, clamp_idx
        )
        if packed.size:
            if packed.size > self._stage_cap:
                self.synchronize()  # copies still reading the old staging rows must finish
                self._stage_cap = max(packed.size, 2 * self._stage_cap)
                self._staging = t.empty((self._batch, self._stage_cap), dtype=t.int64,
                                        pin_memory=True)  # fmt: skip
                self._inbuf = t.empty(self._stage_cap, dtype=t.int64, device=self._device)
            host = self._staging[oslot, : packed.size]
            host.numpy()[:] = packed
            # stream-ordered: runs after the previous step's kernels have read `_inbuf`
            self._inbuf[: packed.size].copy_(host, non_blocking=True)
        wslot = n % (self._D + 1)
        rslot = (n + 1) % (self._D + 1)  # = (n - D) mod (D + 1): spikes emitted D steps ago
        total = nd + nf + nb + nc
        grid_in = (max(1, -(-total // _BLOCK)),)
        _k._inputs_kernel[grid_in](
            self._inbuf, nd, nf, nb, nc,
            self._drive_stamp, self._drive_val, self._forced_stamp, self._block_stamp,
            self._clamp_stamp, self._counters, n, oslot, wslot, 2,
            BLOCK=_BLOCK, enable_fp_fusion=False,
        )  # fmt: skip
        grid = (-(-self.n // _BLOCK),)
        _k._integrate_kernel[grid](
            self._v, self._g, self._last, self._drive_stamp, self._drive_val,
            self._forced_stamp, self._block_stamp, self._clamp_stamp, self._consts,
            self._cand,
            self._out_idx, self._out_flags, self._counters, self._ring, self._ring_cnt,
            self.n, n, self.disc.refractory_steps, oslot, wslot, self._cap, 2,
            BLOCK=_BLOCK, enable_fp_fusion=False,
        )  # fmt: skip
        _k._arrivals_kernel[(_ARRIVAL_PROGRAMS,)](
            self._ring, self._ring_cnt, rslot, self._cap, self._row_ptr, self._dst,
            self._count, self._acc,
            NPROG=_ARRIVAL_PROGRAMS, EDGE_BLOCK=_EDGE_BLOCK, enable_fp_fusion=False,
        )  # fmt: skip
        self._cand_valid = True
        _k._finish_kernel[grid](
            self._v, self._g, self._last, self._clamp_stamp, self._acc, self._consts,
            self._scale, self.n, n, self.disc.refractory_steps,
            BLOCK=_BLOCK, enable_fp_fusion=False,
        )  # fmt: skip

    def _collect(self, k: int) -> list[_Advanced]:
        """Copy the spike lists of the first `k` output slots to the host (one sync)."""
        counters = self._counters[: 2 * k].view(k, 2).cpu().numpy()
        counts, dropped = counters[:, 0], counters[:, 1]
        width = int(counts.max()) if k else 0
        if width:
            idx_all = self._out_idx.view(self._batch, self._cap)[:k, :width].cpu().numpy()
            flg_all = self._out_flags.view(self._batch, self._cap)[:k, :width].cpu().numpy()
        out = []
        for j in range(k):
            c = int(counts[j])
            if c:
                order = np.argsort(idx_all[j, :c], kind="stable")
                idx = idx_all[j, :c][order].astype(np.int32)
                flg = flg_all[j, :c][order]
            else:
                idx = np.zeros(0, dtype=np.int32)
                flg = np.zeros(0, dtype=np.int8)
            out.append(
                _Advanced(
                    spike_idx=idx,
                    forced=(flg & 1).astype(bool),
                    propagated=(flg & 2).astype(bool),
                    n_forced_dropped=int(dropped[j]),
                )
            )
        return out

    def _advance(
        self,
        n: int,
        current_idx: NDArray[np.int32],
        current_mV: NDArray[np.float64],
        forced_idx: NDArray[np.int32],
        block_idx: NDArray[np.int32],
        clamp_idx: NDArray[np.int32],
    ) -> _Advanced:
        self._enqueue(n, 0, current_idx, current_mV, forced_idx, block_idx, clamp_idx)
        return self._collect(1)[0]

    def _run_steps(self, steps: int, schedule: Schedule | None) -> Iterator[NeuralStep]:
        if self.seed is None:
            raise RuntimeError("call reset(seed) before simulate()")
        done = 0
        while done < steps:
            k = min(self._batch, steps - done)
            first = self._step_index
            for j in range(k):
                n = first + j
                drive, iv = schedule.inputs(n) if schedule else (None, None)
                self._enqueue(n, j, *self._validated_inputs(drive, iv))
            advs = self._collect(k)
            for j, adv in enumerate(advs):
                self._step_index = first + j + 1
                yield self._neural_step(first + j, adv)
            done += k

    # --- state import / export ----------------------------------------------------------------
    def state(self) -> EngineState:
        D = self._D
        counts = self._ring_cnt.cpu().numpy()
        ring = self._ring.view(D + 1, self._cap)
        slots = [ring[s, : int(counts[s])].cpu().numpy() for s in range(D + 1)]
        return EngineState(
            step=self._step_index,
            v=self._v.cpu().numpy().copy(),
            g=self._g.cpu().numpy().copy(),
            last_spike=self._last.cpu().numpy().copy(),
            in_flight=ring_to_state(self._step_index, D, slots),
        )

    def _load_state(self, state: EngineState) -> None:
        t = self._t
        if state.v.shape != (self.n,) or state.g.shape != (self.n,):
            raise ValueError("state vectors do not match the network size")
        if len(state.in_flight) != self._D:
            raise ValueError("state has a different number of in-flight slots")
        slots = state_to_ring(state.step, self._D, state.in_flight)
        self._v = t.tensor(state.v, dtype=self._tdtype, device=self._device)
        self._g = t.tensor(state.g, dtype=self._tdtype, device=self._device)
        self._last = t.tensor(state.last_spike, dtype=t.int64, device=self._device)
        ring = self._ring.view(self._D + 1, self._cap)
        for s, arr in enumerate(slots):
            if arr.size:
                ring[s, : arr.size] = t.tensor(arr, dtype=t.int32, device=self._device)
            self._ring_cnt[s] = int(arr.size)
        # stamps refer to step numbers; anything below the restored step never applies again
        for stamp in (self._drive_stamp, self._forced_stamp, self._block_stamp, self._clamp_stamp):
            stamp.fill_(-1)
        self._acc.zero_()
        self._cand_valid = False

    def candidate_potentials(self, neurons: NDArray[np.int64]) -> NDArray[np.float64]:
        if not self._cand_valid:
            raise RuntimeError("candidate potentials require a completed step")
        idx = self._t.tensor(neurons, dtype=self._t.int64, device=self._device)
        out: NDArray[np.float64] = self._cand[idx].to(self._t.float64).cpu().numpy().copy()
        return out

    def _state_finite(self) -> bool:
        t = self._t
        finite = t.isfinite(self._v).all() & t.isfinite(self._g).all()
        if self._cand_valid:
            finite = finite & t.isfinite(self._cand).all()
        return bool(finite.item())
