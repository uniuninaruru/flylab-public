"""Triton kernels of the fused CUDA LIF engine (`flylab.neural.lif_fused` explains the scheme).

Imported only where Triton exists (Linux CUDA torch wheels). Excluded from strict mypy: Triton
kernels are untyped by construction. Every arithmetic expression mirrors the NumPy reference
operation by operation; launch with `enable_fp_fusion=False` so no multiply-add is contracted.
"""

import triton
import triton.language as tl


@triton.jit
def _inputs_kernel(
    buf, nd, nf, nb, nc,
    drive_stamp, drive_val, forced_stamp, block_stamp, clamp_stamp,
    counters, step, oslot, wslot, n_counters_per_slot,
    BLOCK: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    if pid == 0:  # this step's output count, dropped count and ring slot count start at 0
        tl.store(counters + oslot * n_counters_per_slot + 0, 0)
        tl.store(counters + oslot * n_counters_per_slot + 1, 0)
        tl.store(counters + n_counters_per_slot * 4096 + wslot, 0)
    j = pid * BLOCK + tl.arange(0, BLOCK)
    total = nd + nf + nb + nc
    live = j < total
    is_d = live & (j < nd)
    is_f = live & (j >= nd) & (j < nd + nf)
    is_b = live & (j >= nd + nf) & (j < nd + nf + nb)
    is_c = live & (j >= nd + nf + nb)
    # Offsets of each record in `buf` (drive values follow the drive indices).
    off = tl.where(is_d, j, j + nd)
    idx = tl.load(buf + off, mask=live, other=0)
    bits = tl.load(buf + nd + j, mask=is_d, other=0)
    tl.store(drive_val + idx, bits.to(tl.float64, bitcast=True), mask=is_d)
    tl.store(drive_stamp + idx, step, mask=is_d)
    tl.store(forced_stamp + idx, step, mask=is_f)
    tl.store(block_stamp + idx, step, mask=is_b)
    tl.store(clamp_stamp + idx, step, mask=is_c)


@triton.jit
def _integrate_kernel(
    v, g, last, drive_stamp, drive_val, forced_stamp, block_stamp, clamp_stamp, consts, cand,
    out_idx, out_flags, counters, ring, ring_cnt,
    n, step, refractory_steps, oslot, wslot, cap, n_counters_per_slot,
    BLOCK: tl.constexpr,
):  # fmt: skip
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(consts + 0)
    b = tl.load(consts + 1)
    c = tl.load(consts + 2)
    d = tl.load(consts + 3)
    v_rest = tl.load(consts + 4)
    v_th = tl.load(consts + 6)
    vv = tl.load(v + i, mask=m, other=0.0)
    gg = tl.load(g + i, mask=m, other=0.0)
    lst = tl.load(last + i, mask=m, other=0)
    # 1. refractory status at the start of the step; clamped neurons are frozen too
    refractory = (step - lst) < refractory_steps
    clamped = tl.load(clamp_stamp + i, mask=m, other=-1) == step
    active = m & (refractory == 0) & (clamped == 0)
    # 2. exact subthreshold update, operation by operation as in the reference
    t1 = vv - v_rest
    t2 = t1 * a
    t3 = v_rest + t2
    t4 = gg * b
    vn = t3 + t4
    driven = tl.load(drive_stamp + i, mask=m, other=-1) == step
    e = tl.load(drive_val + i, mask=m & driven, other=0.0).to(vv.dtype)
    vn = tl.where(driven, vn + e * c, vn)
    tl.store(cand + i, vn, mask=m)  # phase-2 candidate, for parity diagnostics
    gn = gg * d
    vv = tl.where(active, vn, vv)
    gg = tl.where(active, gn, gg)
    # 3. spikes: threshold crossings plus accepted forced spikes
    forced = m & (tl.load(forced_stamp + i, mask=m, other=-1) == step)
    crossed = active & (vv > v_th)
    forced_ok = active & forced
    spiking = crossed | forced_ok
    n_dropped = tl.sum((forced & (active == 0)).to(tl.int32), axis=0)
    tl.atomic_add(counters + oslot * n_counters_per_slot + 1, n_dropped)
    tl.store(last + i, tl.where(spiking, step, lst), mask=m)
    tl.store(v + i, vv, mask=m)
    tl.store(g + i, gg, mask=m)
    # spike list for the host (unordered) with flags: bit 0 forced, bit 1 propagated
    blocked = tl.load(block_stamp + i, mask=m, other=-1) == step
    propagated = spiking & (blocked == 0)
    s = spiking.to(tl.int32)
    base = tl.atomic_add(counters + oslot * n_counters_per_slot + 0, tl.sum(s, axis=0))
    pos = base + tl.cumsum(s, axis=0) - 1
    flags = forced_ok.to(tl.int8) + 2 * propagated.to(tl.int8)
    tl.store(out_idx + oslot * cap + pos, i, mask=spiking)
    tl.store(out_flags + oslot * cap + pos, flags, mask=spiking)
    # 6. propagated spikes go to the in-flight ring (arrive `delay` steps later)
    p = propagated.to(tl.int32)
    rbase = tl.atomic_add(ring_cnt + wslot, tl.sum(p, axis=0))
    rpos = rbase + tl.cumsum(p, axis=0) - 1
    tl.store(ring + wslot * cap + rpos, i, mask=propagated)


@triton.jit
def _arrivals_kernel(
    ring, ring_cnt, rslot, cap, row_ptr, dst, count, acc,
    NPROG: tl.constexpr, EDGE_BLOCK: tl.constexpr,
):  # fmt: skip
    pid = tl.program_id(0)
    k = tl.load(ring_cnt + rslot)
    for s in range(pid, k, NPROG):
        pre = tl.load(ring + rslot * cap + s).to(tl.int64)
        start = tl.load(row_ptr + pre)
        end = tl.load(row_ptr + pre + 1)
        for e0 in range(start, end, EDGE_BLOCK):
            offs = e0 + tl.arange(0, EDGE_BLOCK)
            em = offs < end
            tgt = tl.load(dst + offs, mask=em, other=0)
            cnt = tl.load(count + offs, mask=em, other=0).to(tl.int64)
            tl.atomic_add(acc + tgt, cnt, mask=em)


@triton.jit
def _finish_kernel(
    v, g, last, clamp_stamp, acc, consts, scale64,
    n, step, refractory_steps,
    BLOCK: tl.constexpr,
):  # fmt: skip
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v_rest = tl.load(consts + 4)
    v_reset = tl.load(consts + 5)
    vv = tl.load(v + i, mask=m, other=0.0)
    gg = tl.load(g + i, mask=m, other=0.0)
    lst = tl.load(last + i, mask=m, other=0)
    clamped = tl.load(clamp_stamp + i, mask=m, other=-1) == step
    spiked = lst == step
    # neurons that were not refractory at the start of the step, did not spike and are not
    # clamped receive (a neuron that just spiked has last == step, so it is excluded here)
    receives = m & ((step - lst) >= refractory_steps) & (clamped == 0)
    cnt = tl.load(acc + i, mask=m, other=0)
    tl.store(acc + i, tl.zeros_like(cnt), mask=m)
    inc = (cnt.to(tl.float64) * tl.load(scale64)).to(gg.dtype)
    gg = tl.where(receives, gg + inc, gg)
    # 5. reset of spiking neurons; clamped neurons held at rest
    vv = tl.where(spiked, v_reset, vv)
    gg = tl.where(spiked, tl.zeros_like(gg), gg)
    vv = tl.where(clamped, v_rest, vv)
    gg = tl.where(clamped, tl.zeros_like(gg), gg)
    tl.store(v + i, vv, mask=m)
    tl.store(g + i, gg, mask=m)
