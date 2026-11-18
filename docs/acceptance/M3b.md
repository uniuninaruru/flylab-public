# M3b report: fused CUDA engine (PR-03b, performance)

Date: 2026-11-09. Measured on the owner's homelab PC: RTX 5070 12 GB, driver 616.92, Windows 11 +
WSL2 distro, Docker Engine + NVIDIA Container Toolkit, image `flylab:dev` with torch
2.15.0+cu132 and Triton 3.9.0. The benchmark script ran at commit `8507026`; the two profiles ran at
`087fd90`, which changes only `scripts/profile_step.py` and docs, not the engines. Every number
below is copied from a record in `docs/acceptance/evidence/`.

PR-03 (M3) showed the eager CUDA engine to be correct but launch- and sync-bound: slower than the
NumPy reference (Q22). This PR adds a second `torch-cuda` implementation, the fused engine
(`flylab.neural.lif_fused`, MODEL_ASSUMPTIONS 2.2–2.3). It runs four Triton kernels per step and
copies spike lists to the host once per 32 steps. Arrivals are exact integer sums. `engine_impl:
auto` selects it whenever it can run.

## Result

| | Eager CUDA (PR-03) | **Fused CUDA (PR-03b)** | NumPy (same host CPU) |
|---|---|---|---|
| B1, full MaleCNS, 100 ms: wall-clock median | 3.442 s | **0.0959 s** | 0.461 s |
| B1 steps per wall second | 290 | **10,431** (1.04 × real time) | 2,169 |
| B2, full MaleCNS, 1 s: wall-clock median | 27.661 s (M3, 2026-11-04) | **1.384 s** (0.75 × real time; best 0.949 s) | not run |
| Spikes per B1 / B2 run | 23,138 / 237,655 | 23,138 / 237,655 | 23,138 / — |

- **Speed**: on B1 the fused engine is **36 × faster than the eager CUDA engine** and **4.8 ×
  faster than NumPy**, at about real time.
- **Correctness is unchanged**: `flylab parity` in lockstep against NumPy is **identical** on the
  full graph for 100 ms and for 1 s (10,000 / 10,000 steps, 237,655 spikes, |Δv| = |Δg| = 0,
  no justified flips, no resyncs). It uses the stricter parity policy from the PR-03 review fixes.
- **GPU tests pass on the host**: `tests/gpu`, 16 tests including 7 for the fused engine.
  - Lockstep against NumPy over 1,500 steps, in float64 and float32.
  - Results do not depend on the batch size (1, 7, 32).
  - Runs are deterministic.
  - Checkpoints move fused → NumPy → fused.

## Measured records

| Record | Engine | Graph | simulated | warm-up + timed | wall s median [p25, p75] (min–max) | steps/s | sim s per wall s | spikes/run | GPU allocator peak reserved / allocated | device-wide used delta (observed) | host RSS peak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `bench-B0-20261109T064609Z.json` | fused float32 | synthetic 20k | 100 ms | 3 + 10 | 0.1173 [0.0932, 0.2820] (0.0904–0.5397) | 8,755 | 0.875 | 128 | 24 / 9 MiB | 223 MiB | 1,090 MiB |
| `bench-B0-1s-20261109T064619Z.json` | fused float32 | synthetic 20k | 1 s | 1 + 3 | 1.4552 [1.4416, 2.5387] (1.4281–3.6222) | 6,872 | 0.687 | 128 | 24 / 9 MiB | 223 MiB | 1,089 MiB |
| `bench-B1-20261109T064627Z.json` | fused float32 | MaleCNS full | 100 ms | 3 + 10 | 0.0959 [0.0952, 0.0965] (0.0944–0.3116) | 10,431 | 1.043 | 23,138 | 248 / 239 MiB | 447 MiB | 2,339 MiB |
| `bench-B2-20261109T064656Z.json` | fused float32 | MaleCNS full | 1 s | 3 + 10 | 1.3840 [1.0135, 2.1268] (0.9488–4.1618) | 7,524 | 0.752 | 237,655 | 248 / 239 MiB | 447 MiB | 2,346 MiB |
| `bench-B1-eager-20261109T064714Z.json` | eager float32 | MaleCNS full | 100 ms | 1 + 3 | 3.4424 [3.0921, 3.5005] (2.7419–3.5587) | 290 | 0.029 | 23,138 | 310 / 293 MiB | 511 MiB | 2,484 MiB |
| `bench-B1-numpy-20261109T064719Z.json` | numpy float32 | MaleCNS full | 100 ms | 1 + 3 | 0.4610 [0.4587, 0.4653] (0.4564–0.4695) | 2,169 | 0.217 | 23,138 | — | — | 1,803 MiB |

Every record has `status: PASS`. Gates: no OOM, all runs finite, allocator peak and the observed
device-wide delta both within the 10 GiB target. The device-wide delta covers the CUDA context
and Triton's kernel images, so it is larger than the allocator peak (review finding 8).

| Parity record | Graph | steps | status | identical | spikes (NumPy / fused) | max \|Δv\|, \|Δg\| mV |
|---|---|---|---|---|---|---|
| `parity-B0-20261109T064552Z.json` | synthetic 20k | 1,000 | **identical** | 1,000 | 128 / 128 | 0, 0 |
| `parity-B1-20261109T064602Z.json` | MaleCNS full | 1,000 | **identical** | 1,000 | 23,138 / 23,138 | 0, 0 |
| `parity-B2-20261109T064807Z.json` | MaleCNS full | 10,000 | **identical** | 10,000 | 237,655 / 237,655 | 0, 0 |

## Where the time went (profiles)

`scripts/profile_step.py` now profiles `simulate()` (the batched path) and counts device time once.
It ran over 300 steps of B1 after 200 warm-up steps; the profiler inflates host times.

| Per step | Eager (`profile-B1-eager-20261109T064934Z.json`) | Fused (`profile-B1-fused-20261109T064921Z.json`) |
|---|---|---|
| Kernel launches | 110 (`cudaLaunchKernel`) | 4 (`cuLaunchKernelEx`) |
| Stream synchronizations | 8.0 | 0.10 |
| Host→device / device→host copies | 1.0 / 7.0 | 1.0 / 0.10 |
| GPU device time | 162.5 µs | 23.2 µs |
| Host self time (with profiler) | 3,632 µs | 136 µs |

The eager re-profile corrects M3's earlier figures (review finding 7). The device time is
162.5 µs per step, not 322 µs, and the copies are one host→device plus seven device→host.

## Not claimed / limitations

- About real time on these workloads only: one forced-spike input population, sparse activity
  (mean active fraction ~1.4e-4 per step). Denser activity means more arrivals per step and will
  be slower. B2's wide spread (0.95–4.16 s) comes from a shared host: desktop and homelab
  services use the same GPU and CPU.
- The fused engine needs a C compiler in the image (Triton builds its launcher). The Dockerfile
  now installs `gcc`, `libc6-dev` and `python3.12-dev`. Kernels compile on first use, about
  3 s once per process, cached in `TRITON_CACHE_DIR`.
- Arrivals are integer sums scaled once by `w_syn`. With several arrivals per neuron and step,
  results can differ from the reference by float64 rounding (MODEL_ASSUMPTIONS 2.3). None of the
  measured runs differed.
- It needs networks whose weights are `w_syn × integer counts` (true for `build_network`);
  otherwise `auto` falls back to the eager engine.
- A further step, capturing a batch of steps in a CUDA graph (≈1.3 µs per kernel instead of
  ≈10 µs per Triton launch, measured on this host), is not needed for real time on B1 and is
  left out.

## How to reproduce

```bash
bash scripts/remote/sync_to_host.sh -F <homelab>/mac/ssh_config homelab-pc
bash scripts/remote/run_gpu_bench.sh -F <homelab>/mac/ssh_config homelab-pc
bash scripts/remote/compose_run.sh -F <homelab>/mac/ssh_config homelab-pc gpu \
  flylab parity --config configs/bench/b2_male_cns_1s.yaml --tag B2
bash scripts/remote/compose_run.sh -F <homelab>/mac/ssh_config homelab-pc gpu \
  python scripts/profile_step.py --config configs/bench/b1_male_cns_100ms.yaml --tag B1-fused --engine-impl fused
```
