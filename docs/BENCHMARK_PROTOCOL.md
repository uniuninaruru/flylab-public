# Benchmark protocol

Implemented in PR-03 as `flylab bench` (`flylab.experiments.benchmark`). Measured results live in
`docs/acceptance/M3.md` and the JSON records in `docs/acceptance/evidence/`. **Never pre-fill
numbers**: every field of a record was measured by the process that wrote it, or is `null` with
a reason.

## Record for every benchmark run (spec 11.1)

| Spec item | Where in the `bench-*.json` record |
|---|---|
| GPU model, compute capability, driver | `engine.cuda_device_name`, `engine.cuda_capability`, `gpu.nvidia_smi_before[].driver_version` |
| CUDA runtime (torch build), PyTorch version | `engine.torch_cuda_build`, `engine.torch_version`, `engine.cuda_arch_list` |
| Host OS / kernel, CPU, RAM | `environment.os` (WSL2 kernel inside WSL), `hardware.cpu_model`, `hardware.logical_cpus`, `hardware.total_ram_bytes` (the WSL VM's), `hardware.cgroup_memory_limit_bytes` (the container's) |
| Dataset version + hash, node set | `dataset.dataset_id`, `dataset.node_set`, `dataset.files[].sha256`, `dataset.graph_fingerprint_sha256` |
| Graph nodes, unique edges, synapse totals | `dataset.n_neurons`, `dataset.n_unique_edges`, `dataset.total_synapses`; after the sign policy `network.n_edges_after_sign_policy` |
| Mean active fraction | `repeats[].mean_active_fraction` (spikes / (N · steps)) and `fraction_spiked_at_least_once` |
| `dt`, seed, dtype, determinism | `workload.dt_ms`, `workload.seed`, `workload.dtype`, `workload.deterministic` |
| Wall-clock time, simulated time, steps/s, events/s | `repeats[].wall_s`, `workload.simulated_ms`, `steps_per_s`, `spikes_per_wall_s`, `synaptic_events_per_wall_s`; distributions in `stats` |
| Host RAM peak | `memory.host_process_peak_rss_bytes` (process lifetime, dataset load included) |
| Allocated / reserved GPU memory | per repeat `gpu_max_memory_allocated_bytes`, `gpu_max_memory_reserved_bytes`; maximum over runs in `memory`; after init in `memory_after_init.device` |
| `nvidia-smi` process peak | `gpu.nvidia_smi_this_process` — usually `visible: false` inside a container under WSL2 (the reason is recorded); `gpu.nvidia_smi_while_loaded[].memory.used` is device-wide and includes other processes (e.g. the homelab LLM) |
| Device-wide CUDA memory snapshots | `gpu.device_memory.before_engine_init`, `.after_engine_init`, `.end`: `device_index`, `free_bytes`, `total_bytes`, `used_bytes` from `torch.cuda.mem_get_info()`; unavailable readings carry `null` values and a `reason` |
| Device-wide increase and allocator difference | `memory.gpu_device_wide_used_delta_after_init_bytes`, `memory.gpu_device_wide_used_delta_end_bytes` relative to the pre-init snapshot; `memory.gpu_device_wide_observed_used_delta_bytes` is the largest eligible device-wide sample delta; `memory.gpu_device_wide_delta_minus_peak_reserved_bytes` is that signed observed delta minus the allocator peak; `gpu.nvidia_smi_device_memory_deltas` records before/while-loaded deltas per `nvidia-smi` row |
| Retained tensors | `memory_after_init.engine_retained_bytes` (edges, state, in-flight spikes) |

## Method

1. Warm-up: `--warmup` untimed runs, default **3** (CUDA context, allocator, kernel caches).
2. Timing: `time.perf_counter()` read after `torch.cuda.synchronize()` at both ends.
3. Repeats: `--repeats` timed runs, default **10**; `stats` reports n, median, quartiles (p25,
   p75, IQR), min, max and mean — never only the best run.
4. Memory: peak allocator stats are reset before each run; allocated and reserved reported apart.
   Outside the timed simulation, capture device-wide free/total CUDA memory before engine
   initialization, after initialization and at the end. Also retain `nvidia-smi` readings before
   initialization and while the engine is loaded, and compute their per-device increases.
5. Phases timed separately (`phases_s`): dataset load, population resolution, sign/removal
   preparation, network build, engine init + copy to the device. The timed region is
   `engine.simulate(steps)` from `reset(seed)`: per-step input generation (Poisson draws), the
   engine step and a counting recorder. A benchmark writes no run artifacts.
6. VRAM gate: `gates.vram_peak_reserved_within_10GiB_target` — peak reserved ≤ 10 GiB on the
   12 GB card. On CUDA, also require
   `gates.vram_observed_device_delta_within_10GiB_target`: the largest sampled device-wide
   increase is ≤ 10 GiB. This uses the CUDA after-init/end deltas and, when exactly one device
   is visible to `nvidia-smi`, its before/while-loaded delta. With multiple devices, the
   `nvidia-smi` rows are reported but excluded from this gate because their ordering need not
   match CUDA logical indices. Unavailable gates are `null`, so a PASS alone does not establish
   the spec's total-process-peak target.
7. Gates: `no_oom`, `all_runs_finite`, the VRAM target. `status: PASS` only if none is false.
8. Benchmark outcomes say nothing about biological plausibility.

Torch's allocator statistics exclude CUDA context, loaded kernel images and allocations made
by libraries outside that allocator. Device-wide increases can expose that extra footprint,
but also include changes by the desktop and other processes sharing the GPU. The signed
difference between the largest observed device-wide delta and torch's peak reserved bytes
compares different measurement scopes and instants: it is an estimate, not a direct measurement of non-allocator
process memory, and it can be negative. A first `mem_get_info()` call can initialize CUDA context
resources before its own baseline; the earlier `nvidia-smi` baseline helps show that overhead.
If an API is unavailable, its readings and derived values remain `null` with a reason.

These are snapshots, not a device-wide peak trace. Even with all memory gates passing, the
total peak attributable to FlyLab is unknown when per-process accounting is unavailable under
WSL2. Keep that limitation next to allocator results (Q20,
[PR-03 finding 8](audit/PR-03-review.md#finding-8--vram-is-reported-from-torchs-allocator-only-about-200-mib-per-run-is-not-counted)).
The committed M3 evidence predates these additional fields and is retained unchanged; the new
device-wide CUDA accounting needs a fresh RTX-host run.

`scripts/profile_step.py` attributes device time using CUDA device-event rows only, avoiding
the duplicate attribution on the CPU operations that launched them. It records host→device
and device→host copy counts separately from their device-side `Memcpy HtoD` and `Memcpy DtoH`
rows; generic `cudaMemcpyAsync` calls do not establish a direction. Profiling includes profiler
overhead and serves attribution, while benchmark timing follows the synchronized method above.

`synaptic_events` counts, for every propagated spike, the edges it is sent along (out-degree);
spikes of the last `delay` steps are counted although they arrive after the end of the run.

## Workloads

| Id | Config | Graph | Duration | Purpose |
|---|---|---|---|---|
| B0 | `configs/bench/b0_synthetic_100ms.yaml` | synthetic random, 20,000 neurons × 20 out-degree | 100 ms | harness check |
| B1 | `configs/bench/b1_male_cns_100ms.yaml` | MaleCNS v1.0 `superclass_annotated` (PR-01) | 100 ms | fits in VRAM, no NaN |
| B2 | `configs/bench/b2_male_cns_1s.yaml` | same | 1 s | M3 gate |

B1/B2 drive every `class == gustatory` neuron with 100 Hz Poisson forced spikes so that activity
propagates through the real connectome. That is an **engineering workload, not an experiment**.

## How to run

On the RTX host (inside WSL2), or from the Mac:

```bash
bash scripts/run_local_gpu_bench.sh                      # on the host
bash scripts/remote/run_gpu_bench.sh -F <ssh_config> <ssh-target>   # from the Mac
bash scripts/remote/fetch_artifacts.sh -F <ssh_config> <ssh-target> 'bench/*.json' 'parity/*.json'
```

The script runs `tests/gpu`, `flylab parity` (B0, B1), `flylab bench` (B0, B1, B2) and a NumPy
timing of B1 for comparison. Single jobs:

```bash
docker compose run --rm gpu flylab bench --config configs/bench/b2_male_cns_1s.yaml --tag B2
docker compose run --rm gpu flylab parity --config configs/bench/b1_male_cns_100ms.yaml --tag B1
```

`--backend`, `--dtype`, `--steps` and `--edge-chunk` override the config (re-validated);
`--warmup`/`--repeats` set the counts. Exit codes: 0 PASS, 3 data error, 4 config error,
5 run error (e.g. no CUDA device), 6 a gate failed.

`scripts/gpu_smoke.py` remains a correctness smoke test (sm_120, a sparse scatter and a short CUDA
vs NumPy LIF lockstep), not a benchmark.

## Engines compared (PR-03b)

`torch-cuda` has two implementations (`engine_impl`, see `docs/EXPERIMENT_CONFIG.md`): `fused`
(Triton kernels, the `auto` choice where it can run) and `eager`. `scripts/run_local_gpu_bench.sh`
times B1 with both and with NumPy (`eager_B1`, `cpu_B1`) and profiles one B1 step loop of each
(`profile_B1`, `profile_B1_eager`; `scripts/profile_step.py` profiles `simulate()`, the batched
path). The engine actually used is recorded in each record under `engine.impl`. Results:
`docs/acceptance/M3b.md`.
