# FlyLab

Reproducible, inspectable spiking-network simulation built on the *Drosophila* **MaleCNS v1.0**
connectome, targeting an NVIDIA RTX 5070 (12 GB, `sm_120`) on the owner's own machine.

> **Status: milestones M3/M3b (CUDA backend) and M4 (scientific validation).** The LIF engine
> runs on the RTX 5070 (`sm_120`) over the full MaleCNS v1.0 graph (166,700 neurons, 24.6 M signed
> edges) with ~0.25 GiB reserved by torch's allocator; CUDA context and other device overhead add
> to that footprint (see M3b). Its spikes match the NumPy reference exactly on the runs measured
> so far (up to 1 s). With the fused Triton engine (PR-03b) a 100 ms full-graph run takes about
> 0.1 s, i.e. about real time for the measured, sparse workloads. PR-04 adds replicated MaleCNS runs
> with controls, a `w_syn` calibration and a cross-check on Shiu et al.'s FlyWire graph (CPU NumPy
> engine). Results: [`M3`](docs/acceptance/M3.md), [`M3b`](docs/acceptance/M3b.md),
> [`M4`](docs/acceptance/M4.md); earlier milestones: [M0](docs/acceptance/M0.md),
> [M1](docs/acceptance/M1.md), [M2](docs/acceptance/M2.md). These are engineering results:
> parameters are uniform and partly ASSUMED, every simulated rate is a model output, and nothing
> here is a validated model of fly behaviour.

The design is in [`docs/IMPLEMENTATION_SPEC.md`](docs/IMPLEMENTATION_SPEC.md). Scientific scope and
non-claims: [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md).

## What exists now

- `flylab.data`: deterministic `body_id → index` map; directed sparse graph
  (`src` = presynaptic → `dst` = postsynaptic) with explicit duplicate and unknown-sign rules;
  no dense N×N matrices; registry of the official MaleCNS v1.0 files with upstream sizes/hashes.
- `flylab.provenance`: the six evidence labels, streaming SHA-256 / GCS-MD5 hashing, host-anonymous
  environment capture.
- `flylab data fetch | inspect | prepare | verify`: download the official files with hash checks,
  audit their schema, build the processed graph for a named node set, verify it.
- `flylab.neural`: discrete-time LIF engines — NumPy reference, PyTorch on CPU and CUDA — with
  stimuli, silencing and synapse removal, checkpoints, and a lockstep CPU↔GPU comparison.
- `flylab run | bench | parity`: experiment runs with full provenance, benchmarks (warm-up,
  repeats, synchronized timing, peak memory) and backend parity checks.
- `flylab replicate run | stats`: seeds × conditions with paired bootstrap statistics, a
  degree-preserving null graph and `w_syn` calibration (PR-04, `docs/acceptance/M4.md`).
- `flylab` CLI also: `version`, `env`, `sources`, `graph inspect`, `memory`.
- Docker image (Ubuntu 24.04 + locked torch 2.15.0) and host scripts for the Windows + WSL2 + Docker
  GPU host, reached from the Mac over SSH/Tailscale.

## Development (Mac or Linux, CPU only)

```bash
uv sync --locked --group dev
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
uv run flylab graph inspect --neurons tests/fixtures/tiny_graph/neurons.csv \
  --edges tests/fixtures/tiny_graph/edges.csv --duplicates sum
```

For the complete torch CPU suite, including Linux without CUDA downloads:

```bash
uv sync --locked --group dev --extra cpu
FLYLAB_REQUIRE_TORCH=1 uv run pytest
```

On Linux, `cpu` selects the hashed `torch==2.15.0+cpu` wheel from the explicit
[PyTorch CPU index](https://download.pytorch.org/whl/cpu). On macOS it selects the existing
PyPI CPU wheel. The `cpu` and `gpu` extras conflict; choose one (do not use `--all-extras`).
The `gpu` extra and Docker build retain their existing PyPI CUDA resolution on Linux.
CI tests Linux CPU wheels on Python 3.12/3.13 and retains macOS to cover its separate PyPI build.
Index configuration follows [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Local dashboard (optional, UI mock)

```bash
uv sync --locked --group dev --extra api
uv run --extra api python -m flylab.api --repo .
# http://127.0.0.1:8765
```

Start/stop an allowlisted experiment and view binned population rates, sampled spikes,
active interventions and labelled provenance. This is an engineering simulation, not validated
behaviour. The body view is a placeholder and GPU runs through this interface are UNVERIFIED.
Setup, private-tunnel instructions and limits: [DASHBOARD.md](docs/DASHBOARD.md).
Screenshots and executed checks: [M7](docs/acceptance/M7.md).

## GPU host (Linux or Windows 11 + WSL2, Docker + NVIDIA Container Toolkit)

GPU work runs in containers on a machine with an NVIDIA GPU, Docker Engine and the NVIDIA
Container Toolkit (measured so far: Windows 11 + WSL2 Ubuntu 24.04, RTX 5070; see
[`docs/ENVIRONMENT_AUDIT.md`](docs/ENVIRONMENT_AUDIT.md)). FlyLab uses its own directory and
Compose project and publishes no ports. On that machine:

```bash
bash scripts/run_local_gpu_smoke.sh   # GPU passthrough, image build, GPU smoke test
bash scripts/run_local_gpu_bench.sh   # M3: GPU tests, CPU/GPU parity, benchmarks
```

From another machine over SSH (`-F` = an ssh_config with a key-only entry for the host):

```bash
bash scripts/remote/windows_audit.sh -F <ssh_config> <ssh-target>   # read-only
bash scripts/remote/sync_to_host.sh  -F <ssh_config> <ssh-target>   # send a committed revision
bash scripts/remote/run_gpu_smoke.sh -F <ssh_config> <ssh-target>   # build + GPU smoke test
bash scripts/remote/run_gpu_bench.sh -F <ssh_config> <ssh-target>   # M3: tests, parity, benchmarks
```

The remote scripts assume a WSL distro and user named by `FLYLAB_WSL_DISTRO` /
`FLYLAB_WSL_USER` (see `scripts/remote/common.sh`).

## Data

No connectome data is stored in this repository. MaleCNS v1.0 is © its authors and licensed
CC BY 4.0 — see [`NOTICE`](NOTICE) and [`docs/SOURCES_AND_LICENSES.md`](docs/SOURCES_AND_LICENSES.md).

## License

Code: Apache-2.0 ([`LICENSE`](LICENSE)).
