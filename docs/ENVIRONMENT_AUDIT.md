# Environment audit

Status legend: **VERIFIED** (seen directly, with date), **UNVERIFIED** (not yet inspected).
Never record host names, addresses or user names here; use `<gpu-host>`.

This public file keeps the hardware and software facts that FlyLab's results depend on. How the
owner's machine is reached and what else runs on it are kept out of the public repository.

## Chosen deployment path (owner decision, 2026-11-04)

**Windows 11 host + WSL2 Ubuntu + Docker, operated from a Mac over SSH.** This is spec section
2.2 option 3 with containers. The GPU driver is the Windows NVIDIA driver; WSL2 and containers
reuse it. No Linux NVIDIA driver is installed. FlyLab runs as a guest on a machine that also runs
other workloads: its own directory, its own Compose project (`flylab`), no published ports, and
it never changes the host's other services or settings.

## Mac client (VERIFIED 2026-11-04)

| Field | Value |
|---|---|
| Role | Development + SSH client only (never runs CUDA) |
| Tools | uv 0.12.17, git 2.54, gh, Python 3.12.15 via uv for the project |
| Docker | not installed (not needed on the Mac) |
| SSH | key-only, dedicated key and pinned host key; scripts take `-F <ssh_config> <ssh-target>` |

## GPU host (VERIFIED 2026-11-04, read-only)

Evidence: `scripts/remote/windows_audit.sh` report and a read-only check inside WSL (raw reports
stay in `artifacts/system/`, gitignored).

| Field | Value |
|---|---|
| OS | Windows 11 Home, build 26200 (10.0.26200.9711) |
| CPU / RAM | AMD Ryzen 7 9700X (8 cores) / 31.1 GB |
| GPU | NVIDIA GeForce RTX 5070, **compute capability 12.0**, 12,227 MiB |
| Driver | **616.92** (≥ 580 required for CUDA 13.x wheels → OK) |
| GPU memory in use before FlyLab | 2,374–2,602 MiB, used by the Windows desktop → ~9.4–9.6 GiB left. Per-process figures from Windows counters overlap (shared surfaces) and do not add up to the adapter total. |
| WSL | WSL 2.7.10.0, WSLg 1.0.73.2 |
| WSL distro used | Ubuntu 24.04.5 LTS, kernel 6.18.33.2-microsoft-standard-WSL2, systemd, 15 GiB RAM visible |
| Docker used by FlyLab | **Docker Engine 29.9.2 inside WSL**, Compose v5.6.1, NVIDIA Container Toolkit 1.20.1, runtimes `nvidia` + `runc`, CDI dirs `/etc/cdi`, `/run/cdi` |
| Non-root user | uid 1000 in group `docker` → FlyLab runs without root |
| Tools in WSL | git present; uv absent (not needed: uv is inside the image) |

## GPU smoke test on the host (VERIFIED 2026-11-04)

Commit `2d4bc9a` synced with `scripts/remote/sync_to_host.sh`, then
`scripts/remote/run_gpu_smoke.sh`. Evidence (redacted):
`docs/acceptance/evidence/gpu-run-20261104T020421Z.log`,
`docs/acceptance/evidence/gpu-smoke-20261104T021216Z.json`.

| Step | Result |
|---|---|
| `scripts/audit_host.sh` inside WSL | PASS (read-only, 0 identifiers leaked) |
| Docker GPU passthrough (`nvidia/cuda:13.0.2-base-ubuntu24.04`, pinned) | PASS — in-container NVIDIA-SMI 615.71.08, KMD 616.92, CUDA UMD 13.4 |
| `docker compose build` | PASS — image `flylab:dev` 9.19 GB; whole run ~8 min incl. ~3 GB download |
| `gpu_smoke.py` | **PASS** — torch `2.15.0+cu132` (CUDA 13.2), RTX 5070, capability (12, 0), arch list `sm_75 … sm_100, sm_120`, matmul finite |
| Sparse propagation CPU vs CUDA | PASS — synthetic 20,000 neurons / 399,800 edges, max abs diff **0.0** (integer addends, exact by construction) |
| GPU memory of the smoke test | 40 MiB allocated / 56 MiB reserved peak (tiny workload; not a capacity measurement) |
| CPU tests inside the container | PASS — 101 passed, 1 skipped (the commit's suite) |
| Provenance | run JSON records commit `2d4bc9a…` via `FLYLAB_GIT_COMMIT` |

Full-graph runs, 100 ms / 1 s timings and the fused engine: `docs/acceptance/M3.md`, `M3b.md`.

## GPU build (smoke-tested on the host 2026-11-04; see above)

| Layer | Choice | Why |
|---|---|---|
| Image base | `ubuntu:24.04` pinned by digest | Plain OS; CUDA runtime comes from the torch wheel, driver from the host |
| Python | 3.12 (Ubuntu package) | Mature wheel coverage |
| torch | 2.15.0, locked in `uv.lock` with hashes (`gpu` extra) | CUDA 13.2 build with Blackwell support; `sm_120` checked on host |
| Driver floor | ≥ 580 | CUDA 13.x minor-version compatibility |
| GPU check | `scripts/run_local_gpu_smoke.sh` | passthrough → build → `gpu_smoke.py` → CPU tests, redacted log |

## Unresolved compatibility risks

1. **Shared GPU.** Other workloads on the host (the desktop, and any model server that keeps a
   model in VRAM) reduce what FlyLab can use. FlyLab's ≤ 10 GiB target does not fit while such a
   model is loaded. Before full-graph runs, check free VRAM.
2. **Shared CPU/RAM.** The WSL distro sees 15 GiB RAM shared with other services; the image build
   and large imports compete with them.
3. ~~torch 2.15.0 `sm_120` kernels~~ — observed on the host (`get_arch_list()` includes `sm_120`).
4. A long run launched over SSH ends if the SSH session drops.
