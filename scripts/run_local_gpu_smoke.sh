#!/usr/bin/env bash
# Build the FlyLab image and run the GPU smoke test on THIS machine (the RTX host).
#
# Run inside WSL2 Ubuntu with Docker Desktop's WSL integration on (or on native Linux with
# the NVIDIA Container Toolkit). Typical use from the Mac:
#   ssh <user>@<gpu-host> wsl.exe -d <distro> --cd ~/flylab -- bash scripts/run_local_gpu_smoke.sh
#
# Steps: (1) Docker GPU passthrough check with a digest-pinned nvidia/cuda image,
#        (2) docker compose build, (3) gpu-smoke container, (4) cpu-tests container.
# Network use: pulls the pinned base images and the locked Python wheels (torch + CUDA ~3 GB).
# It does not install drivers, change Docker or OS settings, open ports or run as root.
# Log (redacted): artifacts/gpu/run-<UTC>.log ; smoke JSON: artifacts/gpu/smoke-<UTC>.json

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 1

# CUDA 13.0 base image: its driver requirement (>= 580) matches the torch 2.15.0 wheels.
CUDA_CHECK_IMAGE="nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:2ab6381d970b211fb93853796dc6707eb8a72575a375c422b17cf4d8b2641701"

for tool in docker python3 git; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool" >&2; exit 1; }
done

mkdir -p artifacts/gpu data
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# Commit of this code. A git-archive deployment has a marker written by sync_to_host.sh; git is
# trusted only when its top level is this repository (not, e.g., a versioned home directory).
is_flylab_checkout() { [ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" = "$REPO_ROOT" ]; }
if [ -f .deployed-commit ]; then
  FLYLAB_GIT_COMMIT=$(cat .deployed-commit)
elif is_flylab_checkout; then
  FLYLAB_GIT_COMMIT=$(git rev-parse HEAD)
else
  FLYLAB_GIT_COMMIT=unknown
fi
export FLYLAB_GIT_COMMIT
LOG="artifacts/gpu/run-$STAMP.log"
declare -A STATUS=()

step() {  # step <name> <command...>; records PASS/FAIL, keeps going
  local name=$1; shift
  printf '\n## %s\n$ %s\n' "$name" "$*"
  if "$@"; then STATUS[$name]=PASS; else STATUS[$name]="FAIL(exit $?)"; fi
}

main() {
  echo "# FlyLab local GPU smoke run"
  echo "started_utc=$STAMP"
  echo "git_commit=$FLYLAB_GIT_COMMIT"
  if [ ! -f .deployed-commit ] && is_flylab_checkout; then
    echo "git_uncommitted_files=$(git status --porcelain | wc -l | tr -d ' ')"
  else
    echo "deployed_without_git=yes (exact copy of the commit above, via git archive)"
  fi
  grep -qi microsoft /proc/version 2>/dev/null && echo "wsl2=yes" || echo "wsl2=no"
  # The GPU may be shared (e.g. with an LLM server). Record what is already in use.
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/gpu_memory_used_total_before=/'

  step passthrough docker run --rm --gpus all "$CUDA_CHECK_IMAGE" nvidia-smi
  step build docker compose build
  step gpu_smoke docker compose run --rm gpu-smoke
  step cpu_tests docker compose run --rm cpu-tests

  echo
  echo "## Summary"
  local overall=PASS
  for name in passthrough build gpu_smoke cpu_tests; do
    echo "$name=${STATUS[$name]:-NOT_RUN}"
    [ "${STATUS[$name]:-}" = PASS ] || overall=FAIL
  done
  echo "overall=$overall"
  [ "$overall" = PASS ]
}

main 2>&1 | python3 scripts/redact.py | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "log: $LOG"
exit "$rc"
