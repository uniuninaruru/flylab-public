#!/usr/bin/env bash
# M3 GPU gate on THIS machine (the RTX host): CUDA tests, CPU/GPU parity, benchmarks.
#
# Run inside WSL2 (or native Linux) after scripts/run_local_gpu_smoke.sh has passed once. From
# the Mac: bash scripts/remote/run_gpu_bench.sh -F <ssh_config> <ssh-target>
#
# Steps (each recorded PASS/FAIL; the script keeps going):
#   build          docker compose build (cached layers: quick when only code changed)
#   gpu_tests      docker compose run --rm gpu-tests            (tests/gpu --run-gpu)
#   parity_B0      flylab parity, synthetic 20k graph, CUDA vs NumPy lockstep, 100 ms
#   parity_B1      flylab parity, full MaleCNS graph, 100 ms     (needs the processed graph)
#                  (torch-cuda runs `engine_impl: auto`: the fused engine where it can run)
#   bench_B0/B1/B2 flylab bench, 3 warm-up + 10 timed repeats    (B1 = 100 ms, B2 = 1 s)
#   bench_B0_1s    B0 workload run for 1 s (1 warm-up + 3 repeats)
#   eager_B1       B1 with the eager (one torch op per phase) CUDA engine, for comparison
#   cpu_B1         NumPy reference timing of B1 (1 warm-up + 3 repeats), for comparison
#   profile_B1     torch.profiler attribution of a CUDA step (scripts/profile_step.py)
# Without data/processed/... the MaleCNS steps are SKIPPED_NO_DATA, which is not a pass.
# The containers have no network; nothing here opens ports or changes settings.
# Outputs: artifacts/bench/*.json, artifacts/parity/*.json, artifacts/gpu/profile-*.json,
#          artifacts/gpu/bench-run-<UTC>.log
# Override counts with FLYLAB_BENCH_WARMUP / FLYLAB_BENCH_REPEATS (defaults 3 / 10).

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT" || exit 1

for tool in docker python3; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool" >&2; exit 1; }
done

WARMUP=${FLYLAB_BENCH_WARMUP:-3}
REPEATS=${FLYLAB_BENCH_REPEATS:-10}
case "$WARMUP$REPEATS" in *[!0-9]*) echo "warm-up/repeats must be integers" >&2; exit 2 ;; esac
PROCESSED=data/processed/male-cns-v1.0/superclass_annotated

mkdir -p artifacts/gpu artifacts/bench artifacts/parity
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ -f .deployed-commit ]; then
  FLYLAB_GIT_COMMIT=$(cat .deployed-commit)
elif [ "$(git -C "$REPO_ROOT" rev-parse --show-toplevel 2>/dev/null)" = "$REPO_ROOT" ]; then
  FLYLAB_GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
else
  FLYLAB_GIT_COMMIT=unknown
fi
export FLYLAB_GIT_COMMIT
LOG="artifacts/gpu/bench-run-$STAMP.log"
declare -A STATUS=()
ORDER=()

step() {  # step <name> <command...>; records PASS/FAIL, keeps going
  local name=$1; shift
  ORDER+=("$name")
  printf '\n## %s\n$ %s\n' "$name" "$*"
  if "$@"; then STATUS[$name]=PASS; else STATUS[$name]="FAIL(exit $?)"; fi
}

skip() {  # skip <name> <reason>
  ORDER+=("$1")
  STATUS[$1]="SKIPPED_$2"
  printf '\n## %s\nskipped: %s\n' "$1" "$2"
}

# shellcheck disable=SC2329  # called through step()
gpu() { docker compose run --rm gpu flylab "$@"; }

main() {
  echo "# FlyLab local GPU benchmark run (M3)"
  echo "started_utc=$STAMP"
  echo "git_commit=$FLYLAB_GIT_COMMIT"
  echo "warmup=$WARMUP repeats=$REPEATS"
  # The GPU may be shared (e.g. with an LLM server). Record what is already in use.
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader \
    2>/dev/null | sed 's/^/gpu_before=/'

  step build docker compose build --quiet
  step gpu_tests docker compose run --rm gpu-tests
  step parity_B0 gpu parity --config configs/bench/b0_synthetic_100ms.yaml --tag B0
  if [ -f "$PROCESSED/graph.npz" ]; then
    step parity_B1 gpu parity --config configs/bench/b1_male_cns_100ms.yaml --tag B1
  else
    skip parity_B1 NO_DATA
  fi
  step bench_B0 gpu bench --config configs/bench/b0_synthetic_100ms.yaml --tag B0 \
    --warmup "$WARMUP" --repeats "$REPEATS"
  step bench_B0_1s gpu bench --config configs/bench/b0_synthetic_100ms.yaml --tag B0-1s \
    --steps 10000 --warmup 1 --repeats 3
  if [ -f "$PROCESSED/graph.npz" ]; then
    step bench_B1 gpu bench --config configs/bench/b1_male_cns_100ms.yaml --tag B1 \
      --warmup "$WARMUP" --repeats "$REPEATS"
    step bench_B2 gpu bench --config configs/bench/b2_male_cns_1s.yaml --tag B2 \
      --warmup "$WARMUP" --repeats "$REPEATS"
    step eager_B1 gpu bench --config configs/bench/b1_male_cns_100ms.yaml --tag B1-eager \
      --engine-impl eager --warmup 1 --repeats 3
    step cpu_B1 gpu bench --config configs/bench/b1_male_cns_100ms.yaml --tag B1-numpy \
      --backend numpy --warmup 1 --repeats 3
    step profile_B1 docker compose run --rm gpu python scripts/profile_step.py \
      --config configs/bench/b1_male_cns_100ms.yaml --tag B1
    step profile_B1_eager docker compose run --rm gpu python scripts/profile_step.py \
      --config configs/bench/b1_male_cns_100ms.yaml --tag B1-eager --engine-impl eager
  else
    skip bench_B1 NO_DATA
    skip bench_B2 NO_DATA
    skip eager_B1 NO_DATA
    skip cpu_B1 NO_DATA
    skip profile_B1 NO_DATA
    skip profile_B1_eager NO_DATA
  fi

  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/gpu_after=/'
  echo
  echo "## Summary"
  local overall=PASS
  for name in "${ORDER[@]}"; do
    echo "$name=${STATUS[$name]}"
    [ "${STATUS[$name]}" = PASS ] || overall=FAIL
  done
  echo "overall=$overall"
  echo "records: artifacts/bench/ artifacts/parity/"
  [ "$overall" = PASS ]
}

main 2>&1 | python3 scripts/redact.py | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "log: $LOG"
exit "$rc"
