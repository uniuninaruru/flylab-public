#!/usr/bin/env bash
# Run scripts/run_local_gpu_bench.sh (M3 gate: CUDA tests, parity, benchmarks) on the GPU host's
# WSL distro, from the Mac.
#
# Usage: bash scripts/remote/run_gpu_bench.sh [-F ssh_config] <ssh-target>
# Prerequisite: bash scripts/remote/sync_to_host.sh ... (the code must already be on the host).
# The host keeps everything under ~/flylab/artifacts/; a copy of the (already redacted) output is
# saved here as artifacts/gpu/remote-bench-<UTC>.log. Fetch the JSON records afterwards with
#   bash scripts/remote/fetch_artifacts.sh -F ssh_config <ssh-target> 'bench/*.json' 'parity/*.json'
# Expect several minutes (the full-graph steps load ~25.6 M edges per job).

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2; }
configure_remote_redaction

mkdir -p "$REPO_ROOT/artifacts/gpu"
LOG="$REPO_ROOT/artifacts/gpu/remote-bench-$(date -u +%Y%m%dT%H%M%SZ).log"

set +e
# shellcheck disable=SC2016  # $HOME must expand on the host, not here
wslbash 'cd "$HOME/flylab" && bash scripts/run_local_gpu_bench.sh' 2>&1 \
  | quiet_ssh_noise \
  | python3 "$REPO_ROOT/scripts/redact.py" \
  | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "local copy: ${LOG#"$REPO_ROOT"/}"
exit "$rc"
