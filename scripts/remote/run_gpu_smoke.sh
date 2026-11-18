#!/usr/bin/env bash
# Run scripts/run_local_gpu_smoke.sh on the GPU host's WSL distro, from the Mac.
#
# Usage: bash scripts/remote/run_gpu_smoke.sh [-F ssh_config] <ssh-target>
# Prerequisite: bash scripts/remote/sync_to_host.sh ... (the code must already be on the host).
#
# The host log stays at ~/flylab/artifacts/gpu/ in WSL; a copy of the (already redacted)
# output is saved here as artifacts/gpu/remote-run-<UTC>.log.
# First run downloads ~3 GB (torch + CUDA wheels) and builds the image: expect minutes.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 2; }
configure_remote_redaction

mkdir -p "$REPO_ROOT/artifacts/gpu"
LOG="$REPO_ROOT/artifacts/gpu/remote-run-$(date -u +%Y%m%dT%H%M%SZ).log"

set +e
# shellcheck disable=SC2016  # $HOME must expand on the host, not here
wslbash 'cd "$HOME/flylab" && bash scripts/run_local_gpu_smoke.sh' 2>&1 \
  | quiet_ssh_noise \
  | python3 "$REPO_ROOT/scripts/redact.py" \
  | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
echo "local copy: ${LOG#"$REPO_ROOT"/}"
exit "$rc"
