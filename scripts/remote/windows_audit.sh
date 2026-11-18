#!/usr/bin/env bash
# Run the read-only Windows audit on the GPU host over SSH, from the Mac.
#
# Usage: bash scripts/remote/windows_audit.sh [-F ssh_config] <ssh-target>
#   e.g. bash scripts/remote/windows_audit.sh -F ~/path/to/homelab/mac/ssh_config homelab-pc
#
# Streams scripts/windows/audit_host.ps1 to PowerShell on the host (nothing is copied or
# installed there), then redacts again locally, including the SSH target name.
# Output: artifacts/system/windows-audit-<UTC>.txt (gitignored). Review before sharing.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,9p' "${BASH_SOURCE[0]}"; exit 2; }
configure_remote_redaction

OUT_DIR="$REPO_ROOT/artifacts/system"
OUT="$OUT_DIR/windows-audit-$(date -u +%Y%m%dT%H%M%SZ).txt"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC2016  # $s is PowerShell's variable
pcps '$s = [Console]::In.ReadToEnd(); Invoke-Expression $s' \
  < "$REPO_ROOT/scripts/windows/audit_host.ps1" 2>&1 \
  | quiet_ssh_noise \
  | python3 "$REPO_ROOT/scripts/redact.py" > "$OUT"

echo "wrote $OUT (review before sharing)"
