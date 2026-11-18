#!/usr/bin/env bash
# Copy result files from the GPU host's ~/flylab/artifacts into this repo's ./artifacts.
#
# Usage: bash scripts/remote/fetch_artifacts.sh [-F ssh_config] <ssh-target> <pattern>...
#   e.g. ... homelab-pc 'bench/*.json' 'parity/*.json' 'gpu/bench-run-*.log'
# Patterns are relative to ~/flylab/artifacts and expanded by the host's shell; only
# A-Z a-z 0-9 . _ - / * are allowed. No absolute paths, leading '-', "..", or
# dot-prefixed path components. Read-only on the host.
# Files are written under ./artifacts (git-ignored). Review them before copying anything into
# docs/acceptance/evidence (they are host-anonymous by design; scan anyway).

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 2; }
[ "${#REST[@]}" -ge 1 ] || { echo "missing <pattern>" >&2; exit 2; }
for pat in "${REST[@]}"; do
  case "$pat" in
    ""|-*|.*|/*|*/.*|*..*|*[!A-Za-z0-9._/*-]*) echo "refused pattern: $pat" >&2; exit 2 ;;
  esac
done
configure_remote_redaction

mkdir -p "$REPO_ROOT/artifacts"
# The patterns are expanded on the host (unquoted on purpose; characters checked above).
# base64 in 76-column lines survives PowerShell's line handling; whitespace is stripped here.
SCRIPT="set -euo pipefail
cd \"\$HOME/flylab/artifacts\"
# Also exclude . and .. when globbing on older Bash versions.
GLOBIGNORE=.:..
shopt -u dotglob
tar -cf - -- ${REST[*]} | base64 -w 76"
wslbash "$SCRIPT" 2>/dev/null | quiet_ssh_noise | tr -d ' \r\n' | base64 -d \
  | tar -C "$REPO_ROOT/artifacts" -xvf - 2>&1 | sed 's/^x /fetched: /' \
  | python3 "$REPO_ROOT/scripts/redact.py"
