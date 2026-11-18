#!/usr/bin/env bash
# Run one FlyLab Compose service on the GPU host's WSL distro, from the Mac.
#
# Usage: bash scripts/remote/compose_run.sh [-F ssh_config] <ssh-target> <service> [command...]
#   e.g. ... homelab-pc data-fetch
#        ... homelab-pc data flylab data inspect --raw data/raw/male-cns-v1.0
#
# Prerequisite: the code is on the host (scripts/remote/sync_to_host.sh).
# Builds the image first (cached layers make this quick when only code changed), then
# `docker compose run --rm <service> [command...]` in ~/flylab. Output is redacted.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 2; }
configure_remote_redaction
[ "${#REST[@]}" -ge 1 ] || { echo "missing <service>" >&2; exit 2; }
SERVICE=${REST[0]}
case "$SERVICE" in
  *[!a-z0-9-]*) echo "invalid service name: $SERVICE" >&2; exit 2 ;;
esac

CMD=""
for arg in "${REST[@]:1}"; do CMD+=" $(printf '%q' "$arg")"; done

SCRIPT="set -euo pipefail
cd \"\$HOME/flylab\"
mkdir -p data/raw data/processed artifacts
export FLYLAB_GIT_COMMIT=\$(cat .deployed-commit 2>/dev/null || echo unknown)
echo \"commit=\$FLYLAB_GIT_COMMIT service=$SERVICE\"
docker compose build --quiet $SERVICE
docker compose run --rm $SERVICE$CMD"

set +e
wslbash "$SCRIPT" 2>&1 | quiet_ssh_noise | python3 "$REPO_ROOT/scripts/redact.py"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
