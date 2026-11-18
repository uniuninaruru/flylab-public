#!/usr/bin/env bash
# Copy one COMMITTED revision of this repo from the Mac to the GPU host's WSL distro.
#
# Usage: bash scripts/remote/sync_to_host.sh [-F ssh_config] <ssh-target> [git-ref]
#   git-ref defaults to HEAD. Uncommitted changes are never sent (git archive).
#
# Manual deployment gate (spec section 14): review the ref first; nothing runs automatically.
# No GitHub credentials are needed on the host.
#
# What it touches on the host:
#   Windows  C:\ProgramData\flylab\flylab-<sha>.tar   (temporary; deleted after unpacking)
#   WSL      ~/flylab of $FLYLAB_WSL_USER in $FLYLAB_WSL_DISTRO: replaced, except artifacts/ and
#            data/ which are kept. The commit hash is written to ~/flylab/.deployed-commit.
# Nothing else (no containers, no services, no settings).

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck source=scripts/remote/common.sh
source "$REPO_ROOT/scripts/remote/common.sh"
parse_target "$@" || { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 2; }
configure_remote_redaction
REF=${REST[0]:-HEAD}

cd "$REPO_ROOT"
SHA=$(git rev-parse --verify "$REF^{commit}")
if [ "$REF" = HEAD ] && [ -n "$(git status --porcelain)" ]; then
  echo "note: working tree has uncommitted changes; they are NOT sent (only $SHA)" >&2
fi

TARBALL=$(mktemp -t flylab-sync).tar
trap 'rm -f "$TARBALL"' EXIT
git archive --format=tar -o "$TARBALL" "$SHA"
echo "==> upload $(du -h "$TARBALL" | cut -f1) for commit $SHA"

pcps 'New-Item -ItemType Directory -Force C:\ProgramData\flylab | Out-Null' 2>&1 \
  | quiet_ssh_noise | python3 "$REPO_ROOT/scripts/redact.py"
scp -q ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} -o BatchMode=yes "$TARBALL" "$TARGET:C:/ProgramData/flylab/flylab-$SHA.tar" 2>&1 \
  | quiet_ssh_noise | python3 "$REPO_ROOT/scripts/redact.py"

echo "==> unpack into WSL: ~/flylab"
wslbash "set -euo pipefail
dest=\"\$HOME/flylab\"
src=/mnt/c/ProgramData/flylab/flylab-$SHA.tar
mkdir -p \"\$dest\"
find \"\$dest\" -mindepth 1 -maxdepth 1 ! -name artifacts ! -name data -exec rm -rf {} +
tar -xf \"\$src\" -C \"\$dest\"
echo $SHA > \"\$dest/.deployed-commit\"
echo \"deployed $SHA to \$dest\"" 2>&1 \
  | quiet_ssh_noise | python3 "$REPO_ROOT/scripts/redact.py"

pcps "Remove-Item -Force C:\\ProgramData\\flylab\\flylab-$SHA.tar" 2>&1 \
  | quiet_ssh_noise | python3 "$REPO_ROOT/scripts/redact.py"
echo "==> done"
