#!/usr/bin/env bash
# FlyLab host audit for Linux and WSL2 (spec section 2.3).
#
# READ-ONLY: installs nothing, changes nothing, needs no sudo, pulls no images.
# Output is passed through scripts/redact.py (hostname, user, home path, IPs, MACs, e-mail,
# tailnet names removed). Tailscale peers are never listed. Review before sharing.
#
# Usage: bash scripts/audit_host.sh [--out FILE | --stdout]
#   default output: artifacts/system/host-audit-<UTC>.txt

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=""
TO_STDOUT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=${2:?--out needs a path}; shift 2 ;;
    --stdout) TO_STDOUT=1; shift ;;
    -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
  echo "python3 is required for redaction; refusing to write an unredacted report" >&2
  exit 1
fi

have() { command -v "$1" >/dev/null 2>&1; }
section() { printf '\n## %s\n' "$1"; }
run() {  # run a read-only command, show it, never fail the audit
  printf '$ %s\n' "$*"
  "$@" 2>&1 || printf '(exit %s)\n' "$?"
}

driver_check() {
  local ver major
  ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
  major=${ver%%.*}
  if [ -z "$ver" ] || ! [[ $major =~ ^[0-9]+$ ]]; then
    echo "driver_check=UNKNOWN (nvidia-smi gave no driver version)"
  elif [ "$major" -ge 580 ]; then
    echo "driver_check=OK driver=$ver (>= 580 needed by torch 2.15.0 / CUDA 13.2 wheels)"
  elif [ "$major" -ge 570 ]; then
    echo "driver_check=TOO_OLD_FOR_CUDA13 driver=$ver (Blackwell OK for CUDA 12.8; FlyLab image needs >= 580)"
  else
    echo "driver_check=TOO_OLD driver=$ver (RTX 50 series needs >= 570 for CUDA 12.8, >= 580 for CUDA 13.x)"
  fi
}

audit() {
  echo "# FlyLab host audit (read-only)"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "script=scripts/audit_host.sh"
  echo "redacted=hostname,user,home,ipv4(except 0.0.0.0/127.*),ipv6,mac,email,*.ts.net"

  section "Kernel and OS"
  run uname -srvm
  if [ -r /etc/os-release ]; then run cat /etc/os-release; else echo "/etc/os-release: not present"; fi
  if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "wsl2=yes"
    echo "note: in WSL2 the GPU driver is the Windows driver; do not install a Linux NVIDIA driver here"
  else
    echo "wsl2=no"
  fi

  section "CPU and memory"
  if have lscpu; then lscpu 2>/dev/null | grep -E '^(Model name|CPU\(s\)|Thread|Core)' ; fi
  if have sysctl && [ "$(uname -s)" = Darwin ]; then run sysctl -n machdep.cpu.brand_string hw.memsize; fi
  if have free; then run free -h; fi

  section "Disk (free space only; nothing is modified)"
  run df -h / "$HOME" "$REPO_ROOT"

  section "NVIDIA GPU"
  if have nvidia-smi; then
    run nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv
    nvidia-smi 2>/dev/null | grep -Eo 'CUDA Version: *[0-9.]+' | head -n1
    driver_check
  else
    echo "nvidia-smi: not found (in WSL2 it should be at /usr/lib/wsl/lib/nvidia-smi)"
  fi

  section "Containers"
  if have docker; then
    run docker --version
    run docker info --format 'server={{.ServerVersion}} os={{.OperatingSystem}} default_runtime={{.DefaultRuntime}} runtimes={{range $k, $v := .Runtimes}}{{$k}} {{end}}'
    run docker compose version
  else
    echo "docker: not found"
  fi
  if have nvidia-ctk; then run nvidia-ctk --version; else echo "nvidia-ctk: not found (not needed with Docker Desktop)"; fi

  section "Toolchain"
  run python3 --version
  if have uv; then run uv --version; else echo "uv: not found (not needed on the host; it is inside the image)"; fi
  if have git; then run git --version; else echo "git: not found"; fi

  section "Remote access (no peer list is recorded)"
  if have tailscale; then
    run tailscale version
    tailscale status --json 2>/dev/null \
      | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("tailscale_backend_state=" + str(d.get("BackendState")))' \
      || echo "tailscale_backend_state=UNKNOWN"
  else
    echo "tailscale: not found in this environment"
  fi

  section "Listening TCP sockets (0.0.0.0 / [::] means reachable from every interface)"
  if have ss; then run ss -tln
  elif have netstat; then netstat -an 2>/dev/null | grep -i listen | head -n 40
  else echo "ss/netstat: not found"
  fi
}

REPORT=$(audit | "$PY" "$REPO_ROOT/scripts/redact.py")

if [ "$TO_STDOUT" -eq 1 ]; then
  printf '%s\n' "$REPORT"
  exit 0
fi
if [ -z "$OUT" ]; then
  mkdir -p "$REPO_ROOT/artifacts/system"
  OUT="$REPO_ROOT/artifacts/system/host-audit-$(date -u +%Y%m%dT%H%M%SZ).txt"
fi
printf '%s\n' "$REPORT" > "$OUT"
echo "wrote $OUT (review before sharing)"
