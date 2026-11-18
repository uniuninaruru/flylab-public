# shellcheck shell=bash
# Shared helpers for driving the Windows GPU host from the Mac. Source it; do not run it.
#
# Works whatever the host's OpenSSH DefaultShell is (PowerShell or cmd): every command is sent
# as `powershell -EncodedCommand` (UTF-16LE base64), and WSL commands are base64 again inside
# it, so no quoting survives to be mangled.
#
# Settings (environment):
#   FLYLAB_WSL_DISTRO  WSL distro that runs Docker   (default: homelab)
#   FLYLAB_WSL_USER    non-root user in docker group (default: homelab)
#
# Target syntax for every remote script:  [-F ssh_config] [-o option]... [-i key] <ssh-target>

FLYLAB_WSL_DISTRO=${FLYLAB_WSL_DISTRO:-homelab}
FLYLAB_WSL_USER=${FLYLAB_WSL_USER:-homelab}
SSH_ARGS=()
TARGET=""
REST=()

case "$FLYLAB_WSL_DISTRO$FLYLAB_WSL_USER" in
  *[!A-Za-z0-9._-]*) echo "FLYLAB_WSL_DISTRO / FLYLAB_WSL_USER: only A-Z a-z 0-9 . _ - allowed" >&2; exit 2 ;;
esac

parse_target() {
  while [ $# -gt 0 ]; do
    case "$1" in
      -F|-o|-i|-J) SSH_ARGS+=("$1" "${2:?$1 needs a value}"); shift 2 ;;
      -*) echo "unsupported option: $1 (use -o Port=N for a port)" >&2; return 2 ;;
      *) TARGET=$1; shift; break ;;
    esac
  done
  REST=("$@")
  [ -n "$TARGET" ]
}

# Resolve only local SSH configuration (-G never opens a connection). Keep both the
# configuration and its identifiers out of logs; some failures happen before host redaction.
# The optional SSH_ARGS expansion also supports macOS Bash 3.2's nounset/empty-array behaviour.
configure_remote_redaction() {
  local config key value
  FLYLAB_REDACT="${FLYLAB_REDACT:+$FLYLAB_REDACT,}${TARGET#*@},${TARGET%@*},$FLYLAB_WSL_DISTRO,$FLYLAB_WSL_USER"
  if config=$(ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} -G "$TARGET" 2>/dev/null); then
    while read -r key value; do
      case "$key" in
        hostname|user) FLYLAB_REDACT+=",$value" ;;
      esac
    done <<< "$config"
  fi
  export FLYLAB_REDACT
}

# BatchMode: never prompt for a password; key-based login must already work.
ssh_host() { ssh ${SSH_ARGS[@]+"${SSH_ARGS[@]}"} -o BatchMode=yes -o ConnectTimeout=15 "$TARGET" "$@"; }

ps_encode() {
  # shellcheck disable=SC2016  # $ProgressPreference is PowerShell's
  printf '$ProgressPreference="SilentlyContinue"\n%s' "$1" | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n'
}

# pcps <powershell script> : run on Windows. Our stdin is passed through to it.
pcps() {
  ssh_host "powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $(ps_encode "$1")"
}

# wslbash <bash script> : run as FLYLAB_WSL_USER inside FLYLAB_WSL_DISTRO.
wslbash() {
  local b64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  pcps "wsl.exe -d $FLYLAB_WSL_DISTRO -u $FLYLAB_WSL_USER -- bash -c 'echo $b64 | base64 -d | bash'"
}

# Drop the post-quantum key-exchange notice from Windows OpenSSH; keep every other line.
quiet_ssh_noise() { grep -v -i -e 'post-quantum' -e 'store now, decrypt later' -e 'may need to be upgraded' || true; }
