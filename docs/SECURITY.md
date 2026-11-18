# Security

## Topology

```
Mac (client) --SSH over Tailscale--> Windows GPU host (OpenSSH Server)
                                       └─ wsl.exe → WSL2 Ubuntu → docker compose (FlyLab containers)
```

Code reaches the host only through the reviewed GitHub repository. The owner reviews changes
before any script runs on the host (manual deployment gate, spec section 14).

## Rules

1. **No public exposure.** Never port-forward 22, 8765 or the Docker daemon on the router.
   Future API: bind `127.0.0.1:8765` only; reach it from the Mac with
   `ssh -N -L 8765:127.0.0.1:8765 <ssh-alias>`.
2. **Key-based SSH.** The owner's homelab SSH config: dedicated ed25519 key file, pinned host key,
   `StrictHostKeyChecking yes`, no agent. Remote scripts add `BatchMode=yes`, so they never prompt.
   Claude never reads private keys and never types passwords.
3. **sshd only from the tailnet.** VERIFIED 2026-11-04 on the GPU host: the sshd firewall rule
   allows only 100.64.0.0/10 and fd7a:115c:a1e0::/48, password authentication is off. (On a new
   host, Windows' default `OpenSSH-Server-In-TCP` rule allows any address; restrict it the same
   way, as an owner action.)
4. **Guest on the homelab.** The GPU host also runs the owner's homelab services. FlyLab uses its
   own directory and Compose project, publishes no ports, and never stops, restarts or deletes
   homelab containers or volumes, and never changes network, firewall, SSH, Tailscale or WSL
   settings. Power actions (shutdown/reboot) are the owner's.
5. **Containers.** No `--privileged`, no Docker socket mount, no host network. Test jobs run with
   `network_mode: none`, all capabilities dropped, `no-new-privileges`, as the unprivileged `ubuntu`
   user. Data is mounted read-only.
6. **No self-hosted GitHub runner** on the GPU host for unreviewed PR code.
7. **Secrets.** Never in Git or transcripts: SSH keys, Tailscale auth keys, neuPrint tokens,
   GitHub tokens. `.env` is gitignored; `.env.example` holds names only.
8. **Logs.** Host audits and smoke logs pass through `scripts/redact.py` (hostname, user, home,
   IPs except 0.0.0.0/127.*, IPv6, MAC, e-mail, `*.ts.net`). Review before committing excerpts.
9. **API (PR-06).** No shell-execution endpoint, no arbitrary path reads, validated payload sizes,
   bounded job queue.
10. Keep Tailscale and OpenSSH up to date (Tailscale 2026 advisory:
   https://tailscale.com/blog/tailscale-ssh-findings-fixes).
