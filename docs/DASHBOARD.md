# Local neural dashboard

The dashboard starts one allowlisted experiment, streams sampled neural activity and records
the normal experiment artifacts. It is an **engineering simulation, not validated behaviour**.
The body panel is a placeholder. No body, sensory-to-body loop or decoder is implemented here.

## Run locally

From the repository root:

```bash
uv sync --locked --group dev --extra api
uv run --extra api python -m flylab.api --repo .
# Open http://127.0.0.1:8765
```

The entrypoint always binds **127.0.0.1**, with one worker. There is no host/bind override.
Use `--port 8766` if the default port is already occupied; do not stop another service to use it.
Starlette provides the small ASGI HTTP/WebSocket layer; Uvicorn and websockets provide the
server and transport. All browser assets ship inside the Python package, with no frontend
build, CDN, analytics, external font request or JavaScript dependency.

Choose a config, seed, backend and step count, then select **Start run**. **Stop** requests
cooperative cancellation. Simulation time and measured wall time are separate. Population
rates include all spikes counted for each recorded population; the raster is explicitly a
sample. The table provides current rates as text alongside the canvas plots.

`synthetic.yaml` and `tiny_fixture.yaml` work without biological data or a GPU. Selecting
an uninstalled backend or absent processed dataset produces a failed run; it does not
silently substitute a different dataset or backend. No data is downloaded by the service.
The datasets endpoint lists only datasets actually loaded and verified in this session.

For a slower visual preview, use:

```bash
uv run --extra api python -m flylab.api --repo . --port 8766 --demo-pace-ms 100
```

This inserts a visible, documented wall-time delay per bin. It does not change model dt,
inputs or neural state updates. Run-side wall-time observations include the observer/pacing
overhead and are **not benchmarks**. The ordinary CLI benchmark remains the performance path.

## Private access from a Mac (spec section 10)

On a reviewed, trusted GPU installation, run the service on that machine's loopback interface.
For a torch backend, install its existing `gpu` extra as well; this dashboard does not alter
the GPU image, Dockerfile or Compose configuration. GPU execution through this UI is
**UNVERIFIED** until the owner tests it on that host.

On the Mac, use the existing private OpenSSH-over-Tailscale setup:

```bash
ssh -N -L 8765:127.0.0.1:8765 <user>@<tailscale-host>
# Open http://127.0.0.1:8765 on the Mac.
```

Replace the placeholders locally or use the owner's existing SSH config alias. The dashboard
does not establish tunnels, handle SSH keys, configure Tailscale SSH, deploy code or change the
host network. No remote connection was made to implement or verify this PR. Do not publish
the service port publicly or replace the bind address with a wildcard. No new Compose port
mapping is introduced.

## API

| Method / endpoint | Contract |
|---|---|
| `GET /api/v1/health` | status, active run ID, limits and GPU verification notice |
| `GET /api/v1/configs` | frozen allowlist names and basic config metadata |
| `GET /api/v1/datasets` | successfully loaded, verified dataset summaries; initially empty |
| `POST /api/v1/experiments` | JSON object with exact `config` name and optional `overrides` |
| `GET /api/v1/experiments/{id}` | state, manifest summary, final metrics and measured elapsed time |
| `POST /api/v1/experiments/{id}/cancel` | empty JSON object; cooperative stop |
| `GET /api/v1/experiments/{id}/artifacts` | completed-run filenames/sizes only |
| `WS /api/v1/experiments/{id}/stream` | manifest, binned rates/sampled spikes, terminal status |

Example request (a configuration choice, not a measurement):

```json
{"config":"tiny_fixture.yaml","overrides":{"seed":42,"backend":"numpy","steps":1000}}
```

Only `seed`, `backend` and `steps` can be overridden. Booleans, fractional/infinite/out-of-range
integers, unknown fields, invalid time grids and unknown backends are rejected. Config names
are exact keys, not filesystem paths. Only top-level `configs/*.yaml` and
`configs/bench/*.yaml` are loaded at service startup; symlinks and oversized configs are
excluded, and a symlinked allowlist root is rejected. Other config directories are excluded.
Restart after editing the operator-controlled allowlist; later file edits do not change
the frozen config snapshots. Research parameters and evidence labels are preserved.

Concurrent starts return 409. The service does not enqueue jobs. Cancelled runs retain
`.partial` artifacts, with writers closed, and are never listed as completed runs.
Artifacts are listed only by generated run ID; there is no arbitrary path/download or shell
endpoint. Processed dataset paths come from the trusted config, not a request.

## Bounds and security

| Resource | Bound / behaviour |
|---|---|
| Run concurrency | 1; no waiting queue |
| Session history | 32 runs; restart after archiving outputs locally |
| Steps / loaded graph | 100,000 steps; 200,000 neurons; 30,000,000 edges |
| Spike work | 2,000,000 actual spike events per run; exceeding it fails the run |
| Wall time | 300 seconds, checked cooperatively by the recorder |
| Disk admission / observation | 512 MiB; admission and per-bin checks, not an OS filesystem quota |
| Large lesion artifacts | graphs over 100,000 edges with edge-removal interventions require offline CLI |
| Recorded populations | 32 |
| Event history | one shared 64-message ring per run, oldest messages dropped |
| Raster / frame | deterministic sample of at most 64 recorded neurons; at most 128 sampled spikes |
| WS subscribers | 4 total; each keeps only a sequence cursor |
| WS frames / sends | 32 KiB outgoing; 4 KiB incoming transport limit; 1-second send deadline |
| Transport | incoming queue 4; HTTP connection concurrency 16; no proxy-header trust |
| POST payload | 4096 bytes, checked against actual streamed bytes as well as Content-Length |
| Browser history | last 200 bins and 4000 sampled spikes |

Sequence numbers and `dropped_before` expose missed history to lagging clients.
`spikes_omitted` reports sampled events removed by the per-message cap; it does not imply
that the full spike train was streamed. Slow sockets time out and close rather than retaining
an unbounded subscriber queue. Disconnecting a browser does not stop its run; reopening the
run URL replays the retained bounded history. This is not an archive/replay of every bin.

Quotas are application-level checks. Dataset loading, native engine operations and artifact
finalization are cooperative rather than forcibly preempted. Buffered writers can overshoot
the observed disk threshold before the next check; use an OS/container quota if a strict
filesystem ceiling is required. Resource limits are engineering choices, not scientific
parameters. The CLI is the path for larger workloads.

The Host header must name loopback, and browser Origin must exactly match the HTTP host and
port. Cross-origin requests and DNS-rebinding hosts are rejected; state changes additionally
require JSON. The page has a restrictive same-origin CSP, frame protection and nosniff.
There is no public authentication system: the local OS account and private tunnel are the
access boundary. Do not expose this ASGI application through a public proxy.

## Verify

```bash
uv sync --locked --group dev --group api-test --extra api --extra gpu
uv run ruff check .
uv run ruff format --check .
uv run mypy
FLYLAB_REQUIRE_API=1 FLYLAB_REQUIRE_TORCH=1 uv run pytest
node --check src/flylab/api/static/app.js
```

The `gpu` extra here enables CPU torch tests on macOS; it is unnecessary for NumPy-only
dashboard tests and is not installed by the Linux dashboard CI job. Without optional
dashboard dependencies its tests skip; `FLYLAB_REQUIRE_API=1` turns their absence into failure.
The separate dashboard workflow installs the extra/test group explicitly.

See [M7 acceptance](acceptance/M7.md) for screenshots, executed checks and unverified work.
