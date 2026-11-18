"""Run with python -m flylab.api; the bind address cannot be overridden."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyLab localhost dashboard")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--demo-pace-ms",
        type=int,
        choices=(0, 20, 50, 100),
        default=0,
        help="optional wall-time delay per bin for visual previews; not model time",
    )
    args = parser.parse_args()
    if not (args.repo / "configs").is_dir() or not 1 <= args.port <= 65535:
        parser.error("a repository configs directory and a valid port are required")
    try:
        import uvicorn

        from flylab.api.server import BIND_ADDRESS, MAX_PAYLOAD_BYTES, create_app
    except ImportError:
        parser.exit(2, "Dashboard dependencies missing: uv sync --extra api\n")
    uvicorn.run(
        create_app(args.repo, pace_ms=args.demo_pace_ms),
        host=BIND_ADDRESS,
        port=args.port,
        workers=1,
        proxy_headers=False,
        server_header=False,
        access_log=False,
        ws_max_size=MAX_PAYLOAD_BYTES,
        ws_max_queue=4,
        limit_concurrency=16,
        timeout_keep_alive=5,
        h11_max_incomplete_event_size=8192,
    )


if __name__ == "__main__":
    main()
