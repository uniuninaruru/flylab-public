"""Local-only Starlette HTTP/WebSocket boundary. No shell or arbitrary file endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect

from flylab.api.manager import BusyError, Limits, RequestError, RunManager
from flylab.experiments.config import ConfigError

BIND_ADDRESS = "127.0.0.1"
MAX_PAYLOAD_BYTES = 4096
MAX_STREAM_BYTES = 32_768
SEND_TIMEOUT_SECONDS = 1.0
STATIC = Path(__file__).with_name("static")


def _local_host(host: str) -> bool:
    try:
        value = urlsplit("http://" + host)
        return (
            value.hostname in {"localhost", BIND_ADDRESS}
            and (value.port is None or 1 <= value.port <= 65535)
            and value.username is None
            and value.password is None
            and not (value.path or value.query or value.fragment)
        )
    except ValueError:
        return False


class LocalBoundary:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        host = headers.get(b"host", b"").decode("latin1")
        origin = headers.get(b"origin")
        allowed = _local_host(host) and (
            origin is None or origin.decode("latin1") == "http://" + host
        )
        if not allowed:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await JSONResponse({"error": "local Host and same-origin access required"}, 403)(
                    scope, receive, send
                )
            return

        async def secure_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; script-src 'self'; style-src 'self'; "
                        b"connect-src 'self'; img-src 'self' data:; "
                        b"frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                    ),
                ]
            await send(message)

        await self.app(scope, receive, secure_send)


async def _payload(request: Request) -> Any:
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        raise RequestError("Content-Type must be application/json")
    size = request.headers.get("content-length")
    if size is not None:
        try:
            if not 0 <= int(size) <= MAX_PAYLOAD_BYTES:
                raise OverflowError
        except ValueError as exc:
            raise RequestError("invalid Content-Length") from exc
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_PAYLOAD_BYTES:
            raise OverflowError
        body.extend(chunk)
    try:
        return json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise RequestError("invalid JSON") from exc


async def send_event(websocket: WebSocket, event: dict[str, Any]) -> None:
    text = json.dumps(event, allow_nan=False, separators=(",", ":"))
    if len(text.encode()) > MAX_STREAM_BYTES:
        await websocket.close(code=1009)
        raise WebSocketDisconnect(1009)
    await asyncio.wait_for(websocket.send_text(text), SEND_TIMEOUT_SECONDS)


def create_app(
    repo_root: Path,
    *,
    out_root: Path | None = None,
    limits: Limits | None = None,
    pace_ms: int = 0,
) -> Starlette:
    manager = RunManager(repo_root, out_root=out_root, limits=limits, pace_ms=pace_ms)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        yield
        await asyncio.to_thread(manager.close)

    async def index(request: Request) -> Response:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    async def assets(request: Request) -> Response:
        name = request.path_params["name"]
        if name not in {"app.js", "style.css"}:
            return JSONResponse({"error": "unknown asset"}, 404)
        return FileResponse(STATIC / name)

    async def health(request: Request) -> Response:
        return JSONResponse(manager.health())

    async def configs(request: Request) -> Response:
        return JSONResponse({"configs": manager.configurations()})

    async def datasets(request: Request) -> Response:
        return JSONResponse({"datasets": manager.verified_datasets()})

    async def start(request: Request) -> Response:
        try:
            payload = await _payload(request)
            return JSONResponse(manager.start(payload), 202)
        except OverflowError:
            return JSONResponse({"error": "payload exceeds 4096 bytes"}, 413)
        except (RequestError, ConfigError) as exc:
            return JSONResponse({"error": str(exc)}, 400)
        except BusyError as exc:
            return JSONResponse({"error": str(exc)}, 409)

    async def detail(request: Request) -> Response:
        try:
            return JSONResponse(manager.snapshot(request.path_params["rid"]))
        except KeyError:
            return JSONResponse({"error": "unknown run"}, 404)

    async def cancel(request: Request) -> Response:
        try:
            # Stop also consumes a bounded JSON body; an HTML form cannot trigger a write.
            payload = await _payload(request)
            if payload != {}:
                raise RequestError("cancel expects an empty JSON object")
            return JSONResponse(manager.cancel(request.path_params["rid"]))
        except OverflowError:
            return JSONResponse({"error": "payload exceeds 4096 bytes"}, 413)
        except RequestError as exc:
            return JSONResponse({"error": str(exc)}, 400)
        except KeyError:
            return JSONResponse({"error": "unknown run"}, 404)

    async def artifacts(request: Request) -> Response:
        try:
            return JSONResponse({"artifacts": manager.artifacts(request.path_params["rid"])})
        except KeyError:
            return JSONResponse({"error": "unknown run"}, 404)

    async def stream(websocket: WebSocket) -> None:
        try:
            run = manager.get(websocket.path_params["rid"])
        except KeyError:
            await websocket.close(code=1008)
            return
        if not manager.claim_client():
            await websocket.close(code=1013)
            return
        try:
            await websocket.accept()
            await send_event(
                websocket,
                {
                    "schema_version": 1,
                    "type": "manifest",
                    "run_id": run.id,
                    "summary": manager.snapshot(run.id)["manifest_summary"],
                },
            )
            cursor = 0
            while True:
                events, missed = run.events.after(cursor)
                for event in events:
                    await send_event(websocket, {**event, "dropped_before": missed})
                    cursor, missed = event["sequence"], 0
                    if event["type"] == "status":
                        await websocket.close(code=1000)
                        return
                try:
                    received = await asyncio.wait_for(websocket.receive(), timeout=0.05)
                    if received["type"] == "websocket.disconnect":
                        return
                    # This is a server-to-client data channel, not a second command endpoint.
                    await websocket.close(code=1008)
                    return
                except TimeoutError:
                    pass
        except TimeoutError:
            await websocket.close(code=1013, reason="client too slow")
        except (WebSocketDisconnect, OSError):
            pass
        finally:
            manager.release_client()

    app = Starlette(
        debug=False,
        lifespan=lifespan,
        middleware=[Middleware(LocalBoundary)],
        routes=[
            Route("/", index),
            Route("/assets/{name}", assets),
            Route("/api/v1/health", health),
            Route("/api/v1/configs", configs),
            Route("/api/v1/datasets", datasets),
            Route("/api/v1/experiments", start, methods=["POST"]),
            Route("/api/v1/experiments/{rid}", detail),
            Route("/api/v1/experiments/{rid}/cancel", cancel, methods=["POST"]),
            Route("/api/v1/experiments/{rid}/artifacts", artifacts),
            WebSocketRoute("/api/v1/experiments/{rid}/stream", stream),
        ],
    )
    app.state.manager = manager
    return app
