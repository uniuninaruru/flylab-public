"""Security, bounded streaming and real NumPy runs for the optional dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

if os.environ.get("FLYLAB_REQUIRE_API") == "1":
    import httpx  # noqa: F401
    import starlette  # noqa: F401
else:
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")

from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from flylab.api import server
from flylab.api.__main__ import main
from flylab.api.manager import Limits, RequestError, RunManager
from flylab.api.server import MAX_STREAM_BYTES, create_app, send_event
from flylab.api.stream import (
    HISTORY_MESSAGES,
    SPIKES_PER_MESSAGE,
    EventBuffer,
    LiveRecorder,
    RunQuotaExceeded,
)
from flylab.experiments.config import load_config
from flylab.experiments.runner import run_experiment
from flylab.neural.engine import NeuralStep
from flylab.provenance.hashing import verify_checksums

ROOT = Path(__file__).resolve().parents[2]
BASE = {"config": "tiny_fixture.yaml", "overrides": {"backend": "numpy"}}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app(ROOT, out_root=tmp_path / "runs", pace_ms=20)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c


def finish(c: TestClient, rid: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result: dict[str, Any] = c.get("/api/v1/experiments/" + rid).json()
        if result["state"] in {"succeeded", "failed", "cancelled"}:
            return result
        time.sleep(0.01)
    pytest.fail("run did not finish within the test deadline")


def test_static_health_allowlist_and_security_headers(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "not validated behaviour" in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/assets/secret.txt").status_code == 404
    names = [c["name"] for c in client.get("/api/v1/configs").json()["configs"]]
    assert "tiny_fixture.yaml" in names
    assert any(n.startswith("bench/") for n in names)
    assert client.get("/api/v1/health").json()["concurrency"] == 1
    assert client.get("/api/v1/datasets").json()["datasets"] == []
    assert client.post("/api/v1/shell", json={"command": "echo unsafe"}).status_code == 404


@pytest.mark.parametrize(
    "name",
    [
        "../tiny_fixture.yaml",
        "/tmp/config.yaml",
        "configs/tiny_fixture.yaml",
        "bench/../../secret.yaml",
        "%2e%2e/secret.yaml",
        "tiny_fixture.yaml/anything",
    ],
)
def test_config_paths_are_exact_allowlist_keys(client: TestClient, name: str) -> None:
    assert client.post("/api/v1/experiments", json={"config": name}).status_code == 400


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": "1"},
        {"seed": 2**32},
        {"steps": 0},
        {"steps": 100_001},
        {"steps": 1.5},
        {"steps": 1001},
        {"backend": "shell"},
        {"dataset": {"path": "/tmp/data"}},
        {"model": {}},
        {"populations": {}},
        {"unknown": 1},
    ],
)
def test_invalid_overrides(client: TestClient, overrides: dict[str, Any]) -> None:
    response = client.post(
        "/api/v1/experiments", json={"config": "tiny_fixture.yaml", "overrides": overrides}
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        "tiny_fixture.yaml",
        {"config": []},
        {"config": "tiny_fixture.yaml", "out_root": "/tmp"},
    ],
)
def test_invalid_top_level_payload(client: TestClient, payload: Any) -> None:
    assert (
        client.post(
            "/api/v1/experiments",
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        ).status_code
        == 400
    )


def test_payload_limit_checks_actual_bytes_and_json(client: TestClient) -> None:
    assert (
        client.post(
            "/api/v1/experiments", content="x" * 4097, headers={"Content-Type": "application/json"}
        ).status_code
        == 413
    )
    # A false small Content-Length cannot bypass the streaming body counter.
    assert (
        client.post(
            "/api/v1/experiments",
            content="x" * 4097,
            headers={"Content-Type": "application/json", "Content-Length": "2"},
        ).status_code
        == 413
    )
    assert (
        client.post(
            "/api/v1/experiments", content="{", headers={"Content-Type": "application/json"}
        ).status_code
        == 400
    )
    assert client.post("/api/v1/experiments", data=BASE).status_code == 400


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "attacker.example"},
        {"Origin": "https://attacker.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
    ],
)
def test_rebinding_and_cross_origin_are_rejected(
    client: TestClient,
    headers: dict[str, str],
) -> None:
    assert client.post("/api/v1/experiments", json=BASE, headers=headers).status_code == 403


def test_allowlist_excludes_symlinks_and_freezes_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    configs = root / "configs"
    configs.mkdir(parents=True)
    fixture = ROOT / "configs/tiny_fixture.yaml"
    (configs / "tiny.yaml").write_bytes(fixture.read_bytes())
    (configs / "escape.yaml").symlink_to(fixture)
    (configs / "bench").symlink_to(ROOT / "configs/bench", target_is_directory=True)
    manager = RunManager(root, out_root=tmp_path / "runs")
    assert [c["name"] for c in manager.configurations()] == ["tiny.yaml"]
    (configs / "tiny.yaml").write_text("not an experiment")
    _, cfg = manager.validate({"config": "tiny.yaml", "overrides": {"seed": 9}})
    assert cfg.seed == 9
    assert cfg.steps == 1000
    with pytest.raises(RequestError, match="allowed"):
        manager.validate({"config": "escape.yaml"})
    manager.close()


def test_real_numpy_run_stream_and_artifacts_match_offline_reference(
    client: TestClient,
    tmp_path: Path,
) -> None:
    result = client.post("/api/v1/experiments", json=BASE)
    assert result.status_code == 202
    rid = result.json()["id"]
    frames = []
    close_code = None
    with client.websocket_connect("ws://127.0.0.1/api/v1/experiments/" + rid + "/stream") as ws:
        while True:
            try:
                message = ws.receive_json()
            except WebSocketDisconnect as exc:
                close_code = exc.code
                break
            assert len(json.dumps(message).encode()) < MAX_STREAM_BYTES
            frames.append(message)
    assert close_code == 1000
    detail = finish(client, rid)
    assert detail["state"] == "succeeded", detail
    summary = detail["manifest_summary"]
    assert summary["dataset"]["dataset_id"] == "synthetic:tiny:v1"
    assert summary["evidence"]["model_output"] == "ASSUMED"
    assert summary["limitations"]
    assert any(m["type"] == "manifest" and m["summary"] for m in frames)
    rates = [m for m in frames if m["type"] == "population_rates"]
    assert rates
    assert any(m["spikes"] for m in rates)
    assert all(len(m["spikes"]) <= SPIKES_PER_MESSAGE for m in rates)
    assert rates[-1]["simulation_time_ms"] == 100.0
    assert any(m["active_interventions"] for m in rates)
    assert frames[-1]["type"] == "status"
    assert client.get("/api/v1/datasets").json()["datasets"][0]["verified_at_load"]
    artifacts = client.get("/api/v1/experiments/" + rid + "/artifacts").json()["artifacts"]
    assert "manifest.json" in [a["name"] for a in artifacts]
    assert all("/" not in a["name"] for a in artifacts)
    manager = cast(Starlette, client.app).state.manager
    assert verify_checksums(manager.out / rid / "checksums.sha256", root=manager.out / rid) == []
    reference = run_experiment(
        ROOT / "configs/tiny_fixture.yaml", tmp_path / "reference", "reference", progress=None
    )[0]
    reference_totals = cast(dict[str, object], reference.metrics["totals"])
    assert detail["metrics"]["totals"]["n_spikes"] == reference_totals["n_spikes"]
    assert detail["metrics"]["populations"] == reference.metrics["populations"]
    # A completed run's elapsed time must stop increasing when the browser reconnects.
    again = client.get("/api/v1/experiments/" + rid).json()
    assert again["wall_seconds"] == detail["wall_seconds"]


def test_concurrency_stop_and_cancelled_artifacts(tmp_path: Path) -> None:
    app = create_app(ROOT, out_root=tmp_path / "runs", pace_ms=100)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        rid = c.post("/api/v1/experiments", json=BASE).json()["id"]
        assert c.post("/api/v1/experiments", json=BASE).status_code == 409
        assert c.post("/api/v1/experiments/" + rid + "/cancel", json={"seed": 3}).status_code == 400
        assert c.post("/api/v1/experiments/" + rid + "/cancel", json={}).status_code == 200
        assert finish(c, rid)["state"] == "cancelled"
        assert c.get("/api/v1/experiments/" + rid + "/artifacts").json()["artifacts"] == []
        assert not (tmp_path / "runs" / rid).exists()
        assert c.get("/api/v1/experiments/absent").status_code == 404
        assert c.post("/api/v1/experiments/absent/cancel", json={}).status_code == 404


def test_session_and_event_quotas(tmp_path: Path) -> None:
    app = create_app(ROOT, out_root=tmp_path / "runs", limits=Limits(max_events=0, max_runs=1))
    with TestClient(app, base_url="http://127.0.0.1") as c:
        rid = c.post("/api/v1/experiments", json=BASE).json()["id"]
        result = finish(c, rid)
        assert result["state"] == "failed"
        assert "quota" in result["error"]
        assert c.post("/api/v1/experiments", json=BASE).status_code == 400


def test_event_ring_drops_old_messages_for_lagging_cursor() -> None:
    ring = EventBuffer(3)
    for n in range(20):
        ring.publish({"type": "test", "n": n})
    events, missed = ring.after(0)
    assert len(ring) == 3
    assert ring.dropped == 17
    assert missed == 17
    assert [e["n"] for e in events] == [17, 18, 19]
    assert ring.after(events[-1]["sequence"]) == ([], 0)
    assert HISTORY_MESSAGES == 64


def test_live_recorder_caps_spikes_and_keeps_exact_population_counts() -> None:
    cfg = load_config(ROOT / "configs/tiny_fixture.yaml")
    pops = {n: np.arange(64, dtype=np.int32) for n in cfg.recorded_populations()}
    frames: list[dict[str, Any]] = []
    recorder = LiveRecorder(cfg, pops, frames.append, threading.Event(), time.monotonic() + 5)
    for n in range(100):
        recorder.record(
            NeuralStep(
                step=n,
                t_ms=n * 0.1,
                dt_ms=0.1,
                spike_idx=np.arange(64, dtype=np.int32),
                forced=np.zeros(64, dtype=bool),
                propagated=np.ones(64, dtype=bool),
                n_forced_dropped=0,
                population_spike_counts=dict.fromkeys(pops, 64),
            )
        )
    assert len(frames) == 1
    assert len(frames[0]["spikes"]) == SPIKES_PER_MESSAGE
    assert frames[0]["spikes_omitted"] > 0
    assert frames[0]["populations"][0]["rate_hz"] == 10_000
    assert len(json.dumps(frames[0]).encode()) < MAX_STREAM_BYTES
    recorder.deadline = 0
    with pytest.raises(RunQuotaExceeded, match="wall-time"):
        recorder.record(
            NeuralStep(
                step=100,
                t_ms=10,
                dt_ms=0.1,
                spike_idx=np.array([], dtype=np.int32),
                forced=np.array([], dtype=bool),
                propagated=np.array([], dtype=bool),
                n_forced_dropped=0,
                population_spike_counts={},
            )
        )


def test_websocket_client_limit_and_origin(tmp_path: Path) -> None:
    app = create_app(ROOT, out_root=tmp_path / "runs", pace_ms=100, limits=Limits(max_clients=1))
    with TestClient(app, base_url="http://127.0.0.1") as c:
        rid = c.post("/api/v1/experiments", json=BASE).json()["id"]
        path = "ws://127.0.0.1/api/v1/experiments/" + rid + "/stream"
        with (
            pytest.raises(WebSocketDisconnect),
            c.websocket_connect(
                path,
                headers={"Origin": "https://attacker.example"},
            ),
        ):
            pass
        with c.websocket_connect(path) as first:
            assert first.receive_json()["type"] == "manifest"
            with pytest.raises(WebSocketDisconnect), c.websocket_connect(path):
                pass
        c.post("/api/v1/experiments/" + rid + "/cancel", json={})


def test_slow_socket_has_a_send_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class SlowSocket:
        async def send_text(self, text: str) -> None:
            await asyncio.sleep(10)

    monkeypatch.setattr(server, "SEND_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(send_event(cast(WebSocket, SlowSocket()), {"type": "test"}))


def test_launch_is_fixed_to_loopback_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    calls: list[dict[str, Any]] = []

    def launch(app: Any, **kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(uvicorn, "run", launch)
    monkeypatch.setattr(sys, "argv", ["dashboard", "--repo", str(ROOT)])
    main()
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["workers"] == 1
    assert calls[0]["ws_max_queue"] == 4
    monkeypatch.setattr(sys, "argv", ["dashboard", "--host", "0.0.0.0"])  # noqa: S104
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
