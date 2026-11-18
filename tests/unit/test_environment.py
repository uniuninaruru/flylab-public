from __future__ import annotations

import getpass
import json
import socket
from pathlib import Path

import pytest

from flylab.provenance.environment import capture_environment, git_state


def test_environment_is_json_and_has_versions() -> None:
    env = capture_environment()
    text = json.dumps(env)
    assert env["schema_version"] == 1
    assert isinstance(env["packages"], dict)
    assert env["packages"]["numpy"] is not None
    assert "python" in text


def test_environment_does_not_identify_host() -> None:
    text = json.dumps(capture_environment())
    secrets = [socket.gethostname().split(".")[0], getpass.getuser(), str(Path.home())]
    for s in secrets:
        if len(s) >= 4:  # very short names would match unrelated text
            assert s not in text


def test_git_state_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLYLAB_GIT_COMMIT", raising=False)
    state = git_state(tmp_path)
    assert state["commit"] is None
    assert state["dirty"] is None


def test_git_state_falls_back_to_deployed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLYLAB_GIT_COMMIT", "a" * 40)
    state = git_state(tmp_path)
    assert state["commit"] == "a" * 40
    assert state["source"] == "env:FLYLAB_GIT_COMMIT"
    monkeypatch.setenv("FLYLAB_GIT_COMMIT", "unknown")
    assert git_state(tmp_path)["commit"] is None
