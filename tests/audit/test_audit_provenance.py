"""Audit findings on provenance: input identity and commit recording."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flylab.data.malecns import ANNOTATIONS, RawDataError, prepare_male_cns
from flylab.provenance.environment import COMMIT_ENV_VAR, git_state
from flylab.provenance.hashing import sha256_file

ISSUE_INPUTS = "https://github.com/uniuninaruru/flylab/issues/5"
ISSUE_COMMIT = "https://github.com/uniuninaruru/flylab/issues/6"


# --- Raw inputs are not checked against the registered upstream objects ----------------------
#
# `_check_inputs` compares SHA-256 only with SOURCE.json, and only when SOURCE.json exists and
# has an entry for the file. `flylab.data.sources` holds the upstream size and MD5 of each file,
# but `prepare` never uses them. Files with the official names but other content (a manual
# download of another release, a truncated copy, or these synthetic tables) are imported and
# the output is labelled dataset "male-cns:v1.0" with EMPIRICAL columns.


def test_prepare_refuses_inputs_that_do_not_match_the_registry(
    male_cns_raw: Path, tmp_path: Path
) -> None:
    assert not (male_cns_raw / "SOURCE.json").exists()  # manual / offline import (spec 6.2)
    with pytest.raises(RawDataError):
        prepare_male_cns(male_cns_raw, tmp_path / "proc", progress=None)


def test_prepare_refuses_inputs_missing_from_source_json(
    male_cns_raw: Path, tmp_path: Path
) -> None:
    # SOURCE.json exists but records only one of the three inputs (e.g. an interrupted fetch);
    # the two unrecorded files are accepted without any hash comparison.
    record = {
        "dataset_id": "male-cns:v1.0",
        "files": {ANNOTATIONS: {"sha256": sha256_file(male_cns_raw / ANNOTATIONS)}},
    }
    (male_cns_raw / "SOURCE.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RawDataError):
        prepare_male_cns(male_cns_raw, tmp_path / "proc", progress=None)


# --- Commit of an unrelated parent repository is recorded --------------------------------------
#
# Code deployed with `git archive` (scripts/remote/sync_to_host.sh) has no .git. If the deploy
# directory lies inside another Git work tree (for example a version-controlled home
# directory), `git rev-parse HEAD` succeeds and returns that other repository's commit, so
# `git_state` reports it with source "git" and never reaches the FLYLAB_GIT_COMMIT fallback.
# scripts/run_local_gpu_smoke.sh line 30 has the same precedence and passes the wrong value
# into the image as FLYLAB_GIT_COMMIT.


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=audit", "-c", "user.email=audit@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip


@pytest.mark.skipif(shutil.which("git") is None, reason="needs the git binary")
def test_git_state_ignores_unrelated_parent_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "home"
    parent.mkdir()
    _git(parent, "init", "-q")
    _git(parent, "commit", "-q", "--allow-empty", "-m", "unrelated parent repository")
    parent_head = _git(parent, "rev-parse", "HEAD")
    deployed = parent / "flylab" / "src"  # git-archive copy: no .git of its own
    deployed.mkdir(parents=True)
    (deployed / "placeholder.txt").write_text("deployed file\n", encoding="utf-8")
    deployed_commit = "b" * 40
    monkeypatch.setenv(COMMIT_ENV_VAR, deployed_commit)

    state = git_state(deployed)
    assert state["commit"] != parent_head
    assert state["commit"] == deployed_commit
