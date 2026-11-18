"""Capture the software environment for a run manifest, without identifying the host.

Deliberately excluded: hostname, user name, home directory, IP/MAC addresses, environment
variables. GPU details (driver, device, CUDA runtime) are captured by the host-side GPU
smoke script, not here, because importing torch can initialise CUDA.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from flylab import SCHEMA_VERSION, __version__

#: Distributions whose installed version is recorded if present. Never imported here.
TRACKED_DISTRIBUTIONS = ("numpy", "pyarrow", "torch", "mujoco", "flybody", "flygym")


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


#: Set by the GPU-host build (Docker build arg) when the code was deployed without `.git`.
COMMIT_ENV_VAR = "FLYLAB_GIT_COMMIT"


def _commit_from_env(reason: str) -> dict[str, object]:
    commit = os.environ.get(COMMIT_ENV_VAR, "").strip()
    if commit and commit != "unknown":
        # Deployed with `git archive` of this commit, so the tree matched it exactly.
        return {"commit": commit, "dirty": None, "source": f"env:{COMMIT_ENV_VAR}", "reason": None}
    return {"commit": None, "dirty": None, "source": None, "reason": reason}


def git_state(repo_dir: str | Path | None = None) -> dict[str, object]:
    """Commit hash and dirty flag of the repository.

    Falls back to $FLYLAB_GIT_COMMIT (set when code was deployed without .git); nulls if neither.
    """
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    git = shutil.which("git")
    if git is None:
        return _commit_from_env("git not found")

    def _git(*args: str) -> str:
        return subprocess.run(
            [git, *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        toplevel = Path(_git("rev-parse", "--show-toplevel"))
        if not _is_flylab_root(toplevel):
            # A git-archive deployment inside some other repository (e.g. a versioned home
            # directory): that repository's HEAD says nothing about this code.
            return _commit_from_env("enclosing git repository is not the FlyLab project")
        commit = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain")
    except (subprocess.CalledProcessError, OSError):
        return _commit_from_env("not a git checkout or no commits yet")
    return {"commit": commit, "dirty": bool(status), "source": "git", "reason": None}


def _is_flylab_root(path: Path) -> bool:
    """True if `path` holds the FlyLab project's own pyproject.toml."""
    pyproject = path / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return re.search(r'(?m)^name\s*=\s*"flylab"\s*$', text) is not None


def capture_environment(repo_dir: str | Path | None = None) -> dict[str, object]:
    """Return a JSON-serialisable, host-anonymous description of the environment."""
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "flylab_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_prefix_is_venv": sys.prefix != sys.base_prefix,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {name: _dist_version(name) for name in TRACKED_DISTRIBUTIONS},
        "git": git_state(repo_dir),
    }
