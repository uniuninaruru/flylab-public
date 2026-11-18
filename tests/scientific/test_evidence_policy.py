"""Guards for the spec's truth boundaries and module boundaries (spec sections 1 and 5)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from flylab.data.sources import RELEASES
from flylab.provenance.labels import Evidence

SRC = Path(__file__).resolve().parents[2] / "src" / "flylab"

# package -> import prefixes it must never use
FORBIDDEN_IMPORTS: dict[str, tuple[str, ...]] = {
    "data": (
        "torch", "mujoco", "flybody", "flygym", "dm_control", "fastapi", "starlette",
        "paramiko", "flylab.neural", "flylab.embodiment", "flylab.api",
    ),
    "provenance": ("torch", "mujoco", "flylab.neural", "flylab.embodiment", "flylab.api"),
    "neural": ("mujoco", "flybody", "flygym", "fastapi", "starlette", "paramiko",
               "flylab.api", "flylab.embodiment"),
    "experiments": ("fastapi", "starlette", "paramiko", "flylab.api"),
    "embodiment": ("flylab.data", "flylab.api", "paramiko"),
    "api": ("paramiko", "subprocess", "pty", "pexpect"),
}  # fmt: skip

SPEC_DOCS = [
    "IMPLEMENTATION_SPEC.md",
    "SOURCES_AND_LICENSES.md",
    "ENVIRONMENT_AUDIT.md",
    "DATASET_COUNT_RECONCILIATION.md",
    "MODEL_ASSUMPTIONS.md",
    "SCIENTIFIC_LIMITATIONS.md",
    "OPEN_QUESTIONS.md",
    "BENCHMARK_PROTOCOL.md",
    "SECURITY.md",
]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package", sorted(FORBIDDEN_IMPORTS))
def test_module_boundaries(package: str) -> None:
    forbidden = FORBIDDEN_IMPORTS[package]
    for py in (SRC / package).rglob("*.py"):
        for name in _imports(py):
            bad = [f for f in forbidden if name == f or name.startswith(f + ".")]
            assert not bad, f"{py.relative_to(SRC)} imports {name} (forbidden in {package}/)"


def test_registered_releases_are_empirical_and_dated() -> None:
    for rel in RELEASES.values():
        assert rel.evidence is Evidence.EMPIRICAL
        assert rel.verified_on
        assert rel.verified_by
        assert rel.license_spdx


@pytest.mark.parametrize("doc", SPEC_DOCS)
def test_required_docs_exist(repo_root: Path, doc: str) -> None:
    p = repo_root / "docs" / doc
    assert p.is_file(), f"missing docs/{doc}"
    assert p.stat().st_size > 200


def test_fixture_marked_synthetic(tiny_dir: Path) -> None:
    for name in ("neurons.csv", "edges.csv", "README.md"):
        head = (tiny_dir / name).read_text(encoding="utf-8")[:300]
        assert "SYNTHETIC" in head.upper()
