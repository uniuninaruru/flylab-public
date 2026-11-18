from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked `gpu` (only on the local CUDA host, never in cloud CI)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(reason="GPU test: run with --run-gpu on the local RTX host")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tiny_dir() -> Path:
    return FIXTURES / "tiny_graph"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
