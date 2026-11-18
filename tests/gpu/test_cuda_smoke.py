"""Runs only on the local RTX host: `uv run pytest tests/gpu --run-gpu`.

Cloud CI never runs these. A skip here is not a pass: GPU status stays UNVERIFIED until
the owner's real log exists (see scripts/run_local_gpu_smoke.sh).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu


def test_cuda_device_is_sm120_and_computes() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.get_device_capability(0) == (12, 0)
    assert "sm_120" in torch.cuda.get_arch_list(), "torch build has no sm_120 kernels"
    x = torch.randn((1024, 1024), device="cuda")
    y = x @ x
    assert bool(torch.isfinite(y).all().item())
    torch.cuda.synchronize()
