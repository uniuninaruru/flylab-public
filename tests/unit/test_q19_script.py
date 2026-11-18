"""The path measures of scripts/analysis/q19_mn9_laterality.py on a hand-checkable toy graph."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "q19_mn9_laterality.py"


@pytest.fixture(scope="module")
def q19() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flylab_q19", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flylab_q19"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_direct_two_hop_and_propagation(q19: ModuleType) -> None:
    # 0 = GRN; 1 excitatory and 2 inhibitory interneurons; 3 = MN; 4 another input of 3
    #   0 -> 1 (4 syn), 0 -> 2 (2), 1 -> 3 (3), 2 -> 3 (5), 4 -> 3 (1), 0 -> 3 (6)
    src = np.array([0, 0, 0, 1, 2, 4])
    dst = np.array([1, 2, 3, 3, 3, 3])
    syn = np.array([4.0, 2.0, 6.0, 3.0, 5.0, 1.0])
    sign = np.array([1, 1, -1, 1, 1], dtype=np.int8)  # per neuron: 2 is inhibitory
    net = q19.Net(5, src, dst, syn, sign)
    g, m = np.array([0]), np.array([3])
    assert net.direct(g, m) == 6.0
    two = net.two_hop(g, m)
    assert two == {"via_excitatory": 12.0, "via_inhibitory": 10.0, "via_unknown_sign": 0.0}
    hop1, hop2 = net.propagate(g, 2)
    # input fraction of 3 from excitatory 0 at hop 1: 6 / (6 + 3 + 5 + 1)
    assert hop1[3] == pytest.approx(6 / 15)
    # hop 2 reaches 3 through 1 only (2 is inhibitory and contributes nothing): 1.0 * 3/15
    assert hop1[1] == pytest.approx(1.0)
    assert hop2[3] == pytest.approx(3 / 15)
    side = np.array(["L", "L", "R", "L", None], dtype=object)
    assert net.side_profile(g, side, outgoing=True) == {"L": round(10 / 12, 4), "R": 0.1667}
    assert net.side_profile(m, side, outgoing=False) == {
        "?": round(1 / 15, 4), "L": round(9 / 15, 4), "R": round(5 / 15, 4)
    }  # fmt: skip
    # MN 3 receives from 0 (6, exc), 1 (3, exc), 2 (5, inh), 4 (1, exc)
    assert net.inputs(m) == {
        "input_partners": 4, "input_synapses": 15, "from_excitatory": 10,
        "from_inhibitory": 5, "from_unknown_sign": 0,
    }  # fmt: skip
