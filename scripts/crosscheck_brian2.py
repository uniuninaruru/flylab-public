"""Cross-check the FlyLab NumPy LIF engine against Brian2 running the Shiu et al. equations.

Not part of CI and not a FlyLab dependency. Run in a throwaway environment:

    uv run --with brian2==2.10.1 python scripts/crosscheck_brian2.py --out artifacts/crosscheck

Brian2 gets the equations of the Shiu et al. (2024) reference code (model.py lines 44-52 at
commit 91bdd1e, plus a per-step external drive term `I`), `method='linear'`, the default
`dt = 0.1 ms`, refractoriness `rfc` and one synaptic delay. FlyLab gets the same synthetic graph,
weights and drive. The script reports spike-train agreement and the final membrane-potential
difference; it does not decide what counts as "agreement" for you. Synthetic graph only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from flylab.data.graph import UnknownSignPolicy
from flylab.data.synthetic import random_graph
from flylab.neural.engine import StepDrive
from flylab.neural.lif_numpy import NumpyLIFEngine
from flylab.neural.network import build_network
from flylab.neural.params import SHIU2024_CODE_V1, discretize
from flylab.provenance.environment import capture_environment


def run_brian2(
    n: int,
    src: np.ndarray[Any, Any],
    dst: np.ndarray[Any, Any],
    w_mv: np.ndarray[Any, Any],
    drive_mv: np.ndarray[Any, Any],
    steps: int,
) -> tuple[list[tuple[int, int]], np.ndarray[Any, Any], str]:
    import brian2 as b2

    b2.prefs.codegen.target = "numpy"  # no compiler needed; float64 throughout
    b2.start_scope()
    p = SHIU2024_CODE_V1.values()
    ms, mv = b2.ms, b2.mV
    dt = b2.defaultclock.dt
    drive = b2.TimedArray(drive_mv * mv, dt=dt)  # value for step k = drive_mv[k, i]
    eqs = """
    dv/dt = (v_0 - v + g + I) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    I = drive(t, i)                : volt (constant over dt)
    rfc                            : second
    """
    ns = {
        "v_0": p["v_rest"] * mv, "v_rst": p["v_reset"] * mv, "v_th": p["v_threshold"] * mv,
        "t_mbr": p["tau_m"] * ms, "tau": p["tau_syn"] * ms, "drive": drive,
    }  # fmt: skip
    neu = b2.NeuronGroup(
        n, eqs, method="linear", threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
        refractory="rfc", namespace=ns,
    )  # fmt: skip
    neu.v = p["v_rest"] * mv
    neu.g = 0 * mv
    neu.rfc = p["t_ref"] * ms
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p["delay"] * ms)
    syn.connect(i=src, j=dst)
    syn.w = w_mv * mv
    mon = b2.SpikeMonitor(neu)
    net = b2.Network(neu, syn, mon)
    net.run(steps * dt)
    step = np.rint(np.asarray(mon.t / dt)).astype(np.int64)
    spikes = sorted(zip(step.tolist(), np.asarray(mon.i).tolist(), strict=True))
    return spikes, np.asarray(neu.v / mv), str(b2.__version__)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("artifacts/crosscheck"))
    ap.add_argument("--neurons", type=int, default=300)
    ap.add_argument("--degree", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--w-syn", type=float, default=0.6, help="mV per synapse (synthetic graph)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    g, _ = random_graph(args.neurons, args.degree, seed=args.seed)
    net = build_network(g, UnknownSignPolicy.ERROR, args.w_syn)
    rng = np.random.default_rng(args.seed)
    base = rng.uniform(4.0, 14.0, net.n_neurons)
    drive = base + rng.normal(0.0, 2.0, (args.steps, net.n_neurons))
    drive[: args.steps // 10] = 0.0  # input starts after 10% of the run

    t0 = time.perf_counter()
    eng = NumpyLIFEngine(net, discretize(SHIU2024_CODE_V1, 0.1))
    eng.reset(args.seed)
    idx = np.arange(net.n_neurons, dtype=np.int32)
    fl: list[tuple[int, int]] = []
    for k in range(args.steps):
        out = eng.step(StepDrive(current_idx=idx, current_mV=drive[k]))
        fl += [(k, int(i)) for i in out.spike_idx]
    t_fl = time.perf_counter() - t0

    t0 = time.perf_counter()
    b2_spikes, b2_v, b2_version = run_brian2(
        net.n_neurons, net.src, net.dst, net.weight_mV, drive, args.steps
    )
    t_b2 = time.perf_counter() - t0

    set_fl, set_b2 = set(fl), set(b2_spikes)
    first_diff = next((a for a, b in zip(sorted(fl), b2_spikes, strict=False) if a != b), None)
    report = {
        "schema_version": 1,
        "what": "FlyLab NumPy LIF vs Brian2 running the Shiu et al. equations; synthetic graph",
        "graph": {"n_neurons": net.n_neurons, "n_edges": net.n_edges, "seed": args.seed,
                  "w_syn_mV": args.w_syn},
        "steps": args.steps, "dt_ms": 0.1,
        "brian2_version": b2_version,
        "n_spikes_flylab": len(fl), "n_spikes_brian2": len(b2_spikes),
        "n_common": len(set_fl & set_b2),
        "n_only_flylab": len(set_fl - set_b2), "n_only_brian2": len(set_b2 - set_fl),
        "first_difference_step_idx": first_diff,
        "identical_spike_trains": sorted(fl) == b2_spikes,
        "final_v_max_abs_diff_mV": float(np.max(np.abs(eng.state().v - b2_v))),
        "wall_time_s": {"flylab_numpy": round(t_fl, 3), "brian2_numpy_target": round(t_b2, 3)},
        "environment": capture_environment(),
    }  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "crosscheck_brian2.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    keys = ("n_spikes_flylab", "n_spikes_brian2", "n_common", "n_only_flylab", "n_only_brian2",
            "identical_spike_trains", "final_v_max_abs_diff_mV")  # fmt: skip
    for key in keys:
        print(f"{key}={report[key]}")
    print(f"report={path}")
    return 0 if report["identical_spike_trains"] else 1


if __name__ == "__main__":
    sys.exit(main())
