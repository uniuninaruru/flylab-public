"""Regenerate the M4 figures from the committed evidence JSON (PR-04).

    uv sync --extra plots   # matplotlib==3.11.2, optional
    uv run --extra plots python scripts/m4/plots.py --evidence docs/acceptance/evidence \\
        --out docs/acceptance/figures

Reads only files under docs/acceptance/evidence/, never run directories, so anyone can redraw
the figures without re-running the simulations. Every figure shows every seed as a dot next to
the mean (no single-seed figure is presented as a result). A figure whose input file is missing
is skipped with a message.

Colours: the reference categorical palette's first three slots (validated all-pairs for colour
vision deficiency), grey for individual seeds; text in ink colours, never in series colours.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SEED = "#a9a8a2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CAPTION = "Model output (label ASSUMED): not a measurement and not behaviour."

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "lines.linewidth": 2,
    }
)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"skip: {path} not found")
        return None
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _save(fig: Any, out: Path, name: str) -> None:
    fig.text(0.01, -0.04, CAPTION, fontsize=7, color=INK2, ha="left", va="top")
    path = out / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _ci(entry: dict[str, Any]) -> tuple[float, float] | None:
    ci = entry.get("ci_mean") or {}
    lo, hi = ci.get("low"), ci.get("high")
    return None if lo is None or hi is None else (float(lo), float(hi))


def calibration_figures(ev: Path, out: Path) -> None:
    data = _load(ev / "m4-calibration-wsyn.json")
    if data is None:
        return
    cal = data["by_readout"]["mn9_left"]
    other = data["by_readout"].get("mn9_right")
    per = cal["per_w_syn"]
    n = len(per)
    cols = 4
    rows = math.ceil(n / cols)

    # 1. MN9 rate vs input rate, one panel per w_syn
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), sharex=True, sharey=True)
    for ax, e, e2 in zip(axes.flat, per, other["per_w_syn"] if other else per, strict=False):
        r = e["rates_hz"]
        for vals in e["readout_by_seed_hz"].values():
            ax.plot(r, vals, "o", color=SEED, ms=3, zorder=1)
        ax.plot(r, e["readout_mean_hz"], "-o", color=BLUE, ms=4, label="MN9 L (mean)", zorder=3)
        if other:
            ax.plot(r, e2["readout_mean_hz"], "-s", color=ORANGE, ms=4, label="MN9 R (mean)",
                    zorder=2)  # fmt: skip
        f = e["D1"]["fraction"]
        ax.set_title(f"w_syn {e['w_syn_mV']:.3f} mV   F = {'n/a' if f is None else f'{f:.2f}'}",
                     fontsize=9)  # fmt: skip
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes.flat:
        ax.set_xticks([0, 50, 100, 150, 200])
    fig.supxlabel("requested LB3c-left Poisson rate (Hz); F = n/a: MN9 silent at every rate")
    fig.supylabel("MN9 rate (Hz)")
    axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("MN9 response vs input rate per w_syn (grey = individual seeds)", x=0.01,
                 ha="left")  # fmt: skip
    _save(fig, out, "m4_calibration_mn9_vs_rate.png")

    # 2. criterion fraction vs w_syn
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for name, color, dx, label in (("D1", BLUE, -0.002, "D1: / max over rates (primary)"),
                                   ("D3", ORANGE, 0.002, "D3: at achieved ~100 Hz")):  # fmt: skip
        xs, ys, lo, hi = [], [], [], []
        for e in per:
            f = e[name]["fraction"]
            ci = e[name]["ci"]
            if f is None:
                continue
            xs.append(e["w_syn_mV"] + dx)
            ys.append(f)
            lo.append(f - (ci["low"] if ci["low"] is not None else f))
            hi.append((ci["high"] if ci["high"] is not None else f) - f)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=color, ms=6, capsize=3, label=label)
    ax.axhline(0.8, color=INK2, ls="--", lw=1)
    ax.text(0.02, 0.8, "target 0.8 (Shiu et al.)", color=INK2, fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())  # fmt: skip
    undefined = [e["w_syn_mV"] for e in per if e["D1"]["fraction"] is None]
    if undefined:
        ax.text(0.02, 0.97, f"not shown (MN9 silent at every rate): w_syn <= {max(undefined):.3f}",
                transform=ax.transAxes, fontsize=7, color=INK2, va="top")  # fmt: skip
    sel = cal["selected"]
    if sel.get("w_syn_mV") is not None:
        ax.axvline(sel["w_syn_mV"], color=BLUE, lw=1, alpha=0.4)
    ax.set_xlabel("w_syn (mV per synapse)")
    ax.set_ylabel("F = MN9(100 Hz) / maximal MN9")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Calibration criterion (95 % percentile interval over joint seed resamples)",
                 fontsize=9)  # fmt: skip
    _save(fig, out, "m4_calibration_fraction.png")

    # 3. whole-brain rate per seed (regimes)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), sharex=True, sharey=True)
    for ax, e in zip(axes.flat, per, strict=False):
        r = e["rates_hz"]
        for vals in e["whole_brain_by_seed_hz"].values():
            ax.plot(r, [max(v, 1e-4) for v in vals], "o", color=BLUE, ms=3.5, alpha=0.75)
        ax.set_yscale("log")
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.set_title(f"w_syn {e['w_syn_mV']:.3f} mV", fontsize=9)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.supxlabel("requested LB3c-left Poisson rate (Hz)")
    fig.supylabel("whole-brain mean rate (Hz)")
    fig.suptitle("Whole-brain mean rate, one dot per seed (log scale)", x=0.01, ha="left")
    _save(fig, out, "m4_calibration_whole_brain.png")


def _dot_rows(
    ax: Axes, stats: dict[str, Any], conditions: Sequence[str], population: str, color: str
) -> None:
    for y, c in enumerate(conditions):
        entry = stats["conditions"][c]["populations"][population]["mean_rate_hz"]
        vals = [v for v in entry["values_by_seed"].values() if v is not None]
        ax.plot(vals, [y] * len(vals), "o", color=SEED, ms=3, zorder=1)
        ci = _ci(entry)
        if ci is not None:
            ax.plot(ci, [y, y], color=color, lw=2, zorder=2)
        ax.plot([entry["mean"]], [y], "D", color=color, ms=6, zorder=3)
    ax.set_yticks(range(len(conditions)))
    ax.set_ylim(len(conditions) - 0.5, -0.5)


def controls_figure(ev: Path, out: Path, group: str) -> None:
    stats = _load(ev / f"m4-{group}.json")
    if stats is None:
        return
    conds = list(stats["conditions"])
    panels = [("mn9_left", "MN9 L", BLUE), ("mn9_right", "MN9 R", ORANGE),
              ("__all__", "whole brain", AQUA)]  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.42 * len(conds) + 1.4), sharey=True)
    for ax, (pop, title, color) in zip(axes, panels, strict=True):
        _dot_rows(ax, stats, conds, pop, color)
        ax.set_title(f"{title}: mean rate (Hz)", fontsize=9)
        ax.set_xlabel("Hz")
    axes[0].set_yticklabels(conds)
    w = stats["provenance"]["group_manifest"]["group_id"]
    fig.suptitle(f"{w}: diamond = mean, bar = 95 % BCa interval, grey = seeds "
                 f"(n = {len(stats['seeds'])})", x=0.01, ha="left")  # fmt: skip
    _save(fig, out, f"m4_{group}_rates.png")

    if any("psth_mean_rate_hz" in v for v in stats["conditions"].values()):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        picks = [("intact", BLUE), ("silence_bridges", ORANGE), ("null_graph", AQUA)]
        for ax, pop, label in ((axes[0], "mn9_left", "MN9 L"), (axes[1], "__all__", "whole brain")):
            for c, color in picks:
                if c not in stats["conditions"]:
                    continue
                ps = stats["conditions"][c]["psth_mean_rate_hz"][pop]
                b = stats["conditions"][c]["psth_bin_ms"]
                k = max(1, round(50 / b))  # display bins of 50 ms (mean of k stored bins)
                coarse = [sum(ps[i : i + k]) / len(ps[i : i + k]) for i in range(0, len(ps), k)]
                t = [i * b * k for i in range(len(coarse))]
                ax.step(t, coarse, where="post", color=color, lw=1.6, label=c)
            ax.set_xlabel("time (ms); stimulus on for 0-1000 ms")
            ax.set_ylabel(f"{label} rate (Hz), mean over seeds")
            ax.set_title(f"{label}, 50 ms bins", fontsize=9)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.suptitle(f"{w}: PSTH, mean over {len(stats['seeds'])} seeds", x=0.01, y=1.03, ha="left")
        _save(fig, out, f"m4_{group}_psth.png")


def _rate_panel(
    ax: Axes,
    conditions: dict[str, Any],
    names: Sequence[str],
    rates: Sequence[float],
    series: Sequence[tuple[str, str, str]],
) -> None:
    for pop, color, label in series:
        means = []
        for r, c in zip(rates, names, strict=True):
            e = conditions[c]["populations"][pop]["mean_rate_hz"]
            vals = [v for v in e["values_by_seed"].values() if v is not None]
            ax.plot([r] * len(vals), vals, "o", color=SEED, ms=3, zorder=1)
            means.append(e["mean"])
        ax.plot(rates, means, "-o", color=color, label=label, zorder=3)


def _lr_panel(ax: Axes, mc: dict[str, Any], title: str) -> None:
    """MN9 contralateral / ipsilateral to the stimulated LB3c (MaleCNS sides), both inputs."""
    cases = (("intact", "mn9_right", "mn9_left"), ("stim_right", "mn9_left", "mn9_right"))
    for j, (c, contra, ipsi) in enumerate(cases):
        for k, (pop, color, name) in enumerate(((contra, BLUE, "contralateral"),
                                                (ipsi, ORANGE, "ipsilateral"))):  # fmt: skip
            e = mc["conditions"][c]["populations"][pop]["mean_rate_hz"]
            vals = [v for v in e["values_by_seed"].values() if v is not None]
            x = j + (k - 0.5) * 0.32
            ax.bar([x], [e["mean"]], width=0.28, color=color, zorder=2,
                   label=f"{name} MN9" if j == 0 else None)  # fmt: skip
            ax.plot([x] * len(vals), vals, "o", color=SEED, ms=3, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["left LB3c driven", "right LB3c driven"])
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def q19_figure(ev: Path, out: Path, controls: Sequence[str], w_cal: float) -> None:
    fw = _load(ev / "m4-m4_q19_flywire630.json")
    cal = _load(ev / "m4-calibration-wsyn.json")
    if fw is None or cal is None:
        return
    groups = [(g, _load(ev / f"m4-{g}.json")) for g in controls]
    fig, axes = plt.subplots(1, 2 + len(groups), figsize=(4.3 * (2 + len(groups)), 3.9),
                             sharey=True)  # fmt: skip
    ax = axes[0]
    names = [c for c in fw["conditions"] if c.startswith("shiu21_r")]
    _rate_panel(ax, fw["conditions"], names, [float(c.split("_r")[1]) for c in names],
                (("mn9_contra", BLUE, "contralateral MN9"), ("mn9_ipsi", ORANGE,
                                                             "ipsilateral MN9")))  # fmt: skip
    ax.set_xlabel("requested rate, 21 left sugar GRNs (Hz)")
    ax.set_ylabel("MN9 rate (Hz)")
    ax.set_title("(a) FlyWire v630 (Shiu et al.'s graph), w_syn 0.275 mV, n = 10", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    for e in cal["by_readout"]["mn9_left"]["per_w_syn"]:
        if abs(e["w_syn_mV"] - w_cal) > 1e-9:
            continue
        r = e["rates_hz"]
        left = e["readout_by_seed_hz"]
        right = next(x for x in cal["by_readout"]["mn9_right"]["per_w_syn"]
                     if abs(x["w_syn_mV"] - w_cal) < 1e-9)["readout_by_seed_hz"]  # fmt: skip
        for vals in list(left.values()) + list(right.values()):
            ax.plot(r, vals, "o", color=SEED, ms=3, zorder=1)
        ax.plot(r, [sum(v[i] for v in right.values()) / len(right) for i in range(len(r))],
                "-o", color=BLUE, label="contralateral MN9 (R; under-traced in v1.0, Q19)",
                zorder=3)  # fmt: skip
        ax.plot(r, [sum(v[i] for v in left.values()) / len(left) for i in range(len(r))],
                "-o", color=ORANGE, label="ipsilateral MN9 (L)", zorder=3)  # fmt: skip
    ax.set_xlabel("requested rate, 12 left LB3c (Hz)")
    ax.set_title(f"(b) MaleCNS v1.0, w_syn {w_cal:.3f} mV, n = 5", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for k, (ax, (g, mc)) in enumerate(zip(axes[2:], groups, strict=True)):
        if mc is None or "stim_right" not in mc["conditions"]:
            ax.set_visible(False)
            continue
        _lr_panel(ax, mc, f"({'cdef'[k]}) MaleCNS {g}, 100 Hz, n = {len(mc['seeds'])}")
    fig.suptitle("Q19: MN9 laterality under the same FlyLab model on two connectomes "
                 "(grey = seeds)", x=0.01, y=1.04, ha="left")  # fmt: skip
    _save(fig, out, "m4_q19_laterality.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", type=Path, default=Path("docs/acceptance/evidence"))
    ap.add_argument("--out", type=Path, default=Path("docs/acceptance/figures"))
    ap.add_argument("--controls", nargs="*", default=[], help="controls group ids to plot")
    ap.add_argument("--w-cal", type=float, default=0.27, help="calibrated w_syn shown in Q19")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    calibration_figures(a.evidence, a.out)
    for g in a.controls:
        controls_figure(a.evidence, a.out, g)
    q19_figure(a.evidence, a.out, a.controls, a.w_cal)


if __name__ == "__main__":
    main()
