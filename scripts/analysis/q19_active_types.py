"""Q19: which cell types fire, on which side, when left sugar GRNs are driven (MaleCNS vs FlyWire).

Reads `spikes.parquet` of one MaleCNS run and one FlyWire v630 cross-check run (both with
`recording.spike_events: true`), counts spikes per neuron, maps neurons to a common type name and
a side, and compares the two datasets type by type. Exploratory, single seed per dataset: it says
where the responding networks differ, not that a difference is robust.

Type names: MaleCNS `flywireType` (the matched FlyWire type; MaleCNS `type` when it is null),
FlyWire `cell_type` (flyconnectome/flywire_annotations). Sides: MaleCNS `somaSide`, else
`rootSide`; FlyWire `side` (true biological side, as in Shiu et al.). v630 root ids that are not
in the current annotations stay untyped.

    uv run python scripts/analysis/q19_active_types.py \\
        --malecns-run <run dir> --malecns-raw data/raw/male-cns-v1.0 \\
        --flywire-run <run dir> --flywire-annotations <flywire_annotations clone> \\
        --out docs/acceptance/evidence/q19-active-types.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.feather as feather
import pyarrow.parquet as pq

TOP = 40


def _counts(run: Path) -> tuple[dict[int, int], dict[str, Any]]:
    t = pq.read_table(run / "spikes.parquet", columns=["body_id"])
    ids, n = np.unique(t["body_id"].to_numpy(), return_counts=True)
    man = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    meta = {
        "run_id": man["run_id"],
        "config_sha256": man["config_sha256"],
        "seed": man["seed"],
        "dataset_id": man["dataset"]["dataset_id"],
        "w_syn_mV": man["model"]["parameters"]["w_syn"]["value"],
        "simulated_ms": man["time"]["simulated_ms"],
        "stimuli": man["stimuli"],
    }
    return {int(i): int(c) for i, c in zip(ids, n, strict=True)}, meta


def _malecns_labels(raw: Path) -> dict[int, tuple[str, str]]:
    ann = feather.read_table(
        raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "type", "flywireType", "somaSide", "rootSide"],
    ).to_pylist()
    out = {}
    for r in ann:
        name = r["flywireType"] or r["type"] or "<untyped>"
        side = r["somaSide"] or r["rootSide"] or "?"
        out[int(r["bodyId"])] = (str(name), str(side))
    return out


def _flywire_labels(fwa: Path) -> dict[int, tuple[str, str]]:
    t = pcsv.read_csv(
        fwa / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv",
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(
            include_columns=["root_id", "cell_type", "side"],
            column_types={"root_id": pa.int64(), "cell_type": pa.string(), "side": pa.string()},
        ),
    )
    sides = {"left": "L", "right": "R", "center": "M"}
    return {
        int(r): (str(c) if c else "<untyped>", sides.get(str(s), "?"))
        for r, c, s in zip(
            t["root_id"].to_pylist(),
            t["cell_type"].to_pylist(),
            t["side"].to_pylist(),
            strict=True,
        )
    }


def _by_type(
    counts: dict[int, int], labels: dict[int, tuple[str, str]], seconds: float
) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for body, n in counts.items():
        name, side = labels.get(body, ("<not in annotations>", "?"))
        agg[name][side] += n / seconds  # summed rate of the type's neurons on that side (Hz)
    return {k: dict(v) for k, v in agg.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--malecns-run", type=Path, required=True)
    ap.add_argument("--malecns-raw", type=Path, required=True)
    ap.add_argument("--flywire-run", type=Path, required=True)
    ap.add_argument("--flywire-annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mc, mc_meta = _counts(a.malecns_run)
    fw, fw_meta = _counts(a.flywire_run)
    m_lab, f_lab = _malecns_labels(a.malecns_raw), _flywire_labels(a.flywire_annotations)
    mt = _by_type(mc, m_lab, mc_meta["simulated_ms"] / 1000)
    ft = _by_type(fw, f_lab, fw_meta["simulated_ms"] / 1000)
    m_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    f_n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for name, side in m_lab.values():
        m_n[name][side] += 1
    for name, side in f_lab.values():
        f_n[name][side] += 1

    def total(d: dict[str, float]) -> float:
        return sum(d.values())

    def lr(d: dict[str, float]) -> float | None:
        left, right = d.get("L", 0.0), d.get("R", 0.0)
        return None if left + right == 0 else round((left - right) / (left + right), 3)

    def row(k: str) -> dict[str, object]:
        m, f = mt.get(k, {}), ft.get(k, {})
        return {
            "type": k,
            "malecns_summed_rate_hz_by_side": {s: round(v, 1) for s, v in sorted(m.items())},
            "flywire_summed_rate_hz_by_side": {s: round(v, 1) for s, v in sorted(f.items())},
            "malecns_n_neurons_by_side": dict(sorted(m_n.get(k, {}).items())),
            "flywire_n_neurons_by_side": dict(sorted(f_n.get(k, {}).items())),
            "malecns_laterality_L_minus_R": lr(m),
            "flywire_laterality_L_minus_R": lr(f),
        }

    skip = {"<untyped>", "<not in annotations>"}
    top_mc = sorted(mt, key=lambda k: -total(mt[k]))[:TOP]
    top_fw = sorted(ft, key=lambda k: -total(ft[k]))[:TOP]
    # the reference pathway: types active in Shiu et al.'s graph, minus the stimulated GRN types
    pathway = [k for k in top_fw if k not in skip and not k.startswith(("LB3", "LB4"))]
    out = {
        "schema_version": 1,
        "kind": "flylab_q19_active_types",
        "evidence": "ASSUMED (model output, one seed per dataset; exploratory)",
        "malecns_run": mc_meta,
        "flywire_run": fw_meta,
        "n_neurons_spiking": {"malecns": len(mc), "flywire": len(fw)},
        "flywire_pathway_types": [row(k) for k in pathway],
        "malecns_top_types": [row(k) for k in top_mc],
        "flywire_top_types": [row(k) for k in top_fw],
        "note": "summed rate = spikes of all neurons of the type on that side / simulated s; "
        "laterality = (L - R) / (L + R); stimulated GRNs are on side L in both runs. "
        "n_neurons_by_side counts annotation rows of the type (MaleCNS v1.0 annotations; FlyWire "
        "current annotations, i.e. v783 ids, while the run used v630). "
        "flywire_pathway_types = the most active FlyWire types (excluding LB3*/LB4* GRNs and "
        "untyped) with the same types' activity in MaleCNS",
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k in pathway:
        r = row(k)
        print(
            k,
            "FW",
            r["flywire_summed_rate_hz_by_side"],
            r["flywire_n_neurons_by_side"],
            "| MC",
            r["malecns_summed_rate_hz_by_side"],
            r["malecns_n_neurons_by_side"],
        )


if __name__ == "__main__":
    main()
