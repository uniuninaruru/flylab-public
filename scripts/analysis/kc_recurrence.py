"""Kenyon-cell recurrent connectivity in MaleCNS v1.0 and FlyWire v783 (PR-04, OPEN_QUESTIONS Q23).

Read-only counts for the high whole-brain regime seen at w_syn >= 0.25 mV on MaleCNS, which is
dominated by Kenyon cells (docs/acceptance/M4.md). Selectors differ between datasets and are
recorded: MaleCNS `type` starting with "KC" (processed node set); FlyWire `cell_class` ==
"Kenyon_Cell" (flywire_annotations) with Shiu et al.'s `Connectivity_783.parquet`.

    uv run python scripts/analysis/kc_recurrence.py \\
        --processed data/processed/male-cns-v1.0/superclass_annotated \\
        --shiu-repo <clone @91bdd1e> --flywire-annotations <clone @a83b277> \\
        --out docs/acceptance/evidence/kc-recurrence.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from flylab.data.processed import load_graph


def _summary(n_kc: int, kk: float, into_kc: float) -> dict[str, object]:
    return {
        "n_kenyon_cells": n_kc,
        "kc_to_kc_synapses": int(kk),
        "kc_to_kc_synapses_per_kc": round(kk / n_kc, 1),
        "synapses_onto_kcs_per_kc": round(into_kc / n_kc, 1),
        "fraction_of_kc_input_from_kcs": round(kk / into_kc, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--shiu-repo", type=Path, required=True)
    ap.add_argument("--flywire-annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    g = load_graph(a.processed)
    t = pq.read_table(a.processed / "neurons.parquet", columns=["type", "consensus_nt"])
    types = [x or "" for x in t["type"].to_pylist()]
    kc = np.array([x.startswith("KC") for x in types])
    w = g.synapse_count.astype(np.float64)
    nt = Counter(v for v, k in zip(t["consensus_nt"].to_pylist(), kc, strict=True) if k)
    mc = _summary(int(kc.sum()), float(w[kc[g.src] & kc[g.dst]].sum()), float(w[kc[g.dst]].sum()))
    mc |= {"selector": "type starts with 'KC'", "consensus_nt": dict(nt)}
    mc_kk = float(w[kc[g.src] & kc[g.dst]].sum()) / int(kc.sum())

    con = pq.read_table(
        a.shiu_repo / "Connectivity_783.parquet",
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"],
    )
    ann = pcsv.read_csv(
        a.flywire_annotations / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv",
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(
            include_columns=["root_id", "cell_class"],
            column_types={"root_id": pa.int64(), "cell_class": pa.string()},
        ),
    )
    ids = np.array(
        [r for r, c in zip(ann["root_id"].to_pylist(), ann["cell_class"].to_pylist(), strict=True)
         if c == "Kenyon_Cell"],
        dtype=np.int64,
    )  # fmt: skip
    pre = np.isin(con["Presynaptic_ID"].to_numpy(), ids)
    post = np.isin(con["Postsynaptic_ID"].to_numpy(), ids)
    cw = con["Connectivity"].to_numpy().astype(np.float64)
    fw = _summary(int(ids.size), float(cw[pre & post].sum()), float(cw[post].sum()))
    fw |= {"selector": "flywire_annotations cell_class == 'Kenyon_Cell'"}
    fw_kk = float(cw[pre & post].sum()) / int(ids.size)

    out = {
        "schema_version": 1,
        "kind": "flylab_kc_recurrence",
        "evidence": "EMPIRICAL (connectome counts); selectors differ between datasets",
        "malecns_v1_0_superclass_annotated": mc,
        "flywire_v783_shiu_connectivity": fw,
        "ratio_kc_to_kc_per_kc_malecns_over_flywire": round(mc_kk / fw_kk, 2),
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
