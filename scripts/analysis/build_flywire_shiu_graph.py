"""Build a FlyLab processed directory from Shiu et al.'s published FlyWire connectivity.

CROSS-CHECK ONLY (PR-04, OPEN_QUESTIONS Q19): lets the FlyLab engine run on the graph that
Shiu et al. (2024) used, so a MaleCNS result can be separated into "connectome difference" and
"FlyLab vs Brian2 model difference". This is the FlyWire *female* brain, not MaleCNS.

Input: a clone of github.com/philshiu/Drosophila_brain_model (MIT) at commit 91bdd1e, which
ships `2023_03_23_connectivity_630_final.parquet` + `2023_03_23_completeness_630_final.csv`
(v630, the graph of the paper's figures) and `Connectivity_783.parquet` + `Completeness_783.csv`.
Nodes = every neuron of the completeness table (as in `model.py`); edges = connectivity rows;
sign = the `Excitatory` column (their Methods rule, one value per presynaptic neuron).

Output goes under data/ (git-ignored); nothing here is redistributed. Usage:

    uv run python scripts/analysis/build_flywire_shiu_graph.py \\
        --shiu-repo <clone> --version 630 --out data/processed/flywire-shiu-630
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from flylab.data.graph import DuplicateRule, build_graph

FILES = {
    "630": ("2023_03_23_connectivity_630_final.parquet", "2023_03_23_completeness_630_final.csv"),
    "783": ("Connectivity_783.parquet", "Completeness_783.csv"),
}
SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from flylab.data.processed import write_processed

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shiu-repo", type=Path, required=True)
    ap.add_argument("--version", choices=sorted(FILES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    con_name, comp_name = FILES[a.version]
    con_path, comp_path = a.shiu_repo / con_name, a.shiu_repo / comp_name
    comp = pcsv.read_csv(comp_path)
    ids = comp.column(0).cast(pa.int64()).to_numpy()
    con = pq.read_table(
        con_path, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"]
    )
    exc = con["Excitatory"].to_numpy()
    if not np.isin(exc, (-1, 1)).all():
        raise SystemExit("Excitatory must be +1/-1")
    graph, report = build_graph(
        ids,
        con["Presynaptic_ID"].to_numpy(),
        con["Postsynaptic_ID"].to_numpy(),
        con["Connectivity"].to_numpy().astype(np.int64),
        exc.astype(np.int8),
        dataset_id=f"flywire:v{a.version}:shiu2024-repo",
        duplicate_rule=DuplicateRule.ERROR,
    )
    write_processed(
        a.out,
        graph,
        report,
        {},
        {
            "release": f"philshiu/Drosophila_brain_model@{SHIU_COMMIT} ({con_name})",
            "node_set": {"rule": f"every row of {comp_name} (as model.py does)"},
            "inputs": [
                {"file": con_name, "sha256": _sha256(con_path)},
                {"file": comp_name, "sha256": _sha256(comp_path)},
            ],
            "note": "CROSS-CHECK ONLY: FlyWire female brain as used by Shiu et al. 2024; "
            "signs from their Excitatory column. Not MaleCNS. Local use, not redistributed.",
        },
    )
    print(f"wrote {a.out}: {graph.n_neurons:,} neurons, {graph.n_edges:,} edges, "
          f"{graph.total_synapses:,} synapses, fingerprint {graph.fingerprint()}")  # fmt: skip


if __name__ == "__main__":
    main()
