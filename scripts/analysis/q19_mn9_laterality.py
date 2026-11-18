"""Q19: which MN9 do LB3c (sugar) GRNs reach, by side, in MaleCNS v1.0 and in FlyWire v783?

Read-only structural analysis; nothing here is simulated. For each dataset and each side pair
(GRN side, MN9 side) it reports:

1. direct GRN -> MN9 synapses;
2. two-hop synapse products sum_X syn(G->X) * syn(X->MN9), split by the sign of X;
3. input-normalised excitatory path strength for 1..4 hops: x_{k+1}[j] = sum_i x_k[i] *
   syn(i->j) / in_synapses(j) over excitatory presynaptic i only (x_0 = 1 on the GRNs);
4. the side profile of the GRNs' direct targets and of MN9's direct inputs (synapse-weighted).

Side labels: MaleCNS `rootSide` (sensory) / `somaSide` (others); FlyWire `side` of
flyconnectome/flywire_annotations. Signs: MaleCNS `consensus_nt` with the Shiu et al. Methods
rule (preset `shiu2024_methods_v1`; histamine/unclear/missing = unknown, excluded); FlyWire the
`Excitatory` column of Shiu et al.'s `Connectivity_783.parquet`.

Usage (repository root; external files are inputs, never committed):

    uv run python scripts/analysis/q19_mn9_laterality.py \\
        --processed data/processed/male-cns-v1.0/superclass_annotated \\
        --raw data/raw/male-cns-v1.0 \\
        --shiu-repo <clone of philshiu/Drosophila_brain_model @91bdd1e> \\
        --flywire-annotations <clone of flyconnectome/flywire_annotations @a83b277> \\
        --out docs/acceptance/evidence/q19-mn9-laterality.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.dataset as pads
import pyarrow.feather as feather
import pyarrow.parquet as pq
from numpy.typing import NDArray

from flylab.data.processed import load_graph
from flylab.neural.signs import SHIU2024_METHODS_V1

HOPS = 4
SIDES = ("L", "R")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


class Net:
    """Directed weighted graph on dense indices with a per-presynaptic-neuron sign."""

    def __init__(
        self,
        n: int,
        src: NDArray[np.int64],
        dst: NDArray[np.int64],
        syn: NDArray[np.float64],
        sign: NDArray[np.int8],  # per neuron (the sign of its output synapses)
    ) -> None:
        self.n, self.src, self.dst, self.syn, self.sign = n, src, dst, syn, sign
        self.in_syn = np.bincount(dst, weights=syn, minlength=n)

    def direct(self, a: NDArray[np.int64], b: NDArray[np.int64]) -> float:
        ma, mb = self._mask(a), self._mask(b)
        return float(self.syn[ma[self.src] & mb[self.dst]].sum())

    def two_hop(self, a: NDArray[np.int64], b: NDArray[np.int64]) -> dict[str, float]:
        ma, mb = self._mask(a), self._mask(b)
        out_a = np.bincount(self.dst[ma[self.src]], weights=self.syn[ma[self.src]],
                            minlength=self.n)  # fmt: skip
        into_b = np.bincount(self.src[mb[self.dst]], weights=self.syn[mb[self.dst]],
                             minlength=self.n)  # fmt: skip
        prod = out_a * into_b
        return {
            "via_excitatory": float(prod[self.sign > 0].sum()),
            "via_inhibitory": float(prod[self.sign < 0].sum()),
            "via_unknown_sign": float(prod[self.sign == 0].sum()),
        }

    def propagate(self, a: NDArray[np.int64], hops: int) -> list[NDArray[np.float64]]:
        x = np.zeros(self.n)
        x[a] = 1.0
        exc = self.sign[self.src] > 0
        w = np.where(exc, self.syn / np.maximum(self.in_syn[self.dst], 1.0), 0.0)
        out = []
        for _ in range(hops):
            x = np.bincount(self.dst, weights=x[self.src] * w, minlength=self.n)
            out.append(x)
        return out

    def side_profile(
        self, a: NDArray[np.int64], side: NDArray[np.object_], outgoing: bool
    ) -> dict[str, float]:
        ma = self._mask(a)
        sel = ma[self.src] if outgoing else ma[self.dst]
        other = self.dst[sel] if outgoing else self.src[sel]
        w = self.syn[sel]
        total = float(w.sum())
        labels = np.array(["?" if s is None else str(s) for s in side[other]], dtype=object)
        return {str(k): round(float(w[labels == k].sum()) / total, 4) for k in sorted(set(labels))}

    def inputs(self, idx: NDArray[np.int64]) -> dict[str, int]:
        """Input partners and synapses of one neuron, split by the presynaptic sign."""
        m = self.dst == int(idx[0])
        s = self.sign[self.src[m]]
        w = self.syn[m]
        return {
            "input_partners": int(m.sum()),
            "input_synapses": int(w.sum()),
            "from_excitatory": int(w[s > 0].sum()),
            "from_inhibitory": int(w[s < 0].sum()),
            "from_unknown_sign": int(w[s == 0].sum()),
        }

    def _mask(self, idx: NDArray[np.int64]) -> NDArray[np.bool_]:
        m = np.zeros(self.n, dtype=bool)
        m[idx] = True
        return m


def analyse(
    net: Net,
    grn: Mapping[str, NDArray[np.int64]],
    mn9: Mapping[str, NDArray[np.int64]],
    side: NDArray[np.object_],
) -> dict[str, object]:
    res: dict[str, object] = {}
    for gs in SIDES:
        prop = net.propagate(grn[gs], HOPS)
        for ms in SIDES:
            res[f"GRN_{gs}->MN9_{ms}"] = {
                "direct_synapses": net.direct(grn[gs], mn9[ms]),
                "two_hop_synapse_products": net.two_hop(grn[gs], mn9[ms]),
                "excitatory_input_fraction_by_hop": [float(x[mn9[ms]].sum()) for x in prop],
            }
        res[f"GRN_{gs}_output_partner_side"] = net.side_profile(grn[gs], side, outgoing=True)
    for ms in SIDES:
        res[f"MN9_{ms}_input_partner_side"] = net.side_profile(mn9[ms], side, outgoing=False)
        res[f"MN9_{ms}_inputs"] = net.inputs(mn9[ms])
    return res


def malecns(processed: Path, raw: Path) -> tuple[dict[str, object], dict[str, object]]:
    g = load_graph(processed)
    n = g.n_neurons
    neurons = pq.read_table(processed / "neurons.parquet", columns=["body_id", "type",
                                                                     "consensus_nt"])  # fmt: skip
    body = neurons["body_id"].to_numpy()
    if not np.array_equal(body, g.index.body_ids):
        raise SystemExit("neurons.parquet order differs from graph.npz")
    ann = feather.read_table(
        raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "type", "somaSide", "rootSide", "somaLocation", "exitNerve",
                 "entryNerve", "instance", "statusLabel", "status"],
    ).to_pylist()  # fmt: skip
    by_id = {int(r["bodyId"]): r for r in ann}
    side = np.empty(n, dtype=object)
    for i, b in enumerate(body):
        r = by_id.get(int(b), {})
        side[i] = r.get("somaSide") or r.get("rootSide")
    types = np.array(neurons["type"].to_pylist(), dtype=object)
    nt = neurons["consensus_nt"].to_pylist()
    table = SHIU2024_METHODS_V1.signs
    sign = np.array([table.get("<null>" if v is None else v, 0) for v in nt], dtype=np.int8)
    grn = {s: np.flatnonzero((types == "LB3c") & (side == s)).astype(np.int64) for s in SIDES}
    mn9 = {s: np.flatnonzero((types == "MN9") & (side == s)).astype(np.int64) for s in SIDES}
    net = Net(n, g.src.astype(np.int64), g.dst.astype(np.int64),
              g.synapse_count.astype(np.float64), sign)  # fmt: skip
    meta = {
        "graph_fingerprint_sha256": g.fingerprint(),
        "n_neurons": n,
        "n_edges": g.n_edges,
        "side_rule": "somaSide, else rootSide (sensory neurons have rootSide only)",
        "sign_rule": "consensus_nt -> shiu2024_methods_v1 (histamine/unclear/missing = 0)",
        "grn_body_ids": {s: [int(body[i]) for i in grn[s]] for s in SIDES},
        "mn9": {
            s: [
                {
                    k: by_id[int(body[i])].get(k)
                    for k in (
                        "instance",
                        "somaSide",
                        "somaLocation",
                        "exitNerve",
                        "statusLabel",
                        "status",
                    )
                }
                | {"body_id": int(body[i])}
                for i in mn9[s]
            ]
            for s in SIDES
        },
    }
    mn9_ids = [int(body[i]) for s in SIDES for i in mn9[s]]
    weights = pads.dataset(
        raw / "connectome-weights-male-cns-v1.0-minconf-0.5.feather", format="feather"
    ).to_table(
        columns=["body_post", "weight"],
        filter=pc.field("body_post").isin(pa.array(mn9_ids, type=pa.int64())),
    )
    meta["mn9_input_synapses_all_segments"] = {
        str(b): int(pc.sum(weights.filter(pc.equal(weights["body_post"], b))["weight"]).as_py())
        for b in mn9_ids
    }
    meta["mn9_input_synapses_all_segments_note"] = (
        "raw weights file, every presynaptic segment including fragments outside the node set"
    )
    return analyse(net, grn, mn9, side), meta


def flywire(shiu: Path, fwa: Path) -> tuple[dict[str, object], dict[str, object]]:
    con = pq.read_table(
        shiu / "Connectivity_783.parquet",
        columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"],
    )
    ann_path = fwa / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
    cols = ["root_id", "cell_type", "side", "soma_x", "cell_sub_class"]
    ann = pcsv.read_csv(
        ann_path,
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(
            include_columns=cols,
            column_types={
                "root_id": pa.int64(),
                "cell_type": pa.string(),
                "side": pa.string(),
                "soma_x": pa.float64(),
                "cell_sub_class": pa.string(),
            },
        ),
    )
    pre = con["Presynaptic_ID"].to_numpy()
    post = con["Postsynaptic_ID"].to_numpy()
    ids = np.unique(np.concatenate([pre, post]))
    src = np.searchsorted(ids, pre)
    dst = np.searchsorted(ids, post)
    n = ids.size
    exc = con["Excitatory"].to_numpy()
    sign = np.zeros(n, dtype=np.int8)
    sign[src] = np.sign(exc).astype(np.int8)  # one value per presynaptic neuron (checked: 0 mixed)
    root = ann["root_id"].to_numpy()
    pos = np.searchsorted(ids, root)
    ok = (pos < n) & (ids[np.minimum(pos, n - 1)] == root)
    side = np.empty(n, dtype=object)
    ctype = np.empty(n, dtype=object)
    raw_side = ann["side"].to_pylist()
    raw_type = ann["cell_type"].to_pylist()
    side_map = {"left": "L", "right": "R", "center": "M"}
    for k in np.flatnonzero(ok):
        side[pos[k]] = side_map.get(str(raw_side[k]), raw_side[k])
        ctype[pos[k]] = raw_type[k]
    grn = {s: np.flatnonzero((ctype == "LB3c") & (side == s)).astype(np.int64) for s in SIDES}
    mn9 = {s: np.flatnonzero((ctype == "CB0701") & (side == s)).astype(np.int64) for s in SIDES}
    net = Net(n, src.astype(np.int64), dst.astype(np.int64),
              con["Connectivity"].to_numpy().astype(np.float64), sign)  # fmt: skip
    shiu_mn9 = 720575940660219265
    meta = {
        "connectivity": "Connectivity_783.parquet (Shiu et al. reference repository)",
        "n_neurons_in_connectivity": int(n),
        "n_rows": int(con.num_rows),
        "side_rule": "flywire_annotations `side` (left/right/center -> L/R/M)",
        "sign_rule": "Shiu et al. `Excitatory` column (their Methods rule on FlyWire)",
        "grn_counts": {s: int(grn[s].size) for s in SIDES},
        "mn9_root_ids": {s: [int(ids[i]) for i in mn9[s]] for s in SIDES},
        "shiu_mn9_root_id": shiu_mn9,
        "shiu_mn9_side": next((str(side[i]) for i in np.flatnonzero(ids == shiu_mn9)), "not found"),
    }
    return analyse(net, grn, mn9, side), meta


def shiu_sugar_sides(fwa: Path, sugar_ids: Sequence[int]) -> dict[str, int]:
    ann = pcsv.read_csv(
        fwa / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv",
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(
            include_columns=["root_id", "side"],
            column_types={"root_id": pa.int64(), "side": pa.string()},
        ),
    )
    lookup = dict(zip(ann["root_id"].to_pylist(), ann["side"].to_pylist(), strict=True))
    out: dict[str, int] = {}
    for i in sugar_ids:
        k = str(lookup.get(i, "not found"))
        out[k] = out.get(k, 0) + 1
    return out


#: The 21 "sugar GRN" root ids of Shiu et al.'s figures.ipynb (FlyWire v630), as used in
#: docs/acceptance/evidence/open-questions-q8-q13-q14.md.
SHIU_SUGAR_V630 = (
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--shiu-repo", type=Path, required=True)
    ap.add_argument("--flywire-annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mc, mc_meta = malecns(a.processed, a.raw)
    fw, fw_meta = flywire(a.shiu_repo, a.flywire_annotations)
    fw_meta["shiu_sugar_v630_ids_side_in_current_annotations"] = shiu_sugar_sides(
        a.flywire_annotations, SHIU_SUGAR_V630
    )
    out = {
        "schema_version": 1,
        "kind": "flylab_q19_mn9_laterality",
        "evidence": "EMPIRICAL (connectome counts and annotations); the input-normalised "
        "excitatory path strength is an ENGINEERED summary, not a dynamics prediction",
        "inputs": {
            "malecns_processed_graph": str(a.processed),
            "malecns_annotations_sha256": _sha256(
                a.raw / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
            ),
            "shiu_connectivity_783_sha256": _sha256(a.shiu_repo / "Connectivity_783.parquet"),
            "flywire_annotations_tsv_sha256": _sha256(
                a.flywire_annotations
                / "supplemental_files"
                / "Supplemental_file1_neuron_annotations.tsv"
            ),
        },
        "malecns": {"meta": mc_meta, "results": mc},
        "flywire_783": {"meta": fw_meta, "results": fw},
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
