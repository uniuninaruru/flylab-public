# Evidence: open questions Q8, Q13, Q14 (and Q16 input), 2026-11-04

Cloud session (Linux x86_64, 4 CPUs, no GPU). Read-only analyses of public data; nothing here is
a biological result. Analysis scripts are reproduced below and were run with
`uv run --no-project --python 3.12 --with pandas==2.3.3 --with pyarrow==26.0.1 python -I <script>`.

## Inputs

| Input | Version / hash | License / terms |
|---|---|---|
| MaleCNS v1.0 flat connectome (fetched with `flylab data fetch`, size + MD5 checked) | annotations `2177e246…a3b2`, neurotransmitters `95c92892…9621`, weights `e35da783…afc1` (SHA-256) | CC-BY (MaleCNS download page) |
| Processed graph (`flylab data prepare`, node set `superclass_annotated`) | fingerprint `4b2a5d7650e68e0c…` — **identical** to the PR-01 run on the owner's host (`prepare-manifest-superclass_annotated-c29d70c.json`); `flylab data verify`: ok | — |
| Shiu et al. 2024, *Nature* 634:210–219, doi:10.1038/s41586-024-07763-9, HTML from nature.com | SHA-256 of the page as fetched `4ce320fd…20ca` | CC BY 4.0 (article page) |
| philshiu/Drosophila_brain_model | `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` | MIT |
| flyconnectome/flywire_annotations | `a83b2776d60d5764cef36b927f5f9679c16c47a2` (2026-09-29) | no license file; README asks to cite Berg et al. 2026, Schlegel et al. 2024, Tastekin et al. 2026, Matsliah et al. 2024, Dorkenwald et al. 2024. Used for analysis only, nothing redistributed |

## Q13 — Shiu et al. Methods vs code

Sentences from the Methods ("Computational model", "Neurotransmitter predictions", "Computational
modelling of water and sugar GRN activity"), quoted verbatim:

```text
Vresting = −52 mV (resting potential from ref. 18);
Vreset = −52 mV (reset potential after spike18);
Vthreshold = −45 mV (threshold for spiking18);
Rmbr = 10  Kohm cm2 (membrane resistance18);
Trefractory = 2.2 ms (refractory period18,59);
Cmbr = 2 µF cm−2 (membrane capacitance18);
Tmbr = Cmb × Rmbr (definition of membrane timescale in a resistor–capacitor circuit);
τ = 5 ms (synapse decay timescale61);
Tdly = 1.8 ms (time delay from spike to change in membrane potential from ref. 62);
Wsyn = 0.275 mV (free parameter; synaptic weight, that is, how much each synapse influences downstream membrane potential);
gi (the synaptic conductance resulting from the aggregate firing of neurons presynaptic to neuron i).
All parameters are taken from previous Drosophila modelling or electrophysiology efforts18,19,61,62, from the synaptic weights from the Flywire connectome (public materialization v.630)18,19, or from the neurotransmitter predictions3,63, except for Wsyn, the single free parameter of the model, which corresponds to how much the downstream membrane potential changes as a result of a single excitatory or inhibitory synapse. We chose Wsyn such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing64,65.
The connection weight, wj,i, between neuron j and neuron i is the synaptic connectivity weight from the Flywire connectivity multiplied by either 1, if neuron j is excitatory or −1, if neuron j is inhibitory, multiplied by Wsyn.
We stimulated particular neurons with Poisson distributed input. The neural model is implemented in Brian2 (ref. 22), and 30 simulations of 1,000 ms for each experiment were performed. Simulation of sugar neuron activation takes approximately 5 min per 1,000 ms trial per central processing unit thread. All 127,400 proofread neurons from Flywire materialization v.630 are included in the model.
We assume GABAergic and glutamatergic neurons are inhibitory66, and that each neuron is either exclusively inhibitory or excitatory. As in ref. 67, we used a cleft score cutoff of 50, and identified the highest neurotransmitter prediction for each presynaptic site and, if greater than half of all the presynaptic sites across the entire neuron are predicted to be inhibitory (GABA or Glut), we assigned this neuron as inhibitory. Neurons predicted to be dopaminergic, octopaminergic or serotonergic are assigned to the excitatory category.
For computational silencing, water or sugar GRNs were activated, and each of the top 200 firing neurons was silenced by eliminating all output of that neuron.
Unless specifically stated, when describing MN9 activity, we refer to the MN9 contralateral to the GRNs activated.
We chose Wsyn such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing64,65.
Upon firing, a neuron’s membrane potential is reset to the resting potential, and cannot change for the duration of the refractory time period.
We implement a leaky integrate-and-fire model as previously described18,19,59,60,61 with α-synapse dynamics, using the following three differential equations and parameters:
```

Every LIF value equals the reference code (`model.py` lines 22–37); `Tmbr = Cmbr × Rmbr =
2 µF cm⁻² × 10 kΩ cm² = 20 ms`. `dt` is not stated in the Methods (the code never sets it; Brian2
default 0.1 ms). The Methods call the synapse "α-synapse" but print the single-exponential
equation `dg/dt = −g/τ`, which is what the code and FlyLab implement.

Cross-check of the sign rule against the authors' published v783 connectivity (one `Excitatory`
value per presynaptic neuron) joined with FlyWire `top_nt`:

```text
presynaptic neurons: 138005  with mixed Excitatory sign: 0
presynaptic neurons not in annotations: 12

(1) FlyWire top_nt (annotations v3.x HEAD) x Shiu 'Excitatory' (per presynaptic neuron):
exc               -1      1
top_nt                     
<missing>          7      5
acetylcholine   1719  84162
dopamine           3   5897
gaba           17731   1348
glutamate      21863   2941
octopamine         1    192
serotonin          9   2127

FlyWire rows with cell_type CB0701 or synonyms mentioning MN9:
                 root_id          cell_class cell_type         top_nt synonyms
393   720575940660219265  brain_motor_neuron    CB0701  acetylcholine      NaN
1523  720575940618238523  brain_motor_neuron    CB0701  acetylcholine      NaN

(2) FlyWire LB3a top downstream types (synapses): CB0407:425, CB0062:326, CB0008:307, DNg67:296, LB3a:176, AN_GNG_30:172, VP5+Z_adPN:112, CB0192:105

(2) FlyWire LB3b top downstream types (synapses): AN_GNG_68:556, LB3b:345, AN_GNG_30:267, CB0248:175, AN_GNG_21:175, DNpe030:149, CB0366:147, CB0393:122

(2) FlyWire LB3c top downstream types (synapses): CB0192:1009, LB3c:947, CB0248:619, CB0616:434, CB0118:426, CB0008:388, AN_GNG_30:338, CB0062:331

(2) FlyWire LB3d top downstream types (synapses): CB0248:602, LB3d:482, AN_GNG_30:455, CB0616:410, CB0118:303, CB0445:296, AN_GNG_21:271, CB0366:260
```

<details><summary><code>flywire_checks.py</code></summary>

```python
"""(1) Shiu et al. sign rule vs FlyWire top_nt; (2) FlyWire LB3a-d downstream type profiles."""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

shiu, fwa = sys.argv[1], sys.argv[2]
con = pq.read_table(
    f"{shiu}/Connectivity_783.parquet",
    columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity", "Excitatory"],
).to_pandas()
ann = pd.read_csv(
    f"{fwa}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    sep="\t",
    usecols=[
        "root_id",
        "cell_class",
        "cell_sub_class",
        "cell_type",
        "top_nt",
        "known_nt",
        "synonyms",
    ],
    dtype={"root_id": "int64"},
    low_memory=False,
)
pre = con.groupby("Presynaptic_ID")["Excitatory"].agg(["min", "max"])
print(
    "presynaptic neurons:",
    len(pre),
    " with mixed Excitatory sign:",
    int((pre["min"] != pre["max"]).sum()),
)
pre = (
    pre["min"]
    .rename("exc")
    .reset_index()
    .merge(ann, left_on="Presynaptic_ID", right_on="root_id", how="left")
)
print("presynaptic neurons not in annotations:", int(pre["root_id"].isna().sum()))
tab = pd.crosstab(pre["top_nt"].fillna("<missing>"), pre["exc"])
print("\n(1) FlyWire top_nt (annotations v3.x HEAD) x Shiu 'Excitatory' (per presynaptic neuron):")
print(tab.to_string())

types = ann.set_index("root_id")["cell_type"]
print("\nFlyWire rows with cell_type CB0701 or synonyms mentioning MN9:")
m = ann[(ann["cell_type"] == "CB0701") | ann["synonyms"].fillna("").str.contains("MN9")]
print(m[["root_id", "cell_class", "cell_type", "top_nt", "synonyms"]].to_string())

out = {}
for t in ["LB3a", "LB3b", "LB3c", "LB3d"]:
    ids = ann.loc[ann["cell_type"] == t, "root_id"]
    sub = con[con["Presynaptic_ID"].isin(ids)].copy()
    sub["post_type"] = sub["Postsynaptic_ID"].map(types).fillna("<untyped>")
    out[t] = sub.groupby("post_type")["Connectivity"].sum()
prof = pd.DataFrame(out).fillna(0)
prof.to_csv(sys.argv[3])
for t in prof:
    top = prof[t].sort_values(ascending=False).head(8)
    print(
        f"\n(2) FlyWire {t} top downstream types (synapses):",
        ", ".join(f"{k}:{int(v)}" for k, v in top.items()),
    )
```

</details>

## Q14 — transmitter columns (MaleCNS v1.0, node set `superclass_annotated`)

```text
nt columns: ['body', 'cell_type', 'total_nt_predictions', 'predicted_nt_confidence', 'predicted_nt', 'ground_truth', 'celltype_total_nt_predictions', 'celltype_predicted_nt', 'celltype_predicted_nt_confidence', 'consensus_nt']
node set: 166700
node-set bodies missing from NT table: 178

predicted_nt: acetylcholine=94947, glutamate=28058, gaba=20231, unclear=16151, dopamine=4443, histamine=2126, serotonin=465, <null>=178, octopamine=101

celltype_predicted_nt: acetylcholine=98512, glutamate=29016, gaba=21483, histamine=7969, unclear=4621, dopamine=4448, serotonin=386, <null>=178, octopamine=87

consensus_nt: acetylcholine=103720, glutamate=29302, gaba=22069, histamine=7891, unclear=2999, dopamine=392, <null>=178, octopamine=101, serotonin=48

ground_truth: <null>=81216, acetylcholine=52779, glutamate=14397, gaba=13150, histamine=4683, dopamine=380, octopamine=51, serotonin=44

neurons with ground_truth: 85484
  predicted_nt: agreement with ground_truth where not unclear = 0.9921 (n=76348); unclear among GT neurons = 0.1069
  celltype_predicted_nt: agreement with ground_truth where not unclear = 0.9930 (n=85425); unclear among GT neurons = 0.0007
  consensus_nt: agreement with ground_truth where not unclear = 1.0000 (n=85484); unclear among GT neurons = 0.0000

consensus_nt == ground_truth where GT present: 1.0
where no GT: consensus == celltype_predicted: 0.9203235815602837  consensus == predicted: 0.8698901694247438
types with >1 distinct predicted_nt: 654 of 11751
types with >1 distinct celltype_predicted_nt: 0 of 11751
types with >1 distinct consensus_nt: 0 of 11751
types with >1 distinct ground_truth: 0 of 3318

untyped node-set neurons: 2194 consensus unclear among them: 0.2415679124886053
```

<details><summary><code>nt_columns.py</code></summary>

```python
"""Empirical relationships between the MaleCNS v1.0 transmitter columns (node set superclass_annotated)."""

import sys
from collections import Counter
import numpy as np
import pyarrow.feather as feather
import pyarrow.compute as pc

raw = sys.argv[1]
ann = feather.read_table(
    f"{raw}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    columns=["bodyId", "superclass", "type", "class"],
).to_pandas()
nt = feather.read_table(f"{raw}/body-neurotransmitters-male-cns-v1.0.feather").to_pandas()
print("nt columns:", list(nt.columns))
ann = ann[ann["superclass"].notna()]
print("node set:", len(ann))
df = ann.merge(nt, left_on="bodyId", right_on="body", how="left")
print("node-set bodies missing from NT table:", int(df["body"].isna().sum()))
cols = ["predicted_nt", "celltype_predicted_nt", "consensus_nt", "ground_truth"]
for c in cols:
    vc = df[c].fillna("<null>").value_counts()
    print(f"\n{c}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
gt = df[df["ground_truth"].notna()]
print(f"\nneurons with ground_truth: {len(gt)}")
for c in ["predicted_nt", "celltype_predicted_nt", "consensus_nt"]:
    known = gt[gt[c] != "unclear"]
    agree = (known[c] == known["ground_truth"]).mean()
    print(
        f"  {c}: agreement with ground_truth where not unclear = {agree:.4f} (n={len(known)}); unclear among GT neurons = {(gt[c] == 'unclear').mean():.4f}"
    )
# consensus vs others
print(
    "\nconsensus_nt == ground_truth where GT present:",
    float((gt["consensus_nt"] == gt["ground_truth"]).mean()),
)
nogt = df[df["ground_truth"].isna()]
print(
    "where no GT: consensus == celltype_predicted:",
    float((nogt["consensus_nt"] == nogt["celltype_predicted_nt"]).mean()),
    " consensus == predicted:",
    float((nogt["consensus_nt"] == nogt["predicted_nt"]).mean()),
)
# within-type constancy
typed = df[df["type"].notna()]
for c in ["predicted_nt", "celltype_predicted_nt", "consensus_nt", "ground_truth"]:
    g = typed.dropna(subset=[c]).groupby("type")[c].nunique()
    print(f"types with >1 distinct {c}: {int((g > 1).sum())} of {len(g)}")
untyped = df[df["type"].isna()]
print(
    "\nuntyped node-set neurons:",
    len(untyped),
    "consensus unclear among them:",
    float((untyped["consensus_nt"] == "unclear").mean()),
)
```

</details>

## Q8 — LB3c (FlyWire "sugar") and MN9 in MaleCNS

FlyWire annotations: `cell_sub_class` "sugar" ↔ `cell_type` LB3c (32 neurons), "sugar/low_salt"
↔ LB3b, "water" ↔ LB3a, "high_salt/heavy_metal" ↔ LB3d. MaleCNS `flywireType` is only "LB3" for
all four, so the a–d letters were checked by downstream partner-type profiles:

```text
Shiu sugar GRN ids (v630, figures.ipynb) found in FlyWire v783 annotations: 20 of 21; types: {'LB3c': np.int64(9), 'LB3d': np.int64(5), 'LB4b': np.int64(4), 'LB3b': np.int64(2)} sub classes: {'sugar': np.int64(9), 'high_salt/heavy_metal': np.int64(5), 'putative_attractive': np.int64(4), 'sugar/low_salt': np.int64(2)}

Cosine similarity of downstream partner-type synapse profiles (rows MaleCNS, cols FlyWire):
           FW_LB3a  FW_LB3b  FW_LB3c  FW_LB3d
MCNS_LB3a    0.875    0.405    0.481    0.504
MCNS_LB3b    0.401    0.922    0.352    0.734
MCNS_LB3c    0.406    0.323    0.932    0.656
MCNS_LB3d    0.472    0.667    0.658    0.953

best FlyWire match per MaleCNS type: {'MCNS_LB3a': 'FW_LB3a', 'MCNS_LB3b': 'FW_LB3b', 'MCNS_LB3c': 'FW_LB3c', 'MCNS_LB3d': 'FW_LB3d'}
MaleCNS LB3a top downstream (FlyWire-named) types: CB0008:486, CB0062:481, DNg67:388, LB3a:306, AN_GNG_30:221, CB0118:196, CB0964:173, CB0393:163
MaleCNS LB3b top downstream (FlyWire-named) types: AN_GNG_68:427, AN_GNG_21:327, LB3b:310, AN_GNG_30:273, LB3d:254, CB0248:251, CB0008:152, CB0366:147
MaleCNS LB3c top downstream (FlyWire-named) types: LB3c:1737, CB0248:1149, CB0118:890, CB0192:766, CB0616:519, AN_GNG_30:511, CB0008:488, CB0499:447
MaleCNS LB3d top downstream (FlyWire-named) types: CB0248:821, LB3d:704, CB0118:695, AN_GNG_30:616, CB0616:485, CB0366:412, LB3c:370, AN_GNG_68:343
```

<details><summary><code>lb3_match.py</code></summary>

```python
"""Do MaleCNS LB3a-d match FlyWire LB3a-d? Compare downstream partner-type profiles (FlyWire naming)."""

import sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

raw, fwa, fw_prof_csv = sys.argv[1], sys.argv[2], sys.argv[3]
ann = pd.read_csv(
    f"{fwa}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    sep="\t",
    usecols=["root_id", "cell_type", "cell_sub_class"],
    dtype={"root_id": "int64"},
    low_memory=False,
)
sugar = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]
hit = ann[ann["root_id"].isin(sugar)]
print(
    f"Shiu sugar GRN ids (v630, figures.ipynb) found in FlyWire v783 annotations: {len(hit)} of 21;",
    "types:",
    dict(hit["cell_type"].value_counts()),
    "sub classes:",
    dict(hit["cell_sub_class"].value_counts()),
)

mann = feather.read_table(
    f"{raw}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    columns=["bodyId", "type", "flywireType"],
).to_pandas()
label = mann.set_index("bodyId")["flywireType"].fillna(mann.set_index("bodyId")["type"])
w = feather.read_table(f"{raw}/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
out = {}
for t in ["LB3a", "LB3b", "LB3c", "LB3d"]:
    ids = pa.array(mann.loc[mann["type"] == t, "bodyId"].to_numpy())
    sub = w.filter(pc.is_in(w["body_pre"], value_set=ids)).to_pandas()
    sub["post_type"] = sub["body_post"].map(label).fillna("<untyped>")
    # FlyWire names LB3 subtypes; MaleCNS flywireType says only "LB3" -> use MaleCNS type for LB3* partners
    sub.loc[sub["post_type"] == "LB3", "post_type"] = sub["body_post"].map(
        mann.set_index("bodyId")["type"]
    )
    out[t] = sub.groupby("post_type")["weight"].sum()
mprof = pd.DataFrame(out).fillna(0)
fprof = pd.read_csv(fw_prof_csv, index_col=0)
idx = sorted((set(mprof.index) | set(fprof.index)) - {"<untyped>"})
M, F = mprof.reindex(idx).fillna(0), fprof.reindex(idx).fillna(0)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


sim = pd.DataFrame(
    {f"FW_{f}": [cos(M[m].to_numpy(), F[f].to_numpy()) for m in M] for f in F},
    index=[f"MCNS_{m}" for m in M],
)
print(
    "\nCosine similarity of downstream partner-type synapse profiles (rows MaleCNS, cols FlyWire):"
)
print(sim.round(3).to_string())
print("\nbest FlyWire match per MaleCNS type:", {r: sim.loc[r].idxmax() for r in sim.index})
for t in M:
    top = M[t].sort_values(ascending=False).head(8)
    print(
        f"MaleCNS {t} top downstream (FlyWire-named) types:",
        ", ".join(f"{k}:{int(v)}" for k, v in top.items()),
    )
```

</details>

MaleCNS bodies by side (`rootSide` for GRNs, `somaSide` for MN9):

```text
LB3c dimorphism: {'<null>': np.int64(23)} fruDsx: {'<null>': np.int64(23)}
   rootSide=L: [78240, 85806, 180314, 190769, 261450, 262567, 512551, 531237, 933317, 158893964, 349137284, 475202322]
   rootSide=R: [72059, 92440, 101087, 120303, 140015, 142827, 202888, 215556, 516217, 942168, 957530]
MN9 dimorphism: {'<null>': np.int64(2)} fruDsx: {'<null>': np.int64(2)}
   somaSide=L: [10331]
   somaSide=R: [16949]
LB3b dimorphism: {'<null>': np.int64(11)} fruDsx: {'<null>': np.int64(11)}
   rootSide=L: [71254, 159772, 272263, 174444965, 766547228]
   rootSide=R: [163597, 183084, 187492, 209155, 557646, 213650853]
LB3d dimorphism: {'<null>': np.int64(26)} fruDsx: {'<null>': np.int64(26)}
   rootSide=L: [93290, 130693, 140851, 178321, 204726, 236475, 312102, 317616, 512552, 516216, 581710868]
   rootSide=R: [33339, 125405, 129171, 158537, 199196, 201075, 210344, 261048, 516218, 550065, 916953, 147619274, 245892505, 422510463, 885642755]
LB4b dimorphism: {'<null>': np.int64(8)} fruDsx: {'<null>': np.int64(8)}
   rootSide=L: [128129, 152400, 816904261, 884238775]
   rootSide=R: [72047, 122806, 538117, 923024]
```

<details><summary><code>sides.py</code></summary>

```python
import sys
import pyarrow.feather as feather

raw = sys.argv[1]
ann = feather.read_table(
    f"{raw}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    columns=["bodyId", "type", "rootSide", "somaSide", "dimorphism", "fruDsx", "superclass"],
).to_pandas()
for t in ["LB3c", "MN9", "LB3b", "LB3d", "LB4b"]:
    sub = ann[ann["type"] == t].sort_values("bodyId")
    print(
        t,
        "dimorphism:",
        dict(sub["dimorphism"].fillna("<null>").value_counts()),
        "fruDsx:",
        dict(sub["fruDsx"].fillna("<null>").value_counts()),
    )
    for side in ("L", "R"):
        col = "somaSide" if t == "MN9" else "rootSide"
        print(f"   {col}={side}:", sub.loc[sub[col] == side, "bodyId"].tolist())
```

</details>

## Q16 input — synapses per neuron

```text
Shiu v630: neurons 127,400, connection rows 14,687,178, synapses 52,793,639, synapses/neuron 414.4
Shiu v783: neurons 138,639, connection rows 15,091,983, synapses 54,492,922, synapses/neuron 393.1
```

MaleCNS v1.0 `superclass_annotated`: 166,700 neurons, 124,177,617 synapses → 744.9 per neuron
(1.80 × FlyWire v630, the graph on which `w_syn = 0.275 mV` was chosen). Synapse definitions
differ (confidence thresholds; MaleCNS includes the VNC), so this is a ratio of counts, not of
biology.
<details><summary><code>syn_per_neuron.py</code></summary>

```python
import sys
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.csv as pcsv

shiu = sys.argv[1]
for tag, con, comp in (
    ("v630", "2023_03_23_connectivity_630_final.parquet", "2023_03_23_completeness_630_final.csv"),
    ("v783", "Connectivity_783.parquet", "Completeness_783.csv"),
):
    t = pq.read_table(f"{shiu}/{con}", columns=["Connectivity"])
    n = pcsv.read_csv(f"{shiu}/{comp}").num_rows
    syn = int(pc.sum(t["Connectivity"]).as_py())
    print(
        f"Shiu {tag}: neurons {n:,}, connection rows {t.num_rows:,}, synapses {syn:,}, synapses/neuron {syn / n:.1f}"
    )
```

</details>

## Exploratory runs (`configs/male_cns_lb3c_mn9.yaml` and rate/seed variants)

Code commit(s): `92b614d` (dirty=False). NumPy backend, float32,
dt 0.1 ms, 1 s per run, forced Poisson spikes in the 12 left LB3c GRNs (`rootSide` L), unknown
signs excluded. Rates are mean Hz over 1 s. **Single seeds unless the seed column differs; `w_syn`
not recalibrated for MaleCNS; exploratory, not validation.**

| Run | Seed | LB3c rate requested (Hz) | Mapping | Column | Overrides | Edges | LB3c L (Hz) | MN9 L (Hz) | MN9 R (Hz) | All neurons (Hz) | Finite | Simulate s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `lb3c` | 1 | 100 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 84.0 | 46.0 | 0.0 | 3.985 | True | 40.3 |
| `lb3c--sens-fast-ionotropic-map` | 1 | 100 | `fast_ionotropic_v1` | `consensus_nt` | — | 24,559,135 | 84.1 | 6.0 | 0.0 | 0.175 | True | 38.0 |
| `lb3c--sens-glutamate-excitatory` | 1 | 100 | `shiu2024_methods_v1` | `consensus_nt` | glutamate=+1 | 24,904,953 | 84.0 | 84.0 | 0.0 | 36.155 | True | 84.8 |
| `lb3c--sens-predicted-nt-column` | 1 | 100 | `shiu2024_methods_v1` | `predicted_nt` | — | 24,821,643 | 84.0 | 29.0 | 0.0 | 4.135 | True | 40.6 |
| `r10` | 1 | 10 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 9.7 | 0.0 | 0.0 | 0.001 | True | 15.7 |
| `r100s2` | 2 | 100 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 81.9 | 42.0 | 0.0 | 4.207 | True | 35.4 |
| `r100s3` | 3 | 100 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 81.7 | 20.0 | 1.0 | 0.440 | True | 28.9 |
| `r100s4` | 4 | 100 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 77.6 | 33.0 | 0.0 | 4.307 | True | 34.7 |
| `r100s5` | 5 | 100 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 85.8 | 19.0 | 0.0 | 4.300 | True | 30.3 |
| `r150` | 1 | 150 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 114.6 | 13.0 | 0.0 | 0.349 | True | 34.9 |
| `r150s2` | 2 | 150 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 113.1 | 25.0 | 1.0 | 0.292 | True | 33.0 |
| `r150s3` | 3 | 150 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 114.5 | 21.0 | 0.0 | 0.376 | True | 30.0 |
| `r150s4` | 4 | 150 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 108.8 | 21.0 | 0.0 | 0.340 | True | 34.8 |
| `r150s5` | 5 | 150 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 116.0 | 21.0 | 1.0 | 0.274 | True | 31.9 |
| `r200` | 1 | 200 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 140.4 | 28.0 | 0.0 | 4.300 | True | 37.3 |
| `r50` | 1 | 50 | `shiu2024_methods_v1` | `consensus_nt` | — | 24,904,953 | 45.5 | 8.0 | 0.0 | 0.148 | True | 24.7 |

Runs `r*` used copies of `configs/male_cns_lb3c_mn9.yaml` with only `experiment_id`, `stimuli[0].rate_hz`, `seed` (`s2`–`s5`) and `recording.spike_events: false` changed and `sensitivity` removed; run `lb3c*` is the committed config itself (`flylab run --config configs/male_cns_lb3c_mn9.yaml`). The same numbers were obtained in an earlier pass at commit `8e92b98` (identical engine code for these runs).

Full per-run provenance (config hash, dataset hashes, git, signs): `open-questions-runs-92b614d.json`.
