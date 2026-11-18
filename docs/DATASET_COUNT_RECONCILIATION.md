# Dataset count reconciliation

Status: **167,106 reproduced exactly; 166,691 within 9 bodies (cause UNKNOWN).**
Measured 2026-11-04 on MaleCNS v1.0 files whose size and MD5 match upstream (SHA-256 in each
processed `manifest.json`). Measured by `flylab data inspect` (schema audit) and
`flylab data prepare` (commits `67dc887` and `c29d70c`, identical graph fingerprint), run on the
owner's GPU host in the `data` container. The distinct-segment count (88,384,522) comes from the
`67dc887` run; since `c29d70c` it is measured only with `--count-all-bodies`.

No count below is a pass/fail denominator. The simulation's node set is a named rule recorded in
every manifest (`node_set.rule`).

## Counts in circulation

| Count | Source | What it counts | Evidence |
|---|---|---|---|
| 166,691 neurons | MaleCNS Cell paper abstract (Berg et al., 2026) | "neurons spanning the brain and nerve cord" | EMPIRICAL (publication) |
| 167,106 neurons | HEREISCB/flybrain README | its graph node count | third-party claim, **reproduced below** |

## What the v1.0 annotation file contains (`body-annotations-male-cns-v1.0-minconf-0.5.feather`)

211,577 rows, one per body (`bodyId` unique, no nulls).

| `status` | bodies | of which `superclass` non-null |
|---|---:|---:|
| Traced | 165,122 | 164,606 |
| Orphan | 15,925 | 32 |
| Glia | 11,864 | 0 |
| Unimportant | 10,751 | 0 |
| (null) | 5,472 | 2,002 |
| Assign | 1,832 | 0 |
| Anchor | 611 | 60 |
| **total** | **211,577** | **166,700** |

`superclass` takes 27 distinct non-null values (largest: `ol_intrinsic` 89,403, `cb_intrinsic` 32,164,
`vnc_intrinsic` 13,161, `visual_projection` 9,201); 44,877 bodies have none.
Photoreceptor types (`R1-R6`, `R7*`, `R8*`, 10 type names): 6,091 bodies, 1,984 of them not Traced;
all of them have a `superclass`.

## Node-set rules (implemented in `flylab.data.malecns.NODE_SETS`)

| Rule | Definition | Bodies | Relation to published counts |
|---|---|---:|---|
| `superclass_annotated` (**default**) | curated `superclass` is non-null | **166,700** | paper 166,691 + 9 |
| `traced` | `status == "Traced"` | 165,122 | — |
| `flybrain_compat` | Traced, or type is a photoreceptor | **167,106** | **= flybrain README exactly** (165,122 + 1,984) |

### 167,106 (flybrain) — RESOLVED

flybrain's `flybrain/connectome.py` (MIT) keeps `status == "Traced"` plus every photoreceptor
body regardless of status. Applying that rule to v1.0 gives 165,122 + 1,984 = 167,106.

### 166,691 (paper) — 9-body difference, cause UNKNOWN

The non-null-`superclass` rule is the closest measured definition (166,700). Hypotheses, none
tested: the paper's snapshot predates v1.0's "minor proofreading changes and refinement of neuron
annotations" (release notes, 2026-06-08); or the paper excludes a few bodies by a criterion not
present in the flat files. Resolving it needs the paper's methods/supplement or the authors.

## Connection table (`connectome-weights-male-cns-v1.0-minconf-0.5.feather`)

Columns `body_pre`, `body_post`, `weight` (all int64, no nulls; `weight` 1–2,591).

| Quantity | Value |
|---|---:|
| rows (one per segment pair as published; uniqueness checked only among the 25,582,938 kept rows, none repeated) | 151,856,684 |
| synapses (sum of `weight`) | 311,833,243 |
| distinct segment ids appearing as pre or post | 88,384,522 |

### Composition of the default node set

`superclass_annotated` is not "Traced neurons": it contains 164,606 Traced, 2,002 null-status,
60 Anchor and 32 Orphan bodies, and leaves out the 516 Traced bodies that have no `superclass`.
The independent audit (issue uniuninaruru/flylab#8) reproduced these counts and found that 1,983
of the 2,002 null-status bodies are `R1-R6` photoreceptors (`ol_sensory`). Choosing this rule over
`traced` is a scientific decision recorded as open question Q7; both are selectable.

Cross-check candidate: the bucket also publishes `connectome-weights-...-traced-only.feather`
(508,025,642 bytes, not registered). An official traced-only edge list could validate the
`traced` node set's edges; not used yet.

Restricted to the default node set (`superclass_annotated`, 166,700 bodies):

| Quantity | Value |
|---|---:|
| rows kept (both ends in node set) = unique directed edges | **25,582,938** |
| synapses kept | **124,177,617** (39.8 % of all) |
| rows with only the presynaptic end in the node set | 112,633,588 |
| rows with only the postsynaptic end in the node set | 4,802,752 |
| rows with neither end in the node set | 8,837,406 |
| node-set bodies appearing in any weights row | 166,576 (124 never appear) |
| isolated neurons in the node-set graph (no edge within it) | 217 |
| self-loops (autapses) | 101 |

Three different numbers are reported on purpose: input rows, unique directed edges and synapses.
The spec's "~125 million synapses" figure is close to the node-set synapse count, but it is a
different definition; neither is used as a denominator.

Most dropped rows are neuron → small-fragment connections (`rows_pre_only`), i.e. synapses onto
segments that are not annotated neurons. Whether to model them (e.g. as an unattributed sink) is
an open modelling question, not an import decision.
