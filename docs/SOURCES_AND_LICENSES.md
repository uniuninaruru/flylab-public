# Sources and licenses

Audit date: **2026-11-04**. Method noted per row. Re-check before relying on any value; upstream
pages and objects can change. FlyLab code is Apache-2.0 (see `OPEN_QUESTIONS.md` Q1).

## Datasets

| Item | Verified fact | How verified | Evidence |
|---|---|---|---|
| MaleCNS release | v1.0 dated **2026-06-08** (v0.9 was 2025-10-05). No later release listed. | https://male-cns.janelia.org/release/ | EMPIRICAL |
| MaleCNS license | Page text: "The Male CNS dataset is licensed under CC-BY", with "CC-BY" linked to https://creativecommons.org/licenses/by/4.0/ (version 4.0 taken from that link) | https://male-cns.janelia.org/download/ (footer and body) | EMPIRICAL |
| Access | Feather files are in public bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, readable over HTTPS at `storage.googleapis.com` **without credentials** (HTTP 200 on HEAD). neuPrint API needs an account token; FlyLab does not need it for the MVP. | `curl -I` on each file | EMPIRICAL |
| File sizes and hashes | The download page lists 7 Feather files; their sizes + MD5 + CRC32C from `x-goog-hash` headers are recorded in `src/flylab/data/sources.py`. Weights file: 1,051,241,946 bytes. The bucket folder holds 11 objects (2026-11-04 listing by the independent audit): also `-traced-only` and `-significant-only` variants of the weights file, which are not registered. | `curl -I` | EMPIRICAL |
| Neuron count (paper) | "This contains 166,691 neurons spanning the brain and nerve cord" (abstract), 11,691 types. | research.google publication page for the Cell paper (Berg et al., 2026) | EMPIRICAL (publication) |
| Neuron count (flybrain) | README says "all 167,106 neurons". Derivation not yet audited. | HEREISCB/flybrain README | third-party claim, UNKNOWN derivation |

Download page column notes (from the page itself): `syn-partners` has columns `x_pre, y_pre, z_pre,
body_pre, conf_pre, x_post, y_post, z_post, body_post, conf_post, primary_post`; synapse locations
are in 8 nm voxels. The column schema of `connectome-weights-*.feather` is not stated on the page;
read from the files on 2026-11-04 (`flylab data inspect`, evidence in
`docs/acceptance/evidence/raw-data-audit-male-cns-v1.0.json`): weights = `body_pre`, `body_post`,
`weight` (int64, 151,856,684 rows); annotations = 211,577 rows keyed by `bodyId` with `status`,
`superclass`, `type`, ...; neurotransmitters = 1,835,518 rows keyed by `body` with `predicted_nt`,
`consensus_nt`, `ground_truth`, ...

## Reference code (consulted, not copied)

| Repository | License (verified via GitHub API + LICENSE file) | Use in FlyLab |
|---|---|---|
| [HEREISCB/flybrain](https://github.com/HEREISCB/flybrain) | MIT (with portions from tel-0s/flyverse, MIT). GitHub shows "NOASSERTION" only because of the extra copyright line. | Reference for MaleCNS LIF and viewer. Adapting code is license-compatible with attribution; audit correctness first. |
| [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain) | **GPL-2.0** | Ideas and benchmarks only. **Do not copy code** into this Apache-2.0 repo. FlyWire female connectome. |
| [TuragaLab/flybody](https://github.com/TuragaLab/flybody) | Apache-2.0 | Dependency of the isolated body env since PR-05 (see below). Not vendored. |
| [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) | Apache-2.0 | Alternative body backend (optional). |
| [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) @ `91bdd1e` | MIT (LICENSE file checked 2026-11-04) | PR-02: parameter values and equations read for `MODEL_ASSUMPTIONS.md`; no code copied. PR-04: its FlyWire v630/v783 connectivity files are read **locally** to build a cross-check graph (`scripts/analysis/build_flywire_shiu_graph.py`, output under git-ignored `data/`) and for the Q19 comparison; `figures.ipynb` read for the sugar GRN and MN9 root ids. Nothing redistributed. |
| [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) @ `a83b277` (2026-09-29) | **no license file**; README asks to cite Berg et al. 2026, Schlegel et al. 2024, Tastekin et al. 2026, Matsliah et al. 2024, Dorkenwald et al. 2024 | PR-02 analyses only (FlyWire `cell_sub_class`/`cell_type`/`top_nt` for Q8/Q13); PR-04: `side`/`cell_type` for the Q19 structural comparison (`scripts/analysis/q19_mn9_laterality.py`). Nothing copied or redistributed; the evidence JSON holds aggregate counts and the root ids already cited in the paper/notebook |
| [brian-team/brian2](https://github.com/brian-team/brian2) 2.5.1 (source read) / 2.10.1 (run) | CeCILL 2.1 (LICENSE file of tag 2.5.1) | PR-02: semantics read from source; `scripts/crosscheck_brian2.py` runs it in a throwaway env (`uv run --with brian2==2.10.1`). **Not** a FlyLab dependency, nothing vendored. |

## Body stack (`envs/body`, PR-05; verified 2026-11-04)

Installed only into the isolated `envs/body` environment (own `pyproject.toml` + `uv.lock`), never
into the main FlyLab lock. No code is copied into this repository. Details and the full version
matrix: `BODY_BACKEND.md`.

| Item | Version / revision | License | How verified |
|---|---|---|---|
| flybody | 0.1.0, git commit `d015e9bfe441bd90ae431bac24c55cb74bdbce26` (`main` HEAD per `git ls-remote`, commit date 2025-07-30). Not on PyPI (404). | Apache-2.0: `LICENSE` at that commit is the Apache 2.0 text (differs from apache.org's file only by one leading blank line). It is the only license file in the repo, so it also covers the model assets (`fruitfly.xml`, meshes). | `git clone` + `diff -w` against https://www.apache.org/licenses/LICENSE-2.0.txt |
| flybody install notes | README: conda env with `python=3.10` and `cudatoolkit=11.8.0`; `pip install -e .` for the core; `[tf]` extra pins `tensorflow==2.8.0`, `nvidia-cudnn-cu11==8.9.*`. FlyLab uses the core only, Python 3.12, no CUDA. `pyproject.toml` pins `numpy==1.26.4`, `requires-python >=3.10`. | — | README + `pyproject.toml` at the commit |
| MuJoCo | 3.17.0 (PyPI upload 2026-10-22) | Apache-2.0 (`License-Expression` in wheel metadata) | installed metadata |
| dm_control | 1.0.50 (PyPI upload 2026-10-23), requires `mujoco>=3.17.0` | Apache-2.0 (PyPI metadata) | PyPI JSON API + installed metadata |
| dm-env 1.6, dm-tree 0.1.10, labmaze 1.0.6, mediapy 1.2.7 | lock | Apache-2.0 | installed metadata |
| NumPy 1.26.4, SciPy 1.17.1, PyOpenGL 3.1.10 | lock | BSD | installed metadata (classifiers) |
| h5py 3.16.0 | lock | BSD-3-Clause | installed metadata |
| glfw (Python) 2.11.0 | lock | MIT | installed metadata |
| FlyGym | 2.1.1 on PyPI (2026-10-24, Python `>=3.12,<3.16`) | Apache-2.0 (PyPI metadata) | PyPI JSON API. **Not installed or tested.** |

## GPU software stack (smoke-tested on the RTX host 2026-11-04)

| Item | Verified fact | Source |
|---|---|---|
| PyTorch | Latest stable **2.15.0** (requires Python ≥ 3.11). PyPI Linux wheel depends on `cuda-toolkit==13.2.2` (a CUDA 13.2 build). Also on `download.pytorch.org/whl/cu132`. cu126 builds stop at 2.14.x and predate Blackwell; not used. | PyPI JSON API, PyTorch wheel index |
| CUDA 13.x driver floor | CUDA 13.0–13.2 apps run on drivers **≥ 580** (minor-version compatibility); 13.2 corresponds to branch R595. CUDA 13 drops pre-Turing GPUs (irrelevant for RTX 5070). | NVIDIA CUDA Toolkit release notes |
| RTX 5070 | Blackwell, compute capability 12.0 (`sm_120`) per spec. The torch 2.15.0 wheel's `get_arch_list()` on the host includes `sm_120` (VERIFIED 2026-11-04, `docs/acceptance/evidence/`). | host run |
| Base images | `ubuntu:24.04@sha256:fe8156…f852` (updated 2026-10-29); `ghcr.io/astral-sh/uv:0.12.17@sha256:10787c…9acc`; GPU passthrough check `nvidia/cuda:13.0.2-base-ubuntu24.04@sha256:2ab638…1701`. All tags confirmed to exist. | Docker Hub API, GHCR registry API |
| CI actions | `actions/checkout` v7.0.1, `astral-sh/setup-uv` v10.3.3, pinned by commit SHA. | GitHub API |

## Papers

- Shiu et al. (2024), *Nature*, doi:10.1038/s41586-024-07763-9 — LIF method precedent, **female** brain
  (FlyWire). PR-02 took parameter **values** (not code) from the paper's reference code,
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) at commit
  `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` (2024-09-14), `model.py` lines 22–37 (quoted in
  `MODEL_ASSUMPTIONS.md` §3). License of that repository: MIT (LICENSE file, "Copyright (c) 2023
  Philip Shiu and Nico Spiller"). The **Methods** were read from nature.com on 2026-11-04 after the
  session network settings were changed (article: CC BY 4.0); every LIF value equals the code, the
  sign rule is quoted in `MODEL_ASSUMPTIONS.md` §4 (`OPEN_QUESTIONS.md` Q13, answered). The
  Supplementary Tables were not read.
- Berg et al. (2026), *Cell*, MaleCNS connectome paper — counts above.
- flybody paper (2025), *Nature*, doi:10.1038/s41586-025-09029-4 — body model (PR-05). Vaxenburg
  et al., *Nature* 643, 1312–1320 (citation as given in the flybody README). Its locomotion
  controllers are learned policies, not connectome-derived. Not yet read for any parameter.

## Python dependencies added in PR-02

| Package | Version (uv.lock) | License (installed metadata) | Use |
|---|---|---|---|
| PyYAML | 6.0.3 | MIT | experiment configs, `yaml.safe_load` only |
| types-PyYAML (dev) | 6.0.12.20261103 | Apache-2.0 | type stubs for mypy |

## Python dependencies added in PR-04

| Package | Version (uv.lock) | License (installed metadata) | Use |
|---|---|---|---|
| matplotlib (optional extra `plots`) | 3.11.2 | Matplotlib license (PSF-based, OSI approved) | `scripts/m4/plots.py` only; nothing under `src/` imports it |
| its dependencies: pillow 13.0.0 (MIT-CMU), contourpy 1.4.0 (BSD-3-Clause), cycler 0.12.1 (BSD), fonttools 4.68.0 (MIT), kiwisolver 1.5.1 (BSD), pyparsing 3.3.3 (MIT), python-dateutil 4.0.0 (BSD-3-Clause), packaging 26.3 (Apache-2.0 OR BSD-2-Clause), tzdata 2026.6 (Apache-2.0) | as listed | as listed | transitive, `plots` extra only |

## Paper text read in PR-04

Shiu et al. 2024 HTML (nature.com, fetched 2026-11-04, CC BY 4.0) for: the `w_syn` criterion and
its references 64 (Dahanukar et al. 2007, *Neuron* 56:503-516) and 65 (Inagaki et al. 2012, *Cell*
148:583-595) — neither defines "maximal MN9 firing" numerically in the text read here (the two
papers themselves were not read); "Recently, the FlyWire brain (FAFB) dataset ... was found to be
left-right inverted. In this paper, we refer to the true biological side."; "Unless specifically
stated, when describing MN9 activity, we refer to the MN9 contralateral to the GRNs activated";
"unilateral sugar GRN activation activates the contralateral MN9 more strongly compared with the
ipsilateral MN9 when either the left (Fig. 1c) or the right (Extended Data Fig. 1d) hemisphere
GRNs are activated"; sugar GRNs were driven at "10-200 Hz".
