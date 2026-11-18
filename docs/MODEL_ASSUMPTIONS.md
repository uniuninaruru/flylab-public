# Model assumptions

Status: **PR-02** adds a CPU leaky integrate-and-fire (LIF) engine: a NumPy reference
(`flylab.neural.lif_numpy`) and a PyTorch CPU engine (`flylab.neural.lif_torch`) with the same
math. This file is the specification those engines implement. If code and this file disagree,
the code is wrong.

Nothing here is a biological measurement of MaleCNS neurons. Every parameter is uniform across
all neurons and carries one evidence label (`flylab.provenance.labels.Evidence`).

## 1. Data-level conventions (PR-00 / PR-01, unchanged)

| Item | Convention | Label |
|---|---|---|
| Edge orientation | `src` = presynaptic, `dst` = postsynaptic (`flylab.data.graph.ORIENTATION`) | ENGINEERED (convention) |
| Node index | body ids sorted ascending → dense `int32` index | ENGINEERED (deterministic) |
| Synapse count | stored as given by the dataset, `int64`, must be > 0 | EMPIRICAL once imported |
| Duplicate (pre, post) rows | refused by default; `SUM` only when chosen explicitly, signs must agree, direction kept | ENGINEERED |
| Sign in the processed graph | PR-01 stores `0` = **unknown** for every MaleCNS edge (not zero weight). The encoding is declared on disk in Parquet field metadata (`flylab.sign_encoding`) and in `graph.npz` (`sign_encoding`), because `count * sign` would silently zero unknown edges. PR-02 assigns signs **at run time** (section 4); the processed files are never rewritten | UNKNOWN in the data |
| Unknown sign handling | consumer must pick `UnknownSignPolicy` (`error`, `exclude`, `as_excitatory`, `as_inhibitory`); the run config has **no default** and the choice is recorded | ENGINEERED; `as_*` only for sensitivity analysis |
| Efficacy | not stored in the graph; `w = w_syn × sign × synapse_count` (section 3) | ASSUMED |
| Dense matrices | refused above 4,096 neurons (`flylab.data.memory`); engines use sparse gather + scatter-add only | ENGINEERED |
| MaleCNS node set | default `superclass_annotated` (non-null curated `superclass`, 166,700 bodies in v1.0); recorded as `node_set.rule` in every manifest | ENGINEERED (rule) over EMPIRICAL annotations |
| MaleCNS edges | weights rows with both ends in the node set; `weight` (synapse confidence >= 0.5) is the synapse count. Rows to segments outside the node set are dropped by the importer, so the engine never sees them (`OPEN_QUESTIONS.md` Q11) | EMPIRICAL |
| Provenance granularity | spec 6.1 sketches per-row `provenance` / `weight_provenance` columns; FlyLab records one evidence label per column in `manifest.json["columns"]` (every written column, including `idx`, `src_idx`, `dst_idx`), since every row of a column has the same origin | ENGINEERED (convention) |
| Input identity | `prepare` refuses inputs whose size + MD5 differ from the registered upstream object; synthetic tests opt out explicitly and are labelled ENGINEERED | ENGINEERED |

## 2. Neuron and synapse model

Continuous-time model, per neuron *i* (current-based, single-exponential synaptic drive):

```text
tau_m · dv_i/dt = (V_rest − v_i) + g_i + e_i(t)
tau_s · dg_i/dt = −g_i
```

- `v` membrane potential (mV). `g` synaptic drive (mV): `R·I_syn` expressed as the voltage it
  would add at steady state. `e` external injected drive (mV): `R·I_ext`, piecewise constant per
  step. The membrane resistance `R` is **UNKNOWN**, so drives are specified in mV, not pA.
- A presynaptic spike from *j* that arrives at *i* adds `w_ji` (mV) to `g_i` (instantaneous jump).
- Spike when `v_i > V_th` (strict). Reset `v_i ← V_reset`, `g_i ← 0`.
- These are the equations of the Shiu et al. (2024) reference code, `model.py` lines 44–52 at
  commit `91bdd1e` (quoted in section 3). That code's comment says "alpha synapse", but the
  equations it executes are the single-exponential form above; FlyLab implements the equations.

### 2.1 Discrete-time scheme (what the engines compute)

Step `n` covers `[t_n, t_{n+1})` with `t_n = n·dt`. State at the start of step `n`: `v`, `g`,
`last_i` (step of the last spike, `−∞` if none) and a FIFO of propagated spike sets for the last
`D` steps. Per-step inputs (from the experiment schedule, never from the engine's own RNG):
injected drive `e[n]`, forced-spike requests `F[n]`, output-blocked set `B[n]`, clamped set `C[n]`.

1. **Refractory status.** `ref_i = (n − last_i) < R` where `R = t_ref/dt` (integer).
   `active_i = ¬ref_i ∧ ¬C_i`.
2. **Subthreshold update** (exact for the linear ODE with `e` constant over the step; same as
   Brian2 `method='linear'`). For active neurons only — inactive neurons keep `v`, `g` unchanged:

   ```text
   a = exp(−dt/tau_m)        d = exp(−dt/tau_s)        c = 1 − a
   b = tau_s/(tau_s − tau_m) · (d − a)      (b = (dt/tau_m)·a  if tau_s = tau_m)
   # for |x| < 1e-3 with x = dt·(1/tau_m − 1/tau_s) the equal, cancellation-free form
   # b = a·(dt/tau_m)·expm1(x)/x is used (continuous at tau_s = tau_m; unused for the Shiu preset)
   v ← V_rest + (v − V_rest)·a + g·b + e·c
   g ← g·d
   ```

3. **Spike generation.** `θ_i = active_i ∧ (v_i > V_th)`; accepted forced spikes
   `φ_i = active_i ∧ F_i`; `spike_i = θ_i ∨ φ_i`. Forced requests on refractory or clamped
   neurons are **dropped and counted** (`n_forced_dropped`). `cause = forced` if `φ_i`, else
   `threshold`. `last_i ← n`. At most one spike per neuron per step ("repeated spikes" cannot
   occur; a forced request on a neuron that also crosses threshold yields one spike, cause `forced`).
4. **Synaptic arrival.** `A_i = Σ_{j→i} w_ji · [j ∈ P[n−D]]`, accumulated in **float64**, then
   rounded once to the state dtype. `g_i ← g_i + A_i` only where the neuron is now not refractory
   (`¬ref_i ∧ ¬spike_i`) and not clamped. Arrivals at refractory, just-spiked or clamped neurons
   are **discarded** — this is Brian2's conditional write for `(unless refractory)` variables,
   which the Shiu code relies on (section 3.2).
5. **Reset.** `spike_i → v_i ← V_reset, g_i ← 0`. `C_i → v_i ← V_rest, g_i ← 0`.
6. **Propagation.** `P[n] = {i : spike_i ∧ ¬B_i}` is queued and arrives in phase 4 of step
   `n + D`, `D = delay/dt` (integer ≥ 1). Spikes already in flight when a silencing window starts
   still arrive.

Consequences, checked by tests:

- After a spike at step `s`, integration resumes at step `s + R`; the first spike from rest under a
  constant drive `e > V_th − V_rest` is at step `k* − 1`, and the inter-spike interval is
  `R + k* − 1` steps, where `k* = min{k : V_rest + e(1 − a^k) > V_th}`.
- A single arrival of weight `w` at step `n₀` gives, `k ≥ 0` steps later,
  `v = V_rest + w · tau_s/(tau_s − tau_m) · (d^k − a^k)` (no other input, no spike).
- **Spike time convention:** a spike recorded at step `n` crossed threshold during
  `[t_n, t_{n+1})` and is reported at `t_n = n·dt` (left edge; same convention as Brian2).
- **Initial conditions:** `v = V_rest`, `g = 0`, no neuron refractory, nothing in flight.

### 2.2 Numerical choices

| Choice | Value | Label / reason |
|---|---|---|
| Integration | exact exponential propagator (above), constants computed in float64 then cast | ENGINEERED: dt-robust and identical to Brian2 `linear` |
| Order within a step | refractory → integrate → threshold → arrivals → reset → enqueue | ENGINEERED: same slot order as Brian2's default schedule `groups, thresholds, synapses, resets` |
| `dt` | set per run (`dt_ms`); `t_ref/dt` and `delay/dt` must be integers (relative tolerance 1e-9), else the config is **rejected** (no silent rounding) | ENGINEERED |
| Delay | one uniform delay for every edge | ASSUMED (section 3) |
| State dtype | `float64` default; `float32` allowed (`dtype` in the config) | ENGINEERED |
| Arrival accumulation | float64 for both dtypes, one rounding to the state dtype | ENGINEERED: makes NumPy and torch agree independent of summation order at the float32 level |
| Propagation | event driven: CSR rows of the spiking presynaptic neurons are gathered, then scatter-added (`np.bincount` / `Tensor.index_add_`) | ENGINEERED; no dense `N×N` |
| RNG | the engines have none; Poisson forced spikes are drawn by the experiment schedule from NumPy `PCG64(SeedSequence([seed, stimulus_index]))` | ENGINEERED: identical inputs for every backend |
| Backends | `numpy` (reference), `torch-cpu`, `torch-cuda` (PR-03): one implementation of the scheme for both torch devices; `torch-cuda` also has a fused implementation (PR-03b, `engine_impl`) | ENGINEERED |
| Fused CUDA implementation | four Triton kernels per step (stamp inputs → integrate/threshold/spike lists → arrivals → apply/reset), no host synchronization inside a batch of `batch_steps` (default 32) steps; spike lists copied to the host once per batch and sorted there | ENGINEERED: the eager engine is launch/sync-bound (M3); results are unchanged by the batch size (tests) |
| Arrival accumulation (fused) | exact integer sums of signed synapse counts per target (int64 atomics), scaled once by `w_syn` in float64, then rounded to the state dtype | ENGINEERED: deterministic whatever order the GPU adds in; differs from the reference only when a neuron receives several arrivals in one step (section 2.3) |
| Edge temporaries (torch) | arrivals processed in chunks of ≤ `max_edges_per_chunk` (default 4,194,304) edges + one neuron's out-degree, ≈ 40 bytes/edge | ENGINEERED: bounded memory; on the CPU the per-target addition order, hence the result, is unchanged (tests) |

### 2.3 CUDA backend: what can differ, and how CPU ↔ GPU results are compared

`torch-cuda` runs the code of `torch-cpu` on the device. Each tensor operation of section 2.1 is
the same IEEE-754 binary operation (`+`, `−`, `×`, comparisons, `where`) executed as its own
kernel, so there is no fused multiply-add across operations, and the propagator constants are
the same float64 values rounded once to the state dtype. **The only operation whose result may
depend on the device is the float64 sum of arrivals inside `index_add_`**: with
`deterministic: false` CUDA adds them with atomics in arbitrary order; with `deterministic: true`
PyTorch uses a reproducible reduction whose order is fixed but not promised to equal the CPU's.

**Engineering margins, not universal error bounds.** The same margin `δ` is enforced on
both |v_CPU − v_GPU| and |g_CPU − g_GPU|. `DEFAULT_MARGIN_MV` specifies baseline tolerances of
**2e-6 mV for float64** and **1e-2 mV for float32**, at the Shiu preset with `dt = 0.1 ms`.
These are ENGINEERED acceptance tolerances. They are not derived bounds for arbitrary graphs,
weights, drives, durations or reset histories; passing parity is empirical evidence for the
particular run. The old float64 baseline of 1e-6 mV did not cover the corrected illustrative
calculation below, so it is raised to 2e-6 mV. The float32 baseline is retained: the existing
host evidence gives no measured basis for choosing a tighter window.

For clarity, the arrival-summation argument is as follows. With unit roundoff `u = 2⁻⁵³`,
`m` finite float64 arrivals, and no overflow/underflow, the usual sequential-sum estimate is
`gamma_(m−1) · sum|w|`, where `gamma_k = k·u / (1 − k·u)` and `k·u < 1`. **Both engines round**,
so their sums can differ by up to `E = 2 · gamma_(m−1) · sum|w|`. To first order this is
`2 · (m−1) · u · sum|w|`; the missing factor 2 in the earlier text mattered.

In the linear subthreshold recurrence, a same-sign additive error of at most `E` in g at
**every step** gives steady-state envelopes `E / (1−d)` in g and
`E · b / ((1−a)(1−d))` in v. A fresh arithmetic error of at most `E_v` in v each step contributes
`E_v / (1−a)` as well. These are repeated-error gains, not the peak response to a single
perturbation. They depend on the discretization, hence on `dt` and both time constants.
For the **illustrative, unmeasured** choice `m = 10⁴` and `sum|w| = 10⁴ mV` each step, the
arrival-only v envelope is approximately **1.11e-6 mV at dt = 0.1 ms**, and **1.11e-5 mV at
0.01 ms**. Neither arrival count nor incoming strength in that example is an established bound
on the MaleCNS workloads. It omits fresh arithmetic errors and threshold/reset discontinuities.

`default_margin_mV` scales each baseline upward by `max(1, G / G_base)`, with
`G = max(1/(1−d), 1/(1−a), b/((1−a)(1−d)))` and `G_base` at the Shiu preset's 0.1 ms step.
The float64 default is thus approximately 2e-5 mV at 0.01 ms, covering the illustrative
arrival-only envelope at both step sizes. This scaling is an engineering allowance for longer
error persistence, **not** a proof covering a run's incoming weights or float32 rounding.
For float32, the float64 arrival sum is rounded to the state dtype, and subsequent operations
round again on already different inputs. A one-ulp argument alone does not bound that accumulated
error. New workloads/discretizations require empirical checks; `--margin-mv` is an explicit,
recorded override and must be finite and positive. The recorded full-graph M3 float32 runs had
max |Δv| = max |Δg| = **0**; their original margins and evidence records remain unchanged.

Comparison policy (`flylab parity`, `tests/gpu`, `scripts/gpu_smoke.py`):

1. Both engines get the same per-step inputs (the schedule draws them once). The network,
   dtype, populations and discretization must match. Thresholds may differ for diagnostic
   comparisons, but the check below uses each engine's own threshold rounded to its state dtype.
2. Compare initial states, every `state_every` steps, at the end, and **before each differing
   spike step**. Both |Δv| and |Δg| must be ≤ δ; `last_spike`, step and every in-flight spike slot
   must match exactly. Non-finite state in either engine fails (`non_finite`), including phase-2
   candidates hidden by a reset or clamp; no NaN can count as equality. Finiteness is checked
   every step; periodic margin checks do not bound unsampled finite excursions between them.
3. Compare spikes every step, including forced/propagated flags on **all common spikes**.
   Different flags always fail. A spike-set difference is a justified near-threshold flip only
   if the pre-step states agree as above and every involved neuron is active, is not a
   forced-only spike, and has an **actual phase-2 candidate potential in both engines** within
   δ of its respective threshold. Each backend retains the candidates it really computed,
   before resets; a reconstructed reference formula cannot excuse an erroneous tested update.
4. After **each** justified flip, load the reference's full post-step state into the tested
   engine with `set_state`, including the step, refractory history and in-flight slots. Continue
   comparing every subsequent step, so every later flip needs its own justification. Record
   `steps_compared`, `justified_flips` (steps with an accepted spike-set difference) and `resyncs`.
   `steps_identical` counts matching spike/flag steps across the whole run. A later failure
   remains a failure. This checks local transition parity along the reference trajectory;
   it does not establish equivalence of independently evolving trajectories after a flip.
5. Population spike counts and the reference's `seed + 1` control are **descriptive information**,
   not acceptance gates or statistical tests (M4 owns statistics; Q21). Tested counts include
   the resynchronized trajectory. All requested steps still contribute to spike totals.

**Fused implementation (`engine_impl: fused`, PR-03b).** The per-neuron arithmetic of
section 2.1 is the same IEEE-754 operation sequence (Triton kernels launched with
`enable_fp_fusion=False`, so no multiply-add is contracted; propagator constants loaded in the
state dtype). Arrivals differ in one way: the fused kernels add the **integer** signed synapse
counts of the arriving edges per target (exact, order-independent) and multiply the total once
by `w_syn`, whereas the reference adds `w_syn × count` terms in float64 in edge order. When a
neuron receives one arrival in a step the two are bitwise equal; with several, they can differ by
float64 rounding of the same order as the reordered sums discussed above, and then by a possible
one-ulp change after rounding to the state dtype. The same margins and comparison policy apply.
The fused engine is deterministic regardless of `deterministic`, because integer addition does
not depend on order.

Measured results are in `docs/acceptance/M3.md` (eager) and `docs/acceptance/M3b.md` (fused); a
statement about CUDA counts as verified only with a log from the owner's RTX host.

## 3. Parameters

Preset `shiu2024_code_v1` (`flylab.neural.params`). Every value is stated in the **Methods** of
Shiu et al. 2024 (*Nature* 634:210–219, doi:10.1038/s41586-024-07763-9, CC BY 4.0; section
"Computational model", read from nature.com on 2026-11-04) **and** is identical in the reference
code, **github.com/philshiu/Drosophila_brain_model, commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
(2024-09-14), `model.py`** (MIT; values copied, no code copied). Verbatim quotes:
`docs/acceptance/evidence/open-questions-q8-q13-q14.md` (`OPEN_QUESTIONS.md` Q13, answered).

| Name | Value | Units | Label | Methods ("Computational model") | Code |
|---|---|---|---|---|---|
| `v_rest` | −52 | mV | LITERATURE_DERIVED | "Vresting = −52 mV (resting potential from ref. 18)" | `model.py:22` `'v_0' : -52 * mV` |
| `v_reset` | −52 | mV | LITERATURE_DERIVED | "Vreset = −52 mV (reset potential after spike)" | `:23` `'v_rst' : -52 * mV` |
| `v_threshold` | −45 | mV | LITERATURE_DERIVED | "Vthreshold = −45 mV (threshold for spiking)" | `:24` `'v_th' : -45 * mV` |
| `tau_m` | 20 | ms | LITERATURE_DERIVED | "Rmbr = 10 Kohm cm2", "Cmbr = 2 µF cm−2", "Tmbr = Cmb × Rmbr" → 20 ms | `:25` `'t_mbr' : 20 * ms` |
| `tau_syn` | 5 | ms | LITERATURE_DERIVED | "τ = 5 ms (synapse decay timescale)" | `:28` `'tau' : 5 * ms` |
| `t_ref` | 2.2 | ms | LITERATURE_DERIVED | "Trefractory = 2.2 ms (refractory period)" | `:31` `'t_rfc' : 2.2 * ms` |
| `delay` | 1.8 | ms | LITERATURE_DERIVED | "Tdly = 1.8 ms (time delay from spike to change in membrane potential)" | `:34` `'t_dly' : 1.8*ms` |
| `w_syn` | 0.275 | mV per synapse | **ASSUMED** | "Wsyn = 0.275 mV (free parameter …)"; "We chose Wsyn such that activation of sugar GRNs at 100 Hz resulted in roughly 80% of maximal MN9 firing" — chosen on FlyWire v630 (female) | `:37` `'w_syn' : .275 * mV` under `# Free parameter`. Not recalibrated for MaleCNS (Q16) |
| `w_syn` (MaleCNS, PR-04) | 0.270 | mV per synapse | **LEARNED** | calibrated for Shiu et al.'s criterion on MaleCNS v1.0 (definition D1, `docs/acceptance/M4.md` §1); weakly identified (tested 0.250, 0.255, 0.260, 0.270 and 0.275 mV compatible); used as a config override (`configs/m4/controls_wcal.yaml`); the preset keeps 0.275 mV | not a synaptic strength (Q16) |
| `dt` | per run, examples use 0.1 | ms | ENGINEERED | **not stated** ("The neural model is implemented in Brian2") | never set; Brian2 2.5.1 (repo `environment.yml`) defaults to `Clock(dt=0.1*ms)` (`brian2/devices/device.py:334`) |

Also from the Methods: "Upon firing, a neuron's membrane potential is reset to the resting
potential, and cannot change for the duration of the refractory time period" (the frozen state of
section 2.1). The Methods call the synapse "α-synapse dynamics" but print `dg/dt = −g/τ`, the
single-exponential form that the code and FlyLab implement. `w_syn` was chosen on "All 127,400
proofread neurons from Flywire materialization v.630"; measured from the reference repository,
that graph has 414.4 synapses per neuron, MaleCNS v1.0 `superclass_annotated` 744.9 (×1.80;
synapse definitions differ).

"LITERATURE_DERIVED" here means: adopted from a published model (Methods text and code) with
citation and version. It does **not** mean the value was measured for any MaleCNS neuron.

A config may override a parameter only by giving a full labelled value
(`{value, units, evidence, source | rationale}`); units must match, and `v_reset < v_threshold`,
`tau_m > 0`, `tau_syn > 0`, `t_ref ≥ 0`, `delay > 0`, `w_syn ≥ 0` are enforced.

### 3.1 How the Shiu code connects parameters (for comparison)

- `create_model` (`model.py:160–183`): `NeuronGroup(..., method='linear', threshold='v > v_th',
  reset='v = v_rst; w = 0; g = 0 * mV', refractory='rfc')`, `neu.v = v_0`, `neu.g = 0`,
  `Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly)`,
  `syn.w = Excitatory x Connectivity × w_syn`. FlyLab: same structure,
  `w = w_syn × sign × synapse_count`.
- Activation (Methods: "We stimulated particular neurons with Poisson distributed input";
  `poi`, `model.py:83–104`): `PoissonInput(target_var='v', weight=w_syn·f_poi)`
  (`0.275 mV × 250 = 68.75 mV` per Poisson event) and `rfc = 0` for the targets. FlyLab has no
  such mode; its forced spikes (section 5) keep refractoriness and bypass the threshold. The two
  are **not** equivalent.
- Silencing (`silence`, `model.py:108–127`): `syn.w[' {} == i'.format(i)] = 0*mV`, i.e. only the
  **outgoing** synapses of a silenced neuron (Brian2 `i` = presynaptic index). The README says
  "all synaptic connections to and from those neurons"; the code does not zero incoming weights,
  and the Methods agree with the code: "each of the top 200 firing neurons was silenced by
  eliminating all output of that neuron". FlyLab's `block_output` mode follows code and Methods.

### 3.2 Brian2 semantics the scheme reproduces (source-checked, Brian2 tag 2.5.1, `577bb35`)

- Default schedule `['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']`
  (`brian2/core/network.py:38–45`).
- Refractoriness: the state updater (slot `groups`) first runs
  `not_refractory = timestep(t - lastspike, dt) >= timestep(ref, dt)`
  (`brian2/groups/neurongroup.py:198–200`); the threshold is `({code}) and not_refractory`
  (`:349`); variables flagged `(unless refractory)` get a conditional write on `not_refractory`
  (`:851–859`), so synaptic `on_pre` writes to them are dropped while refractory; the resetter
  overrides that condition (`:373`).
- Measured with Brian2 2.10.1 (scratch probe; the repeatable comparison is
  `scripts/crosscheck_brian2.py`, section 8): constant drive 20 mV → first spike at step 86 and
  ISI 108 steps at `dt = 0.1 ms`, both equal to the closed form `k* − 1`, `R + k* − 1` with
  `k* = 87`, `R = 22`.

## 4. Transmitter → synapse sign

The sign is a property of the **presynaptic neuron** (one sign per neuron; co-transmission and
receptor diversity are ignored — ASSUMED). It is applied at run time to the in-memory graph;
`graph.npz` is not modified. The run config chooses:

- `synapse_sign.source`: `graph` (use the signs already in the graph — CSV/synthetic fixtures) or
  `transmitter` (map a `neurons.parquet` column).
- `transmitter_column`: required for `transmitter`; one of the PR-01 columns `predicted_nt`,
  `consensus_nt`, `celltype_predicted_nt`, `ground_truth` (or any column present in the table).
  **No default in the schema**; for MaleCNS the measured recommendation is `consensus_nt`, with
  `predicted_nt` as a sensitivity variant (section 4.1, Q14 answered).
- `mapping`: a named preset; `overrides`: per-transmitter changes (sensitivity analysis).
- `unknown_sign_policy`: required.

Preset `fast_ionotropic_v1` (every row ASSUMED):

| Value in column | Sign | Rationale |
|---|---|---|
| `acetylcholine` | +1 | main fast excitatory transmitter of the fly CNS (nicotinic receptors) |
| `gaba` | −1 | fast inhibition via GABA-A-type chloride channels |
| `glutamate` | −1 | in the fly CNS glutamate acts largely through the glutamate-gated chloride channel GluClα; it is **excitatory** at the neuromuscular junction and some central synapses → run the `glutamate = +1` sensitivity variant |
| `histamine` | −1 | photoreceptor output acts on histamine-gated chloride channels; few central histaminergic neurons |
| `dopamine`, `serotonin`, `octopamine` | 0 (unknown) | modulatory, mainly metabotropic; sign depends on receptor → resolved only by the unknown-sign policy |
| `unclear`, null / missing | 0 (unknown) | no usable prediction |
| any other value | — | **error**: unmapped categories are never guessed |

The rationale column is background knowledge, not a citation checked in this session; that is why
every row is ASSUMED and not LITERATURE_DERIVED.

Preset `shiu2024_methods_v1` — the rule stated in the Shiu et al. Methods ("Neurotransmitter
predictions"): "We assume GABAergic and glutamatergic neurons are inhibitory … if greater than
half of all the presynaptic sites across the entire neuron are predicted to be inhibitory (GABA or
Glut), we assigned this neuron as inhibitory. Neurons predicted to be dopaminergic, octopaminergic
or serotonergic are assigned to the excitatory category."

| Value in column | Sign | Label | Note |
|---|---|---|---|
| `acetylcholine` | +1 | LITERATURE_DERIVED | not GABA/Glut → excitatory |
| `gaba`, `glutamate` | −1 | LITERATURE_DERIVED | quoted rule |
| `dopamine`, `serotonin`, `octopamine` | +1 | LITERATURE_DERIVED | "assigned to the excitatory category" |
| `histamine` | 0 (unknown) | ASSUMED | not covered: the Methods list ACh 55%, Glu 24%, GABA 14%, DA/OA/5-HT 7% (sums to 100%) |
| `unclear`, null / missing | 0 (unknown) | ASSUMED | not covered |

The table as a whole is labelled ASSUMED (its weakest rows); manifests carry the per-row labels
(`row_evidence`), and overridden rows are reported as `ASSUMED (override)`. Difference from the
paper: Shiu et al. decided per neuron by a majority vote over presynaptic sites (cleft score cutoff
50); FlyLab maps one neuron-level column value. Cross-check against the authors' published v783
connectivity (`Connectivity_783.parquet`, one `Excitatory` value per presynaptic neuron, 0 of
138,005 mixed) joined with FlyWire `top_nt` (flywire_annotations `a83b277`): ACh 84,162 +1 /
1,719 −1; GABA 1,348 / 17,731; Glu 2,941 / 21,863; DA 5,897 / 3; 5-HT 2,127 / 9; OA 192 / 1. The
minority disagreements are expected because `top_nt` (presynapse-averaged confidence, current
annotation release) is not the per-site vote the paper used.

### 4.1 Which transmitter column (Q14, measured)

MaleCNS v1.0, node set `superclass_annotated` (166,700 neurons); script and full output in
`docs/acceptance/evidence/open-questions-q8-q13-q14.md`. Column definitions are not documented
in the files read here, so the "behaves like" column is an inference from the data.

| Column | `unclear` | Constant within cell type? | Agreement with `ground_truth` (where not unclear) | Behaves like |
|---|---|---|---|---|
| `predicted_nt` | 16,151 (9.7%) | no: varies within 654 of 11,751 types | 99.21% (n = 76,348) | per-body prediction |
| `celltype_predicted_nt` | 4,621 (2.8%) | yes, all types | 99.30% (n = 85,425) | type-level prediction |
| `consensus_nt` | 2,999 (1.8%) | yes, all types | 100% (n = 85,484); where no ground truth: = `celltype_predicted_nt` 92.0%, = `predicted_nt` 87.0% | ground truth where known, else a type-level prediction |
| `ground_truth` | — (81,216 null) | yes, its 3,318 types | — | curated labels for 51.3% of neurons |

178 node-set bodies have no row in the transmitter table (→ null → unknown). The columns differ
most for modulatory transmitters: dopamine 4,443 neurons in `predicted_nt` vs 392 in
`consensus_nt`, serotonin 465 vs 48. **Default for MaleCNS runs: `consensus_nt`** (most complete,
equals the curated labels where they exist); **sensitivity: `predicted_nt`**. Agreement with
`ground_truth` is not an accuracy estimate if the predictor was trained on those labels (unknown).
No confidence threshold is applied (`consensus_nt` has no confidence column).

Every run manifest records: source, column, preset table, overrides, final table, policy,
neurons per transmitter value, edges per resolved sign and the number of unknown-sign edges and
what happened to them.

**Sensitivity option.** `sensitivity:` in the config lists named variants that override
`synapse_sign` fields (e.g. `glutamate: +1`, `unknown_sign_policy: as_excitatory`). The runner
executes the main run and every variant with the same seed, stimuli and interventions, in sibling
run directories that reference each other (`run_group` in each manifest).

## 5. Stimuli (`stimuli:` in the config)

Two biologically different modes, never mixed under one label (spec 8.2):

| Kind | Semantics | Label |
|---|---|---|
| `injected_current` | adds `e = amplitude_mV` to every neuron of the population for steps with `onset ≤ t_n < offset`; overlapping stimuli add. Has no effect while a neuron is refractory or clamped (integration is frozen). Units: mV of `R·I_ext` | ENGINEERED; calibration to pA is UNKNOWN |
| `forced_spikes` | requests spikes, either at explicit `times_ms` (multiples of `dt`) or as a Poisson process `rate_hz`: per neuron and step, Bernoulli with `p = 1 − exp(−rate_hz · dt_ms / 1000)` (probability of ≥ 1 Poisson event in `dt`; at most one spike per step). Accepted forced spikes are real spikes: they reset, start refractoriness and propagate. Requests on refractory or clamped neurons are dropped and counted | ENGINEERED |

## 6. Interventions (`interventions:` in the config)

| Kind / mode | Membrane integration | Spikes | Outgoing propagation | Incoming input | Window |
|---|---|---|---|---|---|
| none (`interventions: []`) | — | — | — | — | matched baseline: same seed, initial conditions, stimuli (`flylab run --with-baseline` writes one next to the run) |
| `silence_neurons`, `mode: block_output` | **continues** | threshold, reset, refractoriness continue; spikes recorded with `propagated = false` | **blocked** for spikes emitted inside the window | unaffected | `onset_ms`/`offset_ms` |
| `silence_neurons`, `mode: clamp_to_rest` | **stops**: `v = V_rest`, `g = 0` every step | none; forced requests dropped | none (no spikes); spikes in flight at onset still arrive | **discarded** | `onset_ms`/`offset_ms` |
| `remove_synapses` | unaffected | unaffected | the selected directed edges (all synapses of each selected pair) are deleted before the run | — | whole run only |

`remove_synapses` selects edges with `pre_population` and/or `post_population` (at least one),
then optionally a reproducible random subset: `k = floor(fraction × n_candidates)` edges drawn
without replacement by NumPy `default_rng(seed)`; `seed` is required when `fraction < 1`. It is
**not** equivalent to killing a neuron. Every removed edge is listed in `interventions.parquet`.

`block_output` for a whole run gives the rest of the network exactly the same input as removing
every outgoing edge of the population (a test checks this); it differs in that the silenced
neurons' own spikes stay visible.

## 7. Outputs and units

Every Parquet file carries `flylab.schema_version` and per-column units in its key-value
metadata; JSON files carry `schema_version`. Times are ms (`t_ms`) or integer steps (`step`),
voltages mV, rates Hz. Neuron indices follow `flylab.data.graph.NeuronIndex` (body ids sorted
ascending). See `flylab.experiments.artifacts` for the exact columns.

## 8. Verification status

| Check | Status | Evidence |
|---|---|---|
| Closed-form single-neuron and single-synapse responses, delay, refractory, reset, forced/clamp/block semantics | PASS (CI-able) | `tests/unit/test_lif_numpy.py` |
| NumPy ↔ torch CPU equivalence, float64 and float32 | PASS where torch is installed: bitwise-identical spikes and states on the tested inputs; tolerances justified in the test docstring | `tests/unit/test_lif_torch.py`, `.github/workflows/neural-torch-cpu.yml` |
| FlyLab NumPy ↔ Brian2 2.10.1 running the Shiu et al. equations (recurrent synthetic graphs, time-varying drive) | PASS on two configurations: identical spike trains (5,307 and 28,851 spikes), final \|Δv\| ≤ 2e-13 mV | `scripts/crosscheck_brian2.py`, `docs/acceptance/evidence/crosscheck-brian2-*.json` |
| Intervention semantics (block_output ≡ removing outgoing edges; clamp vs block; matched baseline; reproducible synapse removal) | PASS | `tests/scientific/test_intervention_semantics.py` |
| Any comparison with biological recordings | **not done** | — |
| MaleCNS run | **exploratory only**: left LB3c → MN9 with sign variants, a rate sweep and 5 seeds at 100/150 Hz, cloud CPU, `w_syn` not recalibrated (Q16) | `docs/acceptance/evidence/open-questions-q8-q13-q14.md` |
| MaleCNS replicated controls and `w_syn` calibration (PR-04) | see `docs/acceptance/M4.md` | `docs/acceptance/evidence/m4-*.json` |
| Replicate statistics, null graph, calibration arithmetic | PASS (CI-able) | `tests/unit/test_stats.py`, `tests/unit/test_null_models.py`, `tests/unit/test_calibration.py`, `tests/integration/test_replicates.py` |
| NumPy ↔ torch CUDA (RTX 5070, `sm_120`), synthetic graphs and the full MaleCNS graph | see `docs/acceptance/M3.md` (policy: section 2.3) | `tests/gpu/test_cuda_engine.py`, `flylab parity` records in `docs/acceptance/evidence/` |

## 9. Controls and replicate statistics (PR-04)

### 9.1 Degree-preserving null graph (`graph_null`)

`flylab.data.null_models.degree_preserving_swap` (ENGINEERED): every edge keeps its presynaptic
neuron, synapse count and sign; the postsynaptic ends are permuted among all edges, then every
self-loop or repeated (pre, post) pair is swapped with a random other edge until none is left
(MaleCNS: 5 repair rounds, 92,857 swaps for seed 1). Exactly preserved: in-degree, out-degree,
out-strength (synapses per presynaptic neuron) and the sign of every neuron's output. Not
preserved: in-strength (Pearson r 0.80 with the original on MaleCNS, seed 1), pair identity
(0.59 % of pairs kept) and the 101 MaleCNS self-loops (none in the null). It is not proven to
sample uniformly from all graphs with these degrees. Signs are assigned per presynaptic neuron
at run time, so the null keeps the same sign table and per-neuron output sign.

### 9.2 Replicates and statistics

Unit of replication: one stimulus seed (Poisson input realisation, and for `graph_null` one
shuffle) on one fixed connectome and parameter set — model input noise, not animals. Paired
comparisons use the same seed in condition and reference (identical Poisson draws for identical
stimuli). Methods (`flylab.experiments.stats`, ENGINEERED): BCa bootstrap interval of the mean
(10,000 resamples; ties counted half as in `scipy.stats.bootstrap`; percentile fallback when
resamples are not finite); Cohen's d_z with a percentile interval (resamples whose differences are
all equal are excluded and counted, which biases the interval for very small n); relative change
with a paired percentile interval; two-sided sign-flip permutation test (exact up to 16 pairs:
smallest attainable p = 2/2^n); Holm step-down adjustment over the family stated in each
`group_stats.json`. Simulated values carry the weakest label of their inputs (ASSUMED as long as
`w_syn` or the sign table is ASSUMED).

### 9.3 `w_syn` calibration

`flylab.experiments.calibration` applies the criterion of Shiu et al. ("activation of sugar GRNs
at 100 Hz resulted in roughly 80% of maximal MN9 firing") with an explicit, ENGINEERED definition
of "maximal MN9 firing" (D1: the largest seed-mean MN9 rate over the swept input rates
10-200 Hz at the same `w_syn`; D2, D3 alternatives) and reports the selected `w_syn` as LEARNED:
calibrated for this criterion on this graph, not a synaptic strength. Results and caveats:
`docs/acceptance/M4.md`.
