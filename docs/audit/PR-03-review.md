# Independent review — PR-03 CUDA backend (`pr-03-cuda`, PR #16)

Spec section 13.5 review, adversarial. Date: 2026-11-04. Reviewer: an independent Claude Code
cloud session that did not write the code under review. It changed no production file: the
deliverables are this report, strict-xfail tests in `tests/audit/test_audit_pr03_*.py`, and one
reproduction script in `docs/audit/repro/`.

| Item | Value |
|---|---|
| Reviewed revision | `pr-03-cuda` @ `b7f646b`, diff against `pr-02-cpu-lif` @ `36c2dd6` (`git diff origin/pr-02-cpu-lif...origin/pr-03-cuda`, 46 files) |
| Method | read every changed file; ran FlyLab's own engines, parity and bench code on synthetic graphs (NumPy and torch-CPU); ran the host scripts against a fake `ssh`; recomputed every M3 number from `docs/acceptance/evidence/`; one strict-xfail test per finding |
| Not possible here | this session has no GPU and no access to the owner's host. CUDA-only behaviour (`index_add_` order, allocator, profiler output) was reviewed by reading the code, the torch 2.15.0 sources installed here, and the committed host evidence |
| Root checks on the reviewed tip | `ruff check`, `ruff format --check`, `mypy` (strict), `pytest`: all pass (288 passed, 7 skipped without torch; 304 passed, 10 skipped with the `gpu` extra, torch 2.15.0 CPU) |
| Findings | 9: **0 High · 4 Medium · 5 Low**. M3's measured results stand (see [M3 conclusions](#do-the-m3-conclusions-stand)); the findings are about what the gates would let through next time and about three reported figures |

Severity scale (same as `docs/audit/AUDIT_2026-11.md`):
- **High**: wrong science or a security exposure is likely now.
- **Medium**: can silently produce wrong results, provenance or host-side effects, or leak data, under plausible conditions.
- **Low**: hardening, accuracy of a reported figure, or a missing check.

Status legend: **PASS** checked, nothing material · **FAIL** a finding below · **UNVERIFIED** could not be checked from here (reason given).

## Summary by area

| Area | Status | Finding |
|---|---|---|
| Parity policy after a justified flip (spec 7.4 "population/statistical equivalence") | **FAIL** (Medium) | [1](#finding-1--after-the-first-justified-flip-nothing-is-compared-and-the-population-fallback-is-not-a-gate) |
| Parity: what justifies a flip | **FAIL** (Medium) | [2](#finding-2--a-flip-is-justified-by-the-reference-alone-states-and-flags-at-that-step-are-not-checked) |
| Parity: non-finite states and margins | **FAIL** (Medium) | [3](#finding-3--a-nan-state-passes-parity-as-identical-with-max-δv--00) |
| `fetch_artifacts.sh` pattern filter | **FAIL** (Medium) | [4](#finding-4--fetch_artifactssh-passes-option-like-patterns-to-the-hosts-tar) |
| Parity: which state is compared | **FAIL** (Low) | [5](#finding-5--the-state-check-enforces-v-only-g-differences-never-fail) |
| `MODEL_ASSUMPTIONS.md` §2.3 margins and derivation | **FAIL** (Low) | [6](#finding-6--the-margin-derivation-in-model_assumptions-23-is-incomplete-and-not-tied-to-the-run) |
| Profiling method (`scripts/profile_step.py`) and the M3/Q22 profile figures | **FAIL** (Low) | [7](#finding-7--profile_steppy-counts-gpu-time-twice-m3q22-say-322-µs-per-step) |
| VRAM accounting (spec 7.3 "and any difference") | **FAIL** (Low) | [8](#finding-8--vram-is-reported-from-torchs-allocator-only-about-200-mib-per-run-is-not-counted) |
| Redaction of remote logs | **FAIL** (Low) | [9](#finding-9--mac-side-log-redaction-does-not-know-the-hosts-account-and-distro-names) |
| CUDA engine: device handling, chunked arrivals, retained-memory accounting | **PASS** (by reading + CPU tests) | — |
| Benchmark timing method (`flylab bench`) | **PASS** | — |
| `provenance/hardware.py`, committed evidence: host anonymity | **PASS** | — |
| `compose.yaml`, `run_local_gpu_bench.sh`, `run_gpu_bench.sh` quoting | **PASS** | — |
| M3 numbers against the evidence records | **PASS**: every number matches a record field; two of those fields are computed or labelled wrongly (findings 7, 8) | [table](#m3-numbers-against-the-evidence) |

The original review supplied `xfail(strict=True)` tests for the findings below. The fixes
remove those markers and retain the tests and finding numbers as passing regressions:

| Finding | Tests |
|---|---|
| 1 | `tests/audit/test_audit_pr03_parity.py::test_finding_1_nothing_is_checked_after_a_justified_flip` |
| 2 | `…::test_finding_2_flip_justified_although_the_test_engine_is_far_off`, `…::test_finding_2_flag_difference_hidden_by_a_justified_flip` |
| 3 | `…::test_finding_3_nan_state_passes_as_identical`, `…::test_finding_3_nan_margin_disables_the_state_check` |
| 4 | `tests/audit/test_audit_pr03_scripts.py::test_finding_4_fetch_refuses_option_like_patterns` |
| 5 | `tests/audit/test_audit_pr03_parity.py::test_finding_5_g_only_difference_passes` |
| 6 | `…::test_finding_6_default_margin_covers_the_documented_bound[0.1]`, `[0.01]` |
| 7 | `tests/audit/test_audit_pr03_bench.py::test_finding_7_profile_counts_device_time_once` (needs torch; skipped without it) |
| 8 | `…::test_finding_8_bench_reports_the_device_wide_increase` |
| 9 | `tests/audit/test_audit_pr03_scripts.py::test_finding_9_remote_log_keeps_the_wsl_account_name` |

All 12 were run with `--runxfail` to confirm that each fails at its final assertion (the claim),
not in its set-up. The parity tests use NumPy engines with one deliberate bug as the engine
under test, so they run in CI without torch or a GPU. To check that they are passable, a
stricter `lockstep` prototype (not committed) was run against the six parity scenarios of
findings 1, 2, 3 and 5: all six pass with it.

---

## Finding 1 — after the first justified flip nothing is compared, and the population fallback is not a gate

**Resolution.** Continued spike/flag comparison after every accepted flip; public `LIFEngineBase.set_state` resynchronizes the full reference state, with `steps_compared`, `justified_flips` and `resyncs` recorded. Population counts are explicitly descriptive.

**Severity: Medium.** This is the most important finding: Q22 says the CUDA performance PR will be
"judged against the same parity tests".

**What.** `lockstep` (`src/flylab/neural/parity.py:165-178`) compares spikes and states only while
`rep.divergence is None`. After the first near-threshold divergence both engines keep running,
but no spike, flag or state is compared again, and `ParityReport.ok` is already final:
`near_threshold_divergence` counts as ok (`parity.py:86-95`). `run_parity` sets
`"status": "PASS" if rep.ok` (`src/flylab/experiments/parity.py:109`); the per-population
`within_seed_to_seed_diff` (`parity.py:100`) is written to the record and never read.

The docs say otherwise. `MODEL_ASSUMPTIONS.md:136-140` (policy step 4): "spike-by-spike comparison
stops and **population spike counts are compared** instead"; the `parity.py` module docstring:
"only population spike counts are compared from then on". They are summed, not compared. Spec 7.4
("Full network: perform population/statistical equivalence") is therefore not implemented as a
check. Q21 defers a *statistical threshold* to M4, but the PR's policy text presents a comparison
that does not exist.

The seed-to-seed yardstick cannot serve as a tolerance for workloads without random input
either: B0 drives only `injected_current`, so its `seed + 1` run is identical and the yardstick
is 0.

**Evidence.**
- `test_finding_1_…`: 300-neuron synthetic graph, float64, default margin. The engine under test
  has a threshold 5e-7 mV higher (below δ = 1e-6, so the step-0 flip is justified by the policy)
  and, from step 1 on, **delivers no synaptic arrivals at all**. Result: `near_threshold_divergence`,
  `ok=True`; population `all`: 776 spikes in the reference, 259 in the engine under test.
- `docs/audit/repro/pr03_parity_experiments.py`, experiment A: 40 synthetic graphs (seeds 0–39,
  300 neurons), float32, default margins, 300 steps. An engine whose `V_th` is 0.2 mV too high
  (20 × δ) gets `ok=True` in **18 of 40** runs; with 0.05 mV, 18/40; with 0.02 mV, 23/40. In every
  passing run the first flip is at steps 132–139, so more than half of each run is never compared.
  float32 is the dtype of every M3 parity record. In float64 (δ = 1e-6) the same experiment gives
  no near-threshold passes: errors from 2e-6 to 1e-4 mV are either caught (`mismatch`) or have no
  visible effect in 300 steps (`identical`).

**Why it passes.** The first threshold crossings of a run overshoot `V_th` by less than
δ = 0.01 mV often enough that the first flip of a wrong engine looks like rounding. One flip
cannot tell a 0.2 mV threshold error from a rounding difference; only continued comparison can.

**Suggested fix.**
1. After a justified flip, do not stop. Either load the reference state into the engine under
   test (`_load_state(reference.state())`, both are host-loadable) and continue spike-by-spike,
   counting justified flips; or keep a per-step population comparison with an explicit,
   documented tolerance that makes `status` FAIL.
2. Make the fallback a gate, or rename it: if Q21 is deferred, `run_parity` should report
   `status: INCONCLUSIVE` (not PASS) after a divergence, and the docs should say "reported", not
   "compared".
3. Record the number of justified flips and the steps actually compared.

## Finding 2 — a flip is justified by the reference alone; states and flags at that step are not checked

**Resolution.** Checks pre-flip v/g and discrete state, both backends' actual retained phase-2 candidates, activity, and common-spike flags. Non-threshold discretization and population mismatches are rejected; diagnostic threshold shifts must satisfy both threshold windows.

**Severity: Medium.**

**What.** `_compare_step` (`parity.py:196-213`) labels a spike-set difference `near_threshold` if
the **reference's** phase-2 candidate potential is within δ of `V_th`. The justification assumes
that the two engines' potentials agree within δ at that step, but:
- the states are compared only every `state_every` steps (default 100) and never at the
  divergence step, so the engine under test may have drifted by any amount; when the flip comes
  before step `state_every`, no state was compared at all (`state_checks = 0`);
- the engine under test's own candidate potential is never looked at;
- when the spike sets differ, the forced/propagated flags of the spikes both engines share are
  not compared (`kind="flags"` is only reached when the spike sets are equal), contrary to
  `MODEL_ASSUMPTIONS.md:135` "Different forced/propagated flags never pass".

**Evidence.**
- `test_finding_2_flip_justified_…`: an engine that applies the injected drive **twice**. The
  reference candidate is `V_th − 1e-7 mV`; the doubled drive adds 0.05 mV (50,000 × δ) and the
  engine fires. Result: `near_threshold_divergence` at step 0, `state_checks = 0`, `ok=True`.
- `test_finding_2_flag_difference_…`: a justified flip (as in finding 1) in the same step as a
  forced, output-blocked spike that the engine under test propagates (it ignores blocking).
  Result: `near_threshold_divergence`, `ok=True`.
- Repro experiment B: float32, drive U(14, 24) mV so that the first crossings (steps 66–70) come
  before the first state check. An engine that scales the drive by 1.002 passes **28/40**, by
  1.01 passes 2/40, always with `state_checks = 0` and reported `max_abs_v_diff_mV = 0.0`. With
  the weaker drive of experiment A the same bugs are caught as `state_mismatch` at step 100, so
  the outcome depends on whether a state check happens to precede the first flip.

**Suggested fix.** At every step where the spike sets differ: compare the two engines' states
before the step (|Δv| and |Δg| ≤ δ, finite); require **both** engines' candidate potentials of
the differing neurons to be within δ of their thresholds; and compare the flags of the common
spikes. Also check that both engines have the same `disc` (today only the network fingerprint and
dtype are checked, so an engine with a different threshold or refractory period is accepted).

## Finding 3 — a NaN state passes parity as "identical" with max |Δv| = 0.0

**Resolution.** Rejects non-finite/non-positive margins at the API and CLI boundary; checks state finiteness at every step, reports `non_finite`, and serializes undefined state differences as null.

**Severity: Medium.** NaN from uninitialised or mis-indexed device memory is a typical GPU
failure, and the parity record is the evidence M3 cites for correctness.

**What.** `check_states` (`parity.py:155-161`) fails only if `dv > delta`. With a NaN in either
state, `np.max(np.abs(...))` is NaN, `NaN > delta` is False, and `max(rep.max_abs_v_diff_mV, dv)`
returns the old value (Python's `max` keeps the first argument when the comparison is False). A
NaN neuron never crosses threshold, so unless the reference neuron spikes, spikes stay equal.
The same comparison makes `--margin-mv nan` (or `margin_mV=float("nan")`) disable the state check
entirely; the margin is not validated (`parity.py:145`, `cli/main.py` `--margin-mv type=float`).

**Evidence.** `test_finding_3_nan_state_…`: one neuron's v becomes NaN at step 5 of 40; result
`identical`, `ok=True`, `max_abs_v_diff_mV = 0.0`, 4 state checks. `test_finding_3_nan_margin_…`:
a 1 mV state error, which the default margin reports as `state_mismatch`, passes as `identical`
with `margin_mV=nan`.

`flylab bench` does check finiteness (`gates.all_runs_finite`); `flylab parity`,
`tests/gpu/test_cuda_engine.py` and `scripts/gpu_smoke.py` rely on `lockstep` alone.

**Suggested fix.** Treat any non-finite value in either state as a failure
(`status: non_finite`), compute the maxima with NaN propagation, and refuse a margin that is not
finite and positive.

## Finding 4 — `fetch_artifacts.sh` passes option-like patterns to the host's `tar`

**Resolution.** Rejects leading option/dot patterns and dot-prefixed components, retains the character whitelist, and uses `tar --` plus explicit dot-glob safeguards.

**Severity: Medium.** The script states "only A-Z a-z 0-9 . _ - / * are allowed, and no absolute
paths or '..'. Read-only on the host." (`scripts/remote/fetch_artifacts.sh:6-7`). FlyLab is a
guest on that host (`CLAUDE.md`), and the filter is meant to make any pattern safe, including
one composed by a later Claude session.

**What.** The filter (`fetch_artifacts.sh:19-20`) refuses a leading `/`, `..` and other
characters, but not a leading `-`. Patterns are then expanded unquoted after `tar -cf -`
(`:29`), so a "pattern" that starts with `-` is a GNU tar option on the host.

**Evidence.**
- `test_finding_4_…` runs the real script with a fake `ssh` that decodes the
  `-EncodedCommand` payload and runs the WSL script locally. `fetch_artifacts.sh fakehost
  --remove-files 'bench/*.json'` passes the filter, copies the file and **deletes it on the
  "host"** (exit 0).
- By GNU tar's documented option syntax (not executed here): options that take a program as
  the next word (`-I PROG` / `--use-compress-program PROG`, `-F PROG` with `-M`) can be built
  from two allowed "patterns", and `-h` makes tar follow symlinks out of `artifacts/`.
- The `.*` pattern contains no `..`, but on bash < 5.2 (e.g. Ubuntu 22.04's bash 5.1) it expands
  to `.` and `..` as well, i.e. the parent of `artifacts/`. Verified locally:
  `shopt -u globskipdots; echo .*` prints `. .. .hidden`. The host runs Ubuntu 24.04.5
  (`docs/ENVIRONMENT_AUDIT.md`), whose bash 5.2 enables `globskipdots` by default, so this part
  is not reachable there today.

**Suggested fix.** Refuse patterns that start with `-` and patterns with a path component that
starts with `.`; pass `--` before the patterns (`tar -cf - -- ${REST[*]}`); set
`shopt -s globskipdots` (or `GLOBIGNORE`) in the host script instead of relying on the default.

## Finding 5 — the state check enforces v only; g differences never fail

**Resolution.** Enforces the margin on g as well as v; compares initial, periodic, pre-flip and final discrete spike history/in-flight slots exactly.

**Severity: Low.**

**What.** `check_states` records `max_abs_g_diff_mV` but sets `state_within_margin = False` only
for v (`parity.py:155-161`; documented as "any |Δv| > δ fails", `MODEL_ASSUMPTIONS.md:132`).
`last_spike` and the in-flight spikes are not compared either; when every step's spikes and flags
agree from a common initial state these two are implied equal, so the practical gap is g (and
the initial state, which `lockstep` does not compare). Spec 7.4 asks to "compare floating-point
states", and a wrong g survives into checkpoints and into every later step once the neuron is
active again.

**Evidence.** `test_finding_5_…`: an engine whose clamped neurons keep their synaptic drive
(phase 5's `g ← 0` missing). Neurons 0–19 are clamped from step 25 to the end after receiving
arrivals: v is identical, g differs by up to 53.7 mV, result `identical`, `ok=True`.

**Suggested fix.** Fail on |Δg| > δ_g (with its own justified bound) and compare `last_spike` and
the in-flight slots exactly at every state check and at the end.

## Finding 6 — the margin derivation in MODEL_ASSUMPTIONS 2.3 is incomplete and not tied to the run

**Resolution.** Rewrites MODEL_ASSUMPTIONS 2.3 with the two-sum factor, dt-dependent recurrence gains and explicit illustrative/engineering limits. Float64 baseline is 2e-6 mV and defaults scale upward with recurrence gain; float32 remains an empirically checked engineering tolerance.

**Severity: Low.** This answers the review question "are the error margins and their derivation
sound?": **the float64 margin is the right order of magnitude for the Shiu preset at
dt = 0.1 ms, but the derivation does not support it as a bound; the float32 argument is
incomplete; and both margins are constants applied to every run.** Measured differences are far
below both margins (max |Δv| = 1.4e-14 mV in float64 non-deterministic mode, 0 elsewhere), so no
false failure has occurred; the risk is in what the margin lets through as a near-threshold
window (findings 1, 2).

**What, point by point** (`docs/MODEL_ASSUMPTIONS.md:119-128`, `parity.py:26-27`):

1. **Factor 2.** "(m − 1)·2⁻⁵³·Σ|w|" bounds one sum's error against the exact sum. The CPU and GPU
   sums both err, so |S_cpu − S_gpu| ≤ 2·(m − 1)·2⁻⁵³·Σ|w|. With the doc's own example
   (m = Σ|w| = 10⁴ mV) and the doc's gain, that is **1.11e-6 mV > δ = 1e-6** at dt = 0.1 ms.
2. **Constant δ, run-dependent bound.** The gain 1/(1 − d) = 50.5 is for dt = 0.1 ms and
   τ_s = 5 ms. `lockstep` uses `DEFAULT_MARGIN_MV[dtype]` for every discretization; at
   dt = 0.01 ms (accepted by `discretize` for the preset) the same formula gives 1.11e-5 mV,
   11 × δ.
3. **The example is not a bound for the data.** m = 10⁴ arrivals of Σ|w| = 10⁴ mV per step is
   illustrative. Nothing derives it from the graph or the workload (B1 records a maximum
   *out*-degree of 11,203; in-degree and in-strength are not reported).
4. **"Gain ≤ ≈ 50" mixes two quantities.** 50 = b/((1 − a)(1 − d)) is the gain for a same-sign
   perturbation of g in *every* step (the accumulated worst case, which is the right quantity
   here). One perturbation reaches v with a peak gain of 0.16 (computed from the preset's
   propagator). The text says "a g perturbation … per perturbation".
5. **float32 omits fresh rounding.** Once v or g differ by one ulp, later float32 operations act
   on different inputs and round differently; those differences accumulate in v with the gain
   1/(1 − a) = 200.5 at dt = 0.1 ms, not 50. A rough worst case (about 4 roundings of
   ulp(50 mV) = 3.8e-6 mV per step × 200) is ≈ 3e-3 mV: still below 1e-2, but with a factor of
   about 3, not 10. "|g| < 128 mV" is assumed, not checked.
6. **δ doubles as the near-threshold window.** That is valid only if |Δv| ≤ δ holds at the flip,
   which is not checked (finding 2). In float32, a window of 0.01 mV is large compared with how
   far first threshold crossings overshoot (finding 1, experiment A).
7. "The only operation whose result may depend on the device is the float64 sum of arrivals"
   is not quite true: the injected drive is also summed with `index_add_`
   (`lif_torch.py:305`), and `ExperimentSchedule.inputs` concatenates overlapping
   `injected_current` stimuli without merging them (`schedule.py:106`), so three or more
   stimuli on one neuron are also summed in device order when `deterministic: false`. The
   comment at `parity.py:26` cites "MODEL_ASSUMPTIONS 2.4"; the section is 2.3.

**Evidence.** `test_finding_6_…[0.1]` and `[0.01]` evaluate the doc's formula with the doc's
example, times 2, for the run's discretization: 1.11e-6 and 1.11e-5 mV against the 1e-6 mV the
policy uses. Constants: `discretize(SHIU2024_CODE_V1, 0.1)` gives a = 0.99501248,
b = 0.00493794, d = 0.98019867.

**Suggested fix.** Compute δ per run from the discretization and a measured bound on per-step
arrivals (max in-strength of the network, or Σ|w| over the step's arrivals, which the engine can
compute), with the factor 2. Use a separate, smaller near-threshold window derived the same way.
For float32, derive the bound including per-operation rounding, or measure it with a
perturbation test.

## Finding 7 — `profile_step.py` counts GPU time twice; M3/Q22 say "322 µs per step"

**Resolution.** Profiler totals use CUDA device rows once and separate HtoD/DtoH events. M3/Q22 mark 322 us and the 8 DtoH label as invalid pending host re-profile; committed evidence is unchanged.

**Severity: Low.** The conclusion (launch/sync-bound) is unchanged and in fact stronger.

**What.** `scripts/profile_step.py:75,80` sums `self_device_time_total` over every row of
`prof.key_averages()`. In torch's kineto profiler a CPU op's `self_device_time_total` is the
time of the kernels it launched, and each kernel is also its own row. torch's own table sums
only device-type rows for this reason (torch 2.15.0, `torch/autograd/profiler_util.py`,
`_build_table`: "in kineto profiler, there're events with the correct device type").

**Evidence.**
- The committed record `profile-B1-20261104T101336Z.json` contains the pairs, with identical
  times and call counts: `aten::bitwise_and` 6.83 µs/step and its `BinaryFunctor<bool…>` kernel
  6.83 µs/step (6 calls each); `aten::bitwise_not` / `bitwise_not_kernel_cuda` 5.42;
  `aten::_assert_async` / `_assert_async_cuda_kernel` 9.78.
- `device_total_us` / 300 steps = 322.4 µs. If every device event was launched under some CPU op,
  the GPU time is half of that, ≈ 161 µs per step. That is an estimate: the record keeps only the
  top 25 rows by each key, so the exact figure cannot be recomputed from it.
- `test_finding_7_…` feeds `main()` one op row and its kernel row (6.83 µs each):
  `device_total_us` comes out 13.66.
- M3.md:69 "8 device→host copies": the profile counts `cudaMemcpyAsync` calls (8 per step)
  without direction. By reading `lif_torch.py`, every B1 step also copies the forced-spike index
  list host→device (`_mask(forced_idx)` → `torch.tensor(…, device=cuda)`), and B1's Poisson
  input (p = 0.01 per step over 1,428 neurons) is empty with probability 0.99¹⁴²⁸ ≈ 6e-7, so at
  most 7 of the 8 are device→host.

**Suggested fix.** Sum device time only over rows whose `device_type` is not CPU; record the
memcpy direction (the device-side `Memcpy HtoD/DtoH` rows); correct M3.md and Q22 after a re-run.

## Finding 8 — VRAM is reported from torch's allocator only; about 200 MiB per run is not counted

**Resolution.** Bench records free/total/used device snapshots before init, after init and at end, device-wide deltas and allocator difference; adds an observed-delta memory gate. Protocol/M3/Q20 document shared-device attribution and missing-peak caveats; host remeasurement remains pending.

**Severity: Low.** The 10 GiB gate outcome does not change (≈ 0.5 GiB either way).

**What.** Spec 7.3 sets the target on "peak total device memory attributable to FlyLab" and asks
to report the allocator peaks, the nvidia-smi process figure "and any difference". The gate
(`benchmark.py:261-268`) and every reported VRAM figure use `torch.cuda.max_memory_reserved()`,
which excludes the CUDA context, loaded kernel images and library workspaces. The per-process
nvidia-smi figure is N/A under WSL2 (Q20), but the records do contain the device-wide
`memory.used` before the run and while FlyLab is loaded, and their difference is not computed or
reported anywhere.

**Evidence** (all from the committed records):

| Record | `memory.used` before → while loaded | increase | torch peak reserved | not in the allocator |
|---|---|---|---|---|
| `bench-B0-20261104T100506Z.json` | 2,122 → 2,345 MiB | 223 MiB | 22 MiB | 201 MiB |
| `bench-B1-20261104T100648Z.json` | 2,122 → 2,633 MiB | 511 MiB | 310 MiB | 201 MiB |
| `bench-B2-20261104T101318Z.json` | 2,122 → 2,633 MiB | 511 MiB | 310 MiB | 201 MiB |
| `bench-B1-numpy-20261104T101323Z.json` | 2,122 → 2,122 MiB | 0 | — | — |

`memory.used` is device-wide and includes other processes, so the increase is an estimate. The
same ≈ 201 MiB in every CUDA run and 0 in the NumPy run are consistent with a fixed per-process
CUDA overhead; that attribution is an inference, not a measurement. M3.md:13 ("peak 310 MiB
reserved"), M3.md "the process figure is the torch allocator's" and README.md:7 ("within ~0.3 GiB
of VRAM") understate FlyLab's footprint as the records show it (≈ 0.5 GiB on the full graph).
`test_finding_8_…` runs `flylab bench` with an engine reporting a 9.5 GiB allocator peak and an
nvidia-smi that sees +10,800 MiB: the record passes the VRAM gate and contains neither figure
of the difference.

**Suggested fix.** Record `while_loaded − before` (and `torch.cuda.mem_get_info()` before init
and after the last run, which is per-device but not per-process), report the non-allocator part,
and state in M3/README that the 310 MiB is the allocator figure. Decide explicitly whether the
gate should use the larger figure.

## Finding 9 — Mac-side log redaction does not know the host's account and distro names

**Resolution.** Shared Mac-side redaction preserves existing literals and adds WSL distro/user plus locally resolved `ssh -G` HostName/User without printing them; every remote script uses the list.

**Severity: Low.** For this owner the WSL user and distro are `homelab`, which the repository
already names (`CLAUDE.md`; see the earlier audit's note), so nothing secret leaked.

**What.** `run_gpu_bench.sh:26` redacts on the Mac with
`FLYLAB_REDACT="${TARGET#*@},${TARGET%@*}"`, i.e. the SSH alias only. Host output is redacted on
the host by `run_local_gpu_bench.sh`, but lines printed before or outside that script (the WSL
shell's own errors, `wsl.exe`, PowerShell) reach the Mac with the host's names intact.
`FLYLAB_WSL_USER`, `FLYLAB_WSL_DISTRO` and the `HostName` the SSH config resolves are not added.
The same pattern is in `compose_run.sh:36` (whose output gets no host-side redaction at all and
which M3 uses for `parity-B2`), `run_gpu_smoke.sh:25` and `windows_audit.sh:26`; those files are
outside this PR's diff.

**Evidence.** `test_finding_9_…`: with `FLYLAB_WSL_USER=alice-wsl`, a host that answers
`bash: line 1: cd: /home/alice-wsl/flylab: No such file or directory` (what happens when the code
was not synced yet) leaves `alice-wsl` in `artifacts/gpu/remote-bench-*.log`.

**Suggested fix.** In `common.sh`, build the redaction list once from the alias, `$FLYLAB_WSL_USER`,
`$FLYLAB_WSL_DISTRO` and `ssh -G "$TARGET"`'s `hostname`/`user`, and use it in every remote script.

---

## Checked with no finding

**CUDA engine (`lif_torch.py`), by reading and with the CPU tests.**
- Device handling: `cuda` resolves to the current device, `cuda:N` is range-checked, other
  device types are refused, and a missing device raises `CudaUnavailableError` before any data is
  loaded (`load_workload`, `_process_globals`, `run_experiment`).
- Phase order, masks and resets mirror `lif_numpy.py` operation by operation. Every
  elementwise step is its own kernel, so there is no cross-operation FMA contraction. The host
  record shows bitwise-equal float32 states on the full graph.
- Chunked arrivals: rows are grouped by the chunk window of their first edge, so a group
  holds at most `max_edges_per_chunk` + max out-degree − 1 edges; the CPU tests check bitwise
  equality for chunk sizes 1, 7 and 1000, and the host log shows the CUDA chunk test passing.
- Retained-memory accounting matches the arrays exactly: 12·E + 8·(N + 1) = 296,043,228 bytes for
  B1, as recorded.
- The "≈ 40 bytes/edge" estimate for chunk temporaries is not measured. On CUDA with
  `deterministic: true`, `index_add_` sorts its indices (radix-sort kernels in the profile), which
  needs extra buffers; with the default chunk of 4,194,304 edges the bound stays at a few hundred
  MB. B1/B2 never reached it: the peak allocation was 7.6 MiB above the retained arrays.
- `torch.use_deterministic_algorithms(True)`: torch 2.15.0's documentation lists
  `torch.index_add` on CUDA among the operations that act deterministically, which matches the
  profile's sort-based kernels. An operation that refused would raise rather than be silently
  disabled (spec 7.4).

**Timing method (`flylab bench`).** The device is synchronized before both clock reads, warm-up
runs are untimed, and median, quartiles, min and max are reported. Peak counters are reset per
run. The timed region includes the schedule's Poisson draws and the counting recorder, as
documented (0.034 ms of 6.7 ms per profiled step). The NumPy comparison uses 1 + 3 runs against
3 + 10 for CUDA, which M3 states.

**Provenance and anonymity.** `hardware.py` records no host name, serial, UUID or PCI id. The
committed evidence was scanned for paths under `/home` or `/Users`, IPv4/IPv6 addresses, e-mail
addresses and the account names: none found. The only dotted quad is the WSL kernel version
6.18.33.2. Container names in the log are random.

**Scripts and containers.**
- `run_local_gpu_bench.sh` records each step's status, keeps going, and redacts its whole
  output.
- `run_gpu_bench.sh` and `compose.yaml` keep PR-00's hardening: no network, all capabilities
  dropped, read-only data, memory and PID limits, GPU through device reservations.

**Spec PR-03 requirements (7.3, 7.4, 11, 13.3).**

| Requirement | Status |
|---|---|
| Host GPU preflight (`gpu_smoke.py`, `sm_120`) | present |
| 100 ms / 1 s synthetic and full-network entrypoints | present (B0, B0-1s, B1, B2) |
| Synchronized timing with warm-up and repeats | present |
| Peak RAM and VRAM | present; findings 8, 7 |
| CPU/GPU comparison | present; findings 1, 2, 3, 5, 6 |
| GPU results UNVERIFIED until host logs exist | followed |

Spec 11.1 fields are present as listed in `BENCHMARK_PROTOCOL.md`. "CUDA runtime" is the
torch build (13.2); the driver's CUDA version is not recorded.

## M3 numbers against the evidence

Every number in `docs/acceptance/M3.md` was recomputed from the records (rounding as printed):

| M3 statement | Record value |
|---|---|
| B0 wall 1.777 [1.440, 2.202] (1.358–2.433) s, 568 steps/s, 0.057, 128 spikes, 6.4e-6, 22/6 MiB, RSS 1,489 MiB | 1.7768 [1.4404, 2.2021] (1.3579–2.4326); 567.97; 0.0568; 128; 6.4e-6; 23,068,672 / 6,232,064 B; 1,561,628,672 B |
| B0-1s 11.855 [11.076, 12.959] (10.296–14.063), 844, 0.084, 128, 6.4e-7, 22/6, 1,490 | 11.8551 [11.0755, 12.9592] (10.2959–14.0634); 843.52; 0.0844; 128; 6.4e-7; same; 1,562,058,752 B |
| B1 2.971 [2.784, 3.097] (2.488–4.465), 337, 0.034, 23,138, 1.39e-4, 310/293 MiB, 2,395 | 2.9713 [2.7839, 3.0967] (2.4877–4.4652); 336.67; 0.0337; 23,138; 1.388e-4; 325,058,560 / 306,814,464 B; 2,511,540,224 B |
| B2 27.661 [26.879, 29.090] (25.816–38.213), 362, 0.036, 237,655, 1.43e-4, 310/293, 2,394 | 27.661 [26.8793, 29.0896] (25.8162–38.2126); 361.62; 0.0362; 237,655; 1.426e-4; 325,058,560 / 306,875,904 B; 2,510,254,080 B |
| B1-numpy 0.543 [0.525, 0.554] (0.507–0.565), 1,841, 0.184, 23,138, RSS 1,738 | 0.5432 [0.5249, 0.554] (0.5067–0.5648); 1,841.08; 0.1841; 23,138; 1,822,068,736 B |
| 13 / 4 / 13 / 13 runs, one spike count per workload | 3+10, 1+3, 3+10, 3+10; spike-count sets {128}, {128}, {23,138}, {237,655} |
| Parity B0 / B1 / B2: identical, 1,000 / 1,000 / 10,000 steps, 128 / 23,138 / 237,655 spikes, Δv = Δg = 0 | same in `parity-B0/B1/B2-*.json` (float32, deterministic, margin 0.01, 10/10/100 state checks) |
| tests/gpu: 33,853 / 33,888 spikes, 1.4e-14 mV non-deterministic float64, 0 otherwise | host log lines 30–39 (`1.4210854715202004e-14`) |
| Retained 282 MiB edges + 2.7 MiB state | 296,043,228 B and 2,833,900 B |
| Phases 0.42 / 0.02 / 0.46 / 0.78 / 0.92 s; 2.95 % and 4.27 % spiked at least once | 0.422 / 0.019 / 0.460 / 0.776 / 0.916; 0.02950, 0.04271 |
| 12,227 MiB, 2,122 MiB used before, driver 616.92, 15.2 GiB VM, 6 GiB limit, 8 CPUs, kernel 6.18.33.2 | same (16,283,983,872 B = 15.17 GiB; 6,442,450,944 B) |
| 5.5 × slower; ≈ 28 s per simulated s; 0.036 | 2.9713 / 0.5432 = 5.47; 27.66 s; 0.0362 |
| Profile: 322 µs device/step, 110 launches, 8 copies, 8 syncs, launches ≈ half of host time | 96,715.8 µs / 300 = 322.4 (**double-counted, finding 7**); 109.96; 8.0 (direction not recorded, finding 7); 8.0; 0.497 |
| "peak 310 MiB reserved" / README "~0.3 GiB of VRAM" | allocator figure correct; device-wide increase 511 MiB (finding 8) |

### Do the M3 conclusions stand?

Yes, for what M3 claims, with three corrections (findings 7, 8 and the "device→host" label).

The parity findings do not undo M3's spike identity, for these reasons:
- Every M3 parity record is `identical`, with Δv = Δg = 0 at every state check, so findings 1,
  2 and 5 played no role.
- Finding 3 could in principle hide a NaN behind "max |Δv| = 0". The bench runs of the same
  workloads report `state_finite: true` for every run, which rules that out for B0, B1 and B2.

Two limits of the evidence that M3 already states:
- The full-graph parity was run in float32 with deterministic algorithms only.
- B0's parity involves just 128 spikes from 41 neurons. The substantive synthetic evidence is
  `tests/gpu`, with 33,853 spikes.

`parity-B2` was run by hand after the script (M3 gives the command); `run_local_gpu_bench.sh`
has no `parity_B2` step.

## Notes (not filed)

- `flylab parity --state-every 0` raises `ValueError` from `lockstep`, which `_measure` does not
  catch: a traceback instead of exit code 5.
- `lockstep` does not check that both engines use the same `disc` or populations (see
  finding 2's fix).
- The `synaptic_events` definition counts spikes of the last `delay` steps that arrive after the
  run; documented.

## Reproduction

```bash
uv sync --locked --group dev                 # add --extra gpu for finding 7's test
uv run pytest tests/audit -rxX               # 12 xfailed (without torch: 11 + finding 7 skipped)
uv run pytest tests/audit --runxfail         # shows each finding's failing assertion
uv run python docs/audit/repro/pr03_parity_experiments.py   # experiments A and B (findings 1, 2)
```

The experiment numbers above were produced on Linux x86_64 with the locked NumPy 2.5.4.

## Resolution verification

All nine findings are addressed locally. The PR-03 audit tests are ordinary passing
regressions with their finding numbers retained; no PR-03 xfail markers remain. The historical
review/evidence above is retained for traceability. No RTX host was contacted for these fixes;
new CUDA profiling and memory telemetry remain UNVERIFIED there until Claude's next host run.
