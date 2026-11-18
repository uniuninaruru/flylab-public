# Experiment config (schema v1) and run artifacts

`flylab run --config <file.yaml>` validates a config, runs it on the CPU engines and writes one run
directory per variant. The model it simulates is specified in `MODEL_ASSUMPTIONS.md`.

```bash
flylab run --config configs/tiny_fixture.yaml --with-baseline   # synthetic fixture
flylab run --config configs/synthetic.yaml --with-baseline      # synthetic random graph
flylab run --config configs/male_cns_template.yaml --validate-only   # rejected until edited
flylab run --config configs/male_cns_lb3c_mn9.yaml   # real MaleCNS (needs fetch + prepare)
```

Options: `--out runs` (parent directory), `--run-id ID` (default `<experiment_id>-<UTC>-<hash>`),
`--with-baseline` (also run the matched baseline: same config without interventions),
`--validate-only` (check config + data, write nothing). Exit codes: 0 ok, 3 data error,
4 config error, 5 run error (e.g. output exists, torch missing). Relative paths in a config are
resolved against the current working directory and recorded as written; prefer relative paths,
because an absolute path (which may contain a user name) would be copied into the manifest.
A run is written to `.<run_id>.partial/` and renamed when complete, so an interrupted run never
looks finished.

## Validation

Two stages, both before anything is written:

1. **Schema** (`flylab.experiments.config`): unknown keys, wrong types, out-of-range values,
   undefined population references, times off the `dt` grid or past the end of the run,
   `dt` that does not divide `t_ref`/`delay`, parameter overrides without units or evidence label.
2. **Data** (`flylab.experiments.populations`, `runner`): every `body_id` must exist, every
   `column` must be a neuron-table column, every `values` entry must match at least one neuron,
   the transmitter column must exist and contain only mapped values, and `unknown_sign_policy:
   error` fails if any edge sign is unknown. Placeholders such as
   `PLACEHOLDER_VERIFY_WITH_DATA` are therefore rejected (spec 8.1 warning).

## Keys

| Key | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `1` |
| `experiment_id` | yes | `[A-Za-z0-9][A-Za-z0-9_.-]*` |
| `description` | no | free text |
| `dataset` | yes | `kind: processed` (`path`, optional `expected_dataset_id`) · `kind: csv_fixture` (`neurons`, `edges`, `duplicate_rule`, `dataset_id`) · `kind: synthetic_random` (`n_neurons`, `mean_out_degree`, `graph_seed`, optional `inhibitory_fraction`) |
| `backend` | yes | `numpy` (reference), `torch-cpu` or `torch-cuda`; a torch backend that cannot run in this process (no torch, no CUDA device) is a run error before anything is written |
| `dtype` | no | `float64` (default) or `float32` |
| `engine_impl` | no | `torch-cuda` only: `auto` (default; the fused Triton engine where it can run, else eager), `eager` or `fused` (a run error if it cannot run here). Written to the frozen config only when not `auto`; the implementation actually used is in `manifest.json` → `engine.impl` |
| `dt_ms`, `steps` | yes | time step (ms) and number of steps |
| `seed` | yes | seeds Poisson stimuli (`SeedSequence([seed, k])`) and the global RNGs |
| `deterministic` | no | default `true`: `torch.use_deterministic_algorithms(True)` for torch runs (on CUDA this fixes the order of the float64 arrival sums; see MODEL_ASSUMPTIONS 2.3) |
| `model` | yes | `parameter_set: shiu2024_code_v1`; `overrides: {name: {value, units, evidence, source or rationale}}` |
| `synapse_sign` | yes | `source: graph` or `transmitter` (+ `transmitter_column`, `mapping`: `fast_ionotropic_v1` or `shiu2024_methods_v1`, `overrides`); `unknown_sign_policy` always required |
| `populations` | yes | `name: {body_ids: [...]}` or `name: {column: ..., values: [...]}`; `__all__` is reserved |
| `stimuli` | yes (may be `[]`) | `injected_current` (`amplitude_mV`) or `forced_spikes` (`rate_hz` or `times_ms`), each with `population`, `onset_ms`, `offset_ms` |
| `interventions` | yes (may be `[]`) | `silence_neurons` (`mode: block_output` or `clamp_to_rest`, `population`, optional window) · `remove_synapses` (`pre_population` and/or `post_population`, optional `fraction` + `seed`) |
| `recording` | yes | `spike_events` (default true), `population_rate_bin_ms` (must divide the run), `populations` (default: all + `__all__`) |
| `sensitivity` | no | list of `{name, synapse_sign: {...}}`; each runs as a sibling run `<run_id>--sens-<name>` |
| `graph_null` | no | `{method: degree_preserving_swap, seed: <int>}`: run on a shuffled **control** graph instead of the connectome (PR-04, `MODEL_ASSUMPTIONS.md` §9); recorded in `manifest.json` → `graph_null` and in `limitations` |

Differences from the proposal in spec 8.1: `stimulus`/`intervention` became lists `stimuli` /
`interventions` (an empty list is the `none` baseline); `neuron_selection` became named
`populations` referenced by name; `magnitude` became `amplitude_mV` / `rate_hz` / `times_ms`
with explicit units.

## Run directory (spec 8.4)

| File | Content |
|---|---|
| `config.yaml` | the validated config in canonical form (re-loadable) |
| `manifest.json` | run id and group, engine and torch settings, labelled parameters, `dt`, discretization, sign mapping/policy/counts, network summary + memory estimate, populations, stimuli, interventions, dataset ids/fingerprints/file hashes (`dataset.biological_data` is true only for a processed directory whose import verified a registered release, never for `synthetic:` ids), environment (no host name or user), limitations |
| `dataset_sha256.txt` | `sha256sum` lines of each dataset file read (empty for `synthetic_random`) |
| `git_commit.txt` | commit, dirty flag, source |
| `spikes.parquet` | one row per spike: `step`, `t_ms` (left edge), `idx`, `body_id`, `cause` (`threshold`/`forced`), `propagated` |
| `population_rates.parquet` | per bin and population: `spike_count`, `rate_hz` = count / (n_neurons × bin width) |
| `interventions.parquet` | one row per stimulated/silenced neuron and per removed edge |
| `metrics.json` | totals, per-population rates, fraction spiking, first-spike latency summary (per neuron, its first spike at or after the earliest stimulus onset; earlier spontaneous spikes do not remove a neuron from the statistic); `vs_baseline` for the main run when `--with-baseline` |
| `benchmark.json` | side observation only: `simulate_wall_time_s`, `steps_per_s`, and `process_peak_rss_bytes` (lifetime maximum of the process, so it includes dataset loading and earlier variants of the same `flylab run`); GPU fields `null` (not measured) |
| `checksums.sha256` | SHA-256 of every file above (`sha256sum -c` compatible) |

Spikes are streamed in row groups of up to 10^6 rows; memory stays O(N + populations × bins),
never O(N × T). Parquet key-value metadata carries `flylab.schema_version`, `flylab.units` and
`flylab.description`.

## Replicate groups (PR-04): `flylab replicate`

A group runs one base experiment config under several **conditions** for several **seeds** and
writes paired statistics (`flylab.experiments.replicates`, `flylab.experiments.stats`).

```bash
flylab replicate run --config configs/m4/<group>.yaml --out runs --jobs 4   # --resume to continue
flylab replicate run --config configs/m4/<group>.yaml --validate-only
flylab replicate stats --group-dir runs/<group_id>                          # recompute statistics
```

| Key | Required | Meaning |
|---|---|---|
| `schema_version`, `kind` | yes | `1`, `flylab_replicate_group` |
| `group_id` | yes | output directory name under `--out` |
| `base_config` | yes | an experiment config (schema v1); its SHA-256 is frozen into `group.yaml` |
| `base_patch` | no | deep-merged into every member (mappings merge, other values replace, `null` deletes) |
| `seeds` | yes | `{first, count}` or a list; every member's `seed` is set to the replicate seed |
| `conditions` | one of | `[{name, patch, description}]`; an empty patch is the base config |
| `sweep` | one of | `[{axis, levels: [{label, patch}]}]`: cartesian product, condition name = labels joined by `_` |
| `comparisons` | no | `[{condition, reference}]`, paired by seed |
| `statistics` | no | `n_boot` (10000), `confidence` (0.95), `seed` (0), `psth` (false), `populations`, `metrics` (subset of `mean_rate_hz`, `fraction_spiking`, `latency_median_ms`), `population_contrasts: [{population, reference}]` (paired within the same runs) |

Rules: a member's merged config is validated like any `flylab run` config; `sensitivity:` is
refused in members (each variant is a condition instead); `graph_null.seed: replicate` gives each
replicate its own null graph. Members with the same seed receive identical Poisson draws for
identical stimuli. `--jobs N` forks N workers that share the loaded graph copy-on-write (Linux /
macOS; Python warns that forking a multi-threaded process can deadlock, so the default is 1).
`--resume` reuses complete members whose `config_sha256` matches and deletes `.partial`
directories; a changed group config or base config is refused.

Group directory: `group.yaml` (frozen canonical config), `members/<condition>--s<seed>/` (ordinary
run directories), `group_manifest.json`, `group_stats.json`, `checksums.sha256`.

`group_stats.json`: per condition, population and metric the per-seed values, mean, sd, median,
range and a BCa bootstrap interval of the mean; per comparison and population contrast the paired
mean difference (BCa interval), Cohen's d_z (percentile interval), relative change (paired
percentile interval), a two-sided sign-flip permutation p-value (exact up to 16 pairs) and its
Holm-adjusted value over the family stated in `statistics_method.multiple_comparisons`. The
`evidence` block gives one label for every simulated value in the file: the weakest label among
the members' inputs (parameters, sign table, stimuli, null graph), order
UNKNOWN < ASSUMED < ENGINEERED < LEARNED < LITERATURE_DERIVED < EMPIRICAL. The unit of
replication is the stimulus seed on one fixed graph — never animals.

`w_syn` calibration from a w_syn × rate sweep: `flylab.experiments.calibration` (used by
`scripts/m4/analyse.py`); definitions in `docs/acceptance/M4.md`.
