# Scientific limitations

Status: **PR-04** adds replicated MaleCNS runs with controls, a `w_syn` calibration against the
criterion of Shiu et al. (2024) and a cross-check on their FlyWire graph
(`docs/acceptance/M4.md`). No comparison with recordings or behaviour exists; every simulated
value is a model output labelled ASSUMED.

## What FlyLab is

- A software project that will run a simple, source-traceable spiking model on the MaleCNS v1.0
  directed connectome, with labelled assumptions and reproducible runs.

## What FlyLab is not, and must never be described as

- Not a complete or faithful model of a fly brain. Connectivity is empirical; dynamics are modelled.
- No claims of consciousness, subjective experience, or authentic fly behaviour.
- Shiu et al. (2024) validated a **female** whole-brain model (FlyWire). That does not validate
  MaleCNS outputs from FlyLab or any body decoder.
- A future body coupling (flybody) is an **engineering demonstration** with an explicit decoder
  (ENGINEERED or LEARNED), not evidence that the connectome generates locomotion.

## Known limitations (will grow)

| # | Limitation | Consequence |
|---|---|---|
| L1 | Neurotransmitter identities are predictions | Synapse signs are uncertain; unknown signs must be handled explicitly |
| L2 | Synapse counts ≠ synaptic strength | Weight scale is a model parameter |
| L3 | Published neuron counts differ between sources | Node set is a named configuration (see `DATASET_COUNT_RECONCILIATION.md`) |
| L4 | Tiny fixture graphs are synthetic | They test code, not biology |
| L5 | Only a GPU smoke test has run (2026-11-04); no full-network GPU run yet | No performance claims yet |
| L6 | Every neuron gets the same LIF parameters (Shiu et al. reference code values, `MODEL_ASSUMPTIONS.md` §3) | Cell-type differences in excitability, size and time constants are absent |
| L7 | `w_syn` (mV per synapse) is a free parameter copied from a female-brain model | Absolute firing rates on MaleCNS are not calibrated; compare conditions, not raw rates |
| L8 | Synapse sign = presynaptic transmitter via an ASSUMED table; modulatory transmitters are "unknown" | Results depend on the mapping and the unknown-sign policy; report sensitivity variants |
| L9 | Current-based synapses, one uniform delay, no gap junctions, neuromodulation, plasticity or noise | Dynamics are a simplification; silence of a model neuron is not a lesion of a fly neuron |
| L10 | Injected drive is in mV of `R·I` (R unknown); forced spikes bypass the membrane | Stimulus strengths cannot be mapped to experimental currents or light intensities |
| L11 | Interventions are idealised: `block_output`, `clamp_to_rest`, `remove_synapses` (`MODEL_ASSUMPTIONS.md` §6) | They approximate, but are not, Kir2.1/TNT/shibire or optogenetic manipulations |
| L12 | `w_syn` on MaleCNS: the preset keeps the female-brain value 0.275 mV; PR-04 calibrated 0.270 mV (LEARNED) for Shiu et al.'s criterion, weakly identified (tested 0.250, 0.255, 0.260, 0.270 and 0.275 mV compatible) | At 0.25–0.275 mV the whole-brain rate switches across seeds between a low (0.1–0.5 Hz) and a high (3–4 Hz, Kenyon-cell-dominated) regime, and MN9 rates depend on the regime; compare conditions seed-paired and report the regime (M4.md, Q23) |
| L13 | Replicates are stimulus seeds on one graph and one parameter set | Intervals describe the model's sensitivity to its own Poisson input, not animal-to-animal variability; they say nothing about whether the model is right |
| L14 | The `w_syn` calibration criterion ("roughly 80% of maximal MN9 firing") is underspecified in the paper; FlyLab fixes a definition (ENGINEERED) | A calibrated `w_syn` is "calibrated for this criterion on this graph", not a synaptic strength; other definitions give other values (M4.md reports D1-D3) |
| L15 | FlyLab's forced Poisson spikes keep the 2.2 ms refractory period; Shiu et al.'s Brian2 `PoissonInput` kicks `v` by 250 × `w_syn` with the targets' refractory period set to 0 | "GRNs at 100 Hz" is not the same stimulus in both models (FlyLab: ~82 Hz achieved); results that depend on the input mode (e.g. laterality, Q19) are not transferable |
| L16 | The degree-preserving null keeps degrees, out-strength and output signs but not in-strength or any higher-order structure | A difference from the null says "the specific wiring matters beyond degrees", not which feature matters |
| L17 | Synapses onto segments outside the node set (112.6 M weights rows, Q11) are ignored | Input normalisation and excitability of every neuron depend on that choice; not tested in PR-04 |
| L18 | MaleCNS v1.0 MN9 R (16949) has ~10 % of the input synapses of its homologues (`statusLabel` "RT Hard to trace") | It cannot be used as a readout; any left/right comparison on MaleCNS must check reconstruction completeness of both sides first (Q19) |

## Body backend (PR-05, flybody)

- The flybody action vector (59 elements in `walk_imitation`, read at runtime) drives MuJoCo
  actuators of a body model. It is a software interface (ENGINEERED), **not** a biological motor
  map; no neuron is mapped to any element.
- The only decoder, `ZeroDecoder`, ignores neural input. It is a null control, not a controller.
- `walk_imitation` in inference mode follows a synthetic reference trajectory with a constant
  reward; stepping it is a software smoke test, not behavioural evidence.
- There is no closed loop. Body results say nothing about the connectome.
