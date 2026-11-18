"""Run artifacts (spec 8.4): chunked Parquet writers, metrics, manifest, checksums.

Layout of runs/<run_id>/ (all files carry `schema_version`; Parquet files carry it, the units
and a description in their key-value metadata):

    config.yaml             frozen, validated config (canonical form)
    manifest.json           provenance: versions, dataset hashes, parameters, signs, ...
    dataset_sha256.txt      sha256sum lines of every dataset file read (empty for synthetic)
    git_commit.txt          commit + dirty flag of the code that ran
    spikes.parquet          one row per spike: step, t_ms, idx, body_id, cause, propagated
    population_rates.parquet  per bin and population: spike_count, rate_hz
    interventions.parquet   one row per stimulus/intervention target (neuron or removed edge)
    metrics.json            totals and per-population summaries
    benchmark.json          only what was measured; null otherwise
    checksums.sha256        SHA-256 of every file above
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from flylab import SCHEMA_VERSION
from flylab.neural.engine import NeuralStep

DEFAULT_CHUNK_ROWS = 1_000_000

SPIKES_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("t_ms", pa.float64()),
        ("idx", pa.int32()),
        ("body_id", pa.int64()),
        ("cause", pa.string()),
        ("propagated", pa.bool_()),
    ]
)
SPIKES_UNITS = {"step": "step", "t_ms": "ms (left edge of the step)", "idx": "neuron index"}

RATES_SCHEMA = pa.schema(
    [
        ("bin", pa.int32()),
        ("t_start_ms", pa.float64()),
        ("t_end_ms", pa.float64()),
        ("population", pa.string()),
        ("n_neurons", pa.int64()),
        ("spike_count", pa.int64()),
        ("rate_hz", pa.float64()),
    ]
)
RATES_UNITS = {"t_start_ms": "ms", "t_end_ms": "ms", "rate_hz": "Hz per neuron"}

INTERVENTIONS_SCHEMA = pa.schema(
    [
        ("category", pa.string()),  # stimulus | intervention
        ("config_index", pa.int32()),  # position in the config list
        ("kind", pa.string()),
        ("mode", pa.string()),
        ("population", pa.string()),
        ("target", pa.string()),  # neuron | edge
        ("idx", pa.int32()),  # neuron, or presynaptic neuron of an edge
        ("body_id", pa.int64()),
        ("dst_idx", pa.int32()),
        ("dst_body_id", pa.int64()),
        ("synapse_count", pa.int64()),
        ("onset_step", pa.int64()),
        ("offset_step", pa.int64()),  # exclusive
        ("amplitude_mV", pa.float64()),
        ("rate_hz", pa.float64()),
    ]
)
INTERVENTIONS_UNITS = {"amplitude_mV": "mV (R*I_ext)", "rate_hz": "Hz", "onset_step": "step"}

OUTPUT_DESCRIPTIONS: dict[str, dict[str, object]] = {
    "spikes.parquet": {"columns": SPIKES_SCHEMA.names, "units": SPIKES_UNITS},
    "population_rates.parquet": {"columns": RATES_SCHEMA.names, "units": RATES_UNITS},
    "interventions.parquet": {"columns": INTERVENTIONS_SCHEMA.names, "units": INTERVENTIONS_UNITS},
}


def _metadata(units: Mapping[str, str], description: str) -> dict[bytes, bytes]:
    return {
        b"flylab.schema_version": str(SCHEMA_VERSION).encode(),
        b"flylab.units": json.dumps(dict(units)).encode(),
        b"flylab.description": description.encode(),
    }


class ChunkedParquetWriter:
    """Buffers column chunks and writes a row group every `chunk_rows` rows."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        units: Mapping[str, str],
        description: str,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ) -> None:
        self.path = path
        self.schema = schema.with_metadata(_metadata(units, description))
        self.chunk_rows = chunk_rows
        self._writer = pq.ParquetWriter(path, self.schema)
        self._buf: list[pa.RecordBatch] = []
        self._rows = 0
        self.rows_written = 0

    def append(self, columns: Mapping[str, object]) -> None:
        batch = pa.RecordBatch.from_pydict(dict(columns), schema=self.schema)
        if batch.num_rows == 0:
            return
        self._buf.append(batch)
        self._rows += batch.num_rows
        if self._rows >= self.chunk_rows:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._writer.write_table(pa.Table.from_batches(self._buf, schema=self.schema))
            self.rows_written += self._rows
            self._buf, self._rows = [], 0

    def close(self) -> None:
        self.flush()
        self._writer.close()


class RunRecorder:
    """Engine recorder: streams spikes to Parquet and accumulates bounded summaries.

    Memory is O(N + n_populations * n_bins + chunk_rows), never O(N * T).
    """

    def __init__(
        self,
        out_dir: Path,
        body_ids: NDArray[np.int64],
        populations: Mapping[str, NDArray[np.int32]],
        steps: int,
        bin_steps: int,
        spike_events: bool,
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
        *,
        reference_step: int = 0,
    ) -> None:
        if reference_step < 0:
            raise ValueError("reference_step must be >= 0")
        self.body_ids = body_ids
        self.reference_step = reference_step
        self.populations = dict(populations)
        self.bin_steps = bin_steps
        self.n_bins = steps // bin_steps
        n = body_ids.size
        self.spike_count = np.zeros(n, dtype=np.int64)
        self.first_spike_step = np.full(n, -1, dtype=np.int64)
        # First spike at or after `reference_step` (the latency metric); -1 = none yet.
        self.first_spike_after_ref = np.full(n, -1, dtype=np.int64)
        self.n_threshold = 0
        self.n_forced = 0
        self.n_not_propagated = 0
        self.n_forced_dropped = 0
        self.bin_counts = {name: np.zeros(self.n_bins, dtype=np.int64) for name in populations}
        self._member = {}
        for name, idx in populations.items():
            m = np.zeros(n, dtype=bool)
            m[idx] = True
            self._member[name] = m
        self._writer = None
        if spike_events:
            self._writer = ChunkedParquetWriter(
                out_dir / "spikes.parquet",
                SPIKES_SCHEMA,
                SPIKES_UNITS,
                "one row per spike; idx follows flylab.data.graph.NeuronIndex",
                chunk_rows,
            )

    def abort(self) -> None:
        """Close open writers after cancellation; leave the .partial directory incomplete."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def record(self, step: NeuralStep) -> None:
        idx = step.spike_idx
        self.n_forced_dropped += step.n_forced_dropped
        if idx.size == 0:
            return
        self.spike_count[idx] += 1
        first = self.first_spike_step[idx]
        self.first_spike_step[idx] = np.where(first < 0, step.step, first)
        if step.step >= self.reference_step:
            after = self.first_spike_after_ref[idx]
            self.first_spike_after_ref[idx] = np.where(after < 0, step.step, after)
        n_forced = int(step.forced.sum())
        self.n_forced += n_forced
        self.n_threshold += int(idx.size) - n_forced
        self.n_not_propagated += int((~step.propagated).sum())
        b = step.step // self.bin_steps
        for name, m in self._member.items():
            self.bin_counts[name][b] += int(m[idx].sum())
        if self._writer is not None:
            self._writer.append(
                {
                    "step": np.full(idx.size, step.step, dtype=np.int64),
                    "t_ms": np.full(idx.size, step.t_ms, dtype=np.float64),
                    "idx": idx,
                    "body_id": self.body_ids[idx],
                    "cause": np.where(step.forced, "forced", "threshold").tolist(),
                    "propagated": step.propagated,
                }
            )

    def close(self, out_dir: Path, dt_ms: float) -> None:
        if self._writer is not None:
            self._writer.close()
        w = ChunkedParquetWriter(
            out_dir / "population_rates.parquet",
            RATES_SCHEMA,
            RATES_UNITS,
            "rate_hz = spike_count / (n_neurons * bin width in s)",
        )
        bin_ms = self.bin_steps * dt_ms
        bins = np.arange(self.n_bins, dtype=np.int32)
        for name, counts in self.bin_counts.items():
            n_pop = int(self.populations[name].size)
            w.append(
                {
                    "bin": bins,
                    "t_start_ms": bins * bin_ms,
                    "t_end_ms": (bins + 1) * bin_ms,
                    "population": [name] * self.n_bins,
                    "n_neurons": np.full(self.n_bins, n_pop, dtype=np.int64),
                    "spike_count": counts,
                    "rate_hz": counts / (n_pop * bin_ms / 1000.0),
                }
            )
        w.close()

    def mean_rates_hz(self, dt_ms: float, steps: int) -> dict[str, float]:
        seconds = steps * dt_ms / 1000.0
        return {
            name: float(self.spike_count[idx].sum() / (idx.size * seconds))
            for name, idx in self.populations.items()
        }

    def population_metrics(
        self, dt_ms: float, steps: int, reference_ms: float
    ) -> dict[str, dict[str, object]]:
        if abs(reference_ms - self.reference_step * dt_ms) > 1e-9 * max(1.0, abs(reference_ms)):
            raise ValueError(
                f"reference_ms {reference_ms} != reference_step {self.reference_step} x dt {dt_ms}"
            )
        out: dict[str, dict[str, object]] = {}
        seconds = steps * dt_ms / 1000.0
        for name, idx in self.populations.items():
            counts = self.spike_count[idx]
            fired = self.first_spike_step[idx] >= 0
            after = self.first_spike_after_ref[idx]
            lat = (after[after >= 0] - self.reference_step) * dt_ms
            out[name] = {
                "n_neurons": int(idx.size),
                "n_spikes": int(counts.sum()),
                "mean_rate_hz": float(counts.sum() / (idx.size * seconds)),
                "fraction_spiking": float(fired.mean()),
                "first_spike_latency_ms": {
                    "reference_ms": reference_ms,
                    "n_neurons": int(lat.size),
                    "min": float(lat.min()) if lat.size else None,
                    "median": float(np.median(lat)) if lat.size else None,
                    "max": float(lat.max()) if lat.size else None,
                    "note": "per neuron, first spike at or after the reference time",
                },
            }
        return out


def write_interventions(
    path: Path, rows: Sequence[Mapping[str, object]], chunk_rows: int = DEFAULT_CHUNK_ROWS
) -> int:
    """`rows` are column dicts (one dict per stimulus/intervention, arrays of equal length)."""
    w = ChunkedParquetWriter(
        path,
        INTERVENTIONS_SCHEMA,
        INTERVENTIONS_UNITS,
        "one row per target neuron of each stimulus/silencing, one per removed edge",
        chunk_rows,
    )
    for cols in rows:
        n = max(len(v) for v in cols.values() if isinstance(v, list | np.ndarray))
        full = {}
        for name in INTERVENTIONS_SCHEMA.names:
            v = cols.get(name)
            full[name] = v if isinstance(v, list | np.ndarray) else [v] * n
        w.append(full)
    w.close()
    return w.rows_written


def write_json(path: Path, obj: object) -> None:
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8"
    )


def read_parquet_metadata(path: Path) -> dict[str, str]:
    md = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in md.items()}
