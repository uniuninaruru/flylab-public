"""Load one experiment config's main variant up to a ready network, timing each phase.

Shared by `flylab bench` and `flylab parity`. The variant goes through the same validation
as `flylab run` (`runner._prepare`), so a benchmark can never run a network `flylab run`
would refuse.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from flylab.experiments.config import ExperimentConfig, load_config, parse_config
from flylab.experiments.dataset import LoadedDataset, load_dataset
from flylab.experiments.populations import resolve_populations
from flylab.experiments.runner import PreparedRun, RunError, _build_network, _prepare
from flylab.experiments.schedule import ExperimentSchedule, compile_silences, compile_stimuli
from flylab.neural.network import SynapticNetwork


@dataclass(frozen=True, eq=False)
class Workload:
    config: ExperimentConfig
    config_path: str | None
    config_sha256: str
    data: LoadedDataset
    pops: dict[str, NDArray[np.int32]]
    prep: PreparedRun
    network: SynapticNetwork
    phases_s: dict[str, float]

    def schedule(self, seed: int | None = None) -> ExperimentSchedule:
        """Fresh per-step inputs; `seed` defaults to the config's (Poisson draws depend on it)."""
        cfg = self.config
        return ExperimentSchedule(
            compile_stimuli(cfg.stimuli, self.pops, cfg.dt_ms),
            compile_silences(cfg.interventions, self.pops, cfg.dt_ms),
            cfg.seed if seed is None else seed,
        )

    def recorded(self) -> dict[str, NDArray[np.int32]]:
        return {k: self.pops[k] for k in self.config.recorded_populations()}


def with_overrides(cfg: ExperimentConfig, **overrides: object) -> ExperimentConfig:
    """The config with top-level fields replaced (None = keep), validated again from scratch."""
    doc = yaml.safe_load(cfg.to_yaml())
    doc.update({k: v for k, v in overrides.items() if v is not None})
    return parse_config(doc)


def load_workload(config: ExperimentConfig | str | Path, **overrides: object) -> Workload:
    path = None if isinstance(config, ExperimentConfig) else str(config)
    cfg = config if isinstance(config, ExperimentConfig) else load_config(config)
    if any(v is not None for v in overrides.values()):
        cfg = with_overrides(cfg, **overrides)
    if cfg.backend != "numpy":  # fail before loading a multi-GB dataset
        from flylab.neural.lif_torch import TorchUnavailableError, check_backend

        try:
            check_backend(cfg.backend)
        except TorchUnavailableError as exc:
            raise RunError(str(exc)) from exc
    phases: dict[str, float] = {}

    t0 = time.perf_counter()
    data = load_dataset(cfg.dataset)
    phases["load_dataset"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pops = resolve_populations(cfg.populations, data)
    phases["resolve_populations"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    prep = _prepare("main", cfg, data, pops)
    phases["prepare_signs_and_removal"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    net = _build_network(prep)
    phases["build_network"] = time.perf_counter() - t0
    return Workload(
        config=cfg,
        config_path=path,
        config_sha256=hashlib.sha256(cfg.to_yaml().encode()).hexdigest(),
        data=data,
        pops=pops,
        prep=prep,
        network=net,
        phases_s=phases,
    )
