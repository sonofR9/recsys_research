from __future__ import annotations

from dataclasses import replace
from itertools import islice
import json
import os
from pathlib import Path
from types import MethodType

import torch

from dcn.datasets.yambda import UserSample
from experiments.g2_esasrec.analysis.fit_evidence import (
    FitProbe,
    fit_device_evidence,
    write_fit_evidence,
)
from experiments.g2_esasrec.configs.local import (
    CONTROL_BATCHES,
    LIGR_WIDTHS,
    build_component,
)
from experiments.g2_esasrec.protocol.manifest import approved_manifest
from neuralrec.run.callbacks import ResourceUsageCallback
from neuralrec.utils import EXTRA_METRICS, to_float
from utils.locks import hold


def _batch_size() -> int:
    raw = os.environ.get("G2_FIT_BATCH_SIZE", "")
    if not raw.isdigit() or int(raw) not in CONTROL_BATCHES:
        raise ValueError("G2_FIT_BATCH_SIZE must be one approved control batch")
    return int(raw)


batch_size = _batch_size()
experiment = replace(
    build_component(
        "ligr_gbce",
        batch_size=batch_size,
        ligr_multiplier=6,
        gbce_t=0.75,
        run_name=f"g2_fit_probe_ligr_m6_b{batch_size}_native50m_diagnostic",
    ),
    user_sample=UserSample(max_users=2_000, seed=42),
    num_epochs=1,
)
experiment.__dict__["sequence_callbacks"] = []
original_create_trainer = experiment.create_trainer


def create_trainer(owner, model, optimizer):
    trainer = original_create_trainer(model, optimizer)
    trainer.callbacks = [
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ResourceUsageCallback)
    ]

    def train_one_step(current) -> None:
        owner.__dict__["fit_probe_device"] = fit_device_evidence(current.device)
        current._fire_callbacks("on_train_begin", current.state)
        loader = current._prepared_train_iterator or iter(current.train_loader)
        current._prepared_train_iterator = None
        try:
            current.train_epoch(
                0,
                islice(loader, current.gradient_accumulation_steps),
            )
            owner.__dict__["fit_probe_fits"] = current.global_step == 1
        except torch.cuda.OutOfMemoryError:
            owner.__dict__["fit_probe_fits"] = False
            torch.cuda.empty_cache()
        current._fire_callbacks("on_train_end", current.state)

    trainer.train = MethodType(train_one_step, trainer)
    return trainer


def finish(owner, runner) -> None:
    resources = runner.state.get(EXTRA_METRICS, {}).get("resources", {})
    peak_memory = to_float(resources.get("peak_memory_gb")) or 0.0
    fits = owner.__dict__.get("fit_probe_fits") is True
    device_evidence = owner.__dict__.get("fit_probe_device")
    if not isinstance(device_evidence, dict):
        raise RuntimeError("fit probe device evidence is absent")
    root = Path(owner.base_path) / "logs" / owner.run_name
    artifact = root / "fit_probe.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact_document = {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m-diagnostic-2000-users-by-id",
        "ligr_multiplier": 6,
        "ffn_width": LIGR_WIDTHS[6],
        "loss_kind": "gbce",
        "gbce_t": 0.75,
        "optimizer_steps": 1,
        **device_evidence,
        "batch_size": batch_size,
        "fits": fits,
        "peak_memory_gb": peak_memory,
        "user_sample": owner.user_sample.name,
        "user_sample_query": owner.user_sample.duckdb_query("users"),
    }
    artifact_temporary = artifact.with_suffix(".json.tmp")
    artifact_temporary.write_text(
        json.dumps(artifact_document, indent=2, sort_keys=True) + "\n"
    )
    artifact_temporary.replace(artifact)
    index = Path(owner.base_path) / "logs/g2_fit_probes_native50m.json"
    with hold(index.with_suffix(".lock"), "fit evidence"):
        probes = []
        if index.exists():
            document = json.loads(index.read_text())
            probes = [FitProbe(**row) for row in document["probes"]]
        by_batch = {probe.batch_size: probe for probe in probes}
        by_batch[batch_size] = FitProbe(batch_size, fits, str(artifact))
        write_fit_evidence(
            [by_batch[value] for value in sorted(by_batch)],
            index,
        )


experiment.create_trainer = MethodType(create_trainer, experiment)
experiment.finish = MethodType(finish, experiment)
