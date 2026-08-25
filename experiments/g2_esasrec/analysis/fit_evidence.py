from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import torch
from experiments.g2_esasrec.configs.local import CONTROL_BATCHES, LIGR_WIDTHS
from experiments.g2_esasrec.protocol.manifest import approved_manifest

FIT_DEVICE_NAME = "NVIDIA A100-SXM4-80GB"
FIT_DEVICE_COMPUTE_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class FitProbe:
    batch_size: int
    fits: bool
    artifact: str


@dataclass(frozen=True)
class FitEvidence:
    probes: tuple[FitProbe, ...]

    @property
    def eligible_batches(self) -> set[int]:
        return {probe.batch_size for probe in self.probes if probe.fits}


def fit_device_evidence(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        raise RuntimeError("fit probe requires NVIDIA A100-SXM4-80GB CUDA")
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    if name != FIT_DEVICE_NAME or capability != FIT_DEVICE_COMPUTE_CAPABILITY:
        raise RuntimeError(
            "fit probe requires NVIDIA A100-SXM4-80GB with capability 8.0"
        )
    return {
        "device_name": name,
        "device_compute_capability": list(capability),
    }


def load_fit_evidence(path: Path) -> FitEvidence:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fit evidence {path}") from error
    expected = {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m-diagnostic-2000-users-by-id",
        "ligr_multiplier": 6,
        "ffn_width": LIGR_WIDTHS[6],
        "loss_kind": "gbce",
        "gbce_t": 0.75,
        "optimizer_steps": 1,
        "device_name": FIT_DEVICE_NAME,
        "device_compute_capability": list(FIT_DEVICE_COMPUTE_CAPABILITY),
    }
    for name, value in expected.items():
        if document.get(name) != value:
            raise ValueError(f"fit evidence {name} changed")
    raw_probes = document.get("probes")
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError("fit evidence has no probes")
    probes = []
    for row in raw_probes:
        if not isinstance(row, dict):
            raise ValueError("fit probe must be an object")
        batch_size = row.get("batch_size")
        fits = row.get("fits")
        artifact = row.get("artifact")
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or not isinstance(fits, bool)
            or not isinstance(artifact, str)
            or not artifact
        ):
            raise ValueError("fit probe is incomplete")
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        try:
            artifact_document = json.loads(artifact_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"cannot read fit probe artifact {artifact_path}"
            ) from error
        artifact_expected = expected | {
            "batch_size": batch_size,
            "fits": fits,
        }
        for name, value in artifact_expected.items():
            if artifact_document.get(name) != value:
                raise ValueError(f"fit probe artifact {name} changed")
        probes.append(FitProbe(batch_size, fits, artifact))
    if len({probe.batch_size for probe in probes}) != len(probes):
        raise ValueError("fit evidence contains duplicate batch sizes")
    if {probe.batch_size for probe in probes} != set(CONTROL_BATCHES):
        raise ValueError("fit evidence requires every approved control batch")
    return FitEvidence(tuple(probes))


def _atomic_write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_one_step_fit_probe(
    step: Callable[[], object],
    *,
    batch_size: int,
    destination: Path,
    device: torch.device,
) -> FitProbe:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("fit-probe batch_size must be a positive integer")
    device_evidence = fit_device_evidence(device)
    peak_memory_gb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fits = True
    except torch.cuda.OutOfMemoryError:
        fits = False
    finally:
        if device.type == "cuda":
            peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            torch.cuda.empty_cache()
    document = {
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
        "peak_memory_gb": peak_memory_gb,
    }
    _atomic_write(destination, document)
    return FitProbe(batch_size, fits, str(destination))


def write_fit_evidence(probes: list[FitProbe], destination: Path) -> None:
    if not probes:
        raise ValueError("cannot write empty fit evidence")
    document = {
        "manifest_sha256": approved_manifest().sha256,
        "dataset_size": "native-50m-diagnostic-2000-users-by-id",
        "ligr_multiplier": 6,
        "ffn_width": LIGR_WIDTHS[6],
        "loss_kind": "gbce",
        "gbce_t": 0.75,
        "optimizer_steps": 1,
        "device_name": FIT_DEVICE_NAME,
        "device_compute_capability": list(FIT_DEVICE_COMPUTE_CAPABILITY),
        "probes": [
            {
                "batch_size": probe.batch_size,
                "fits": probe.fits,
                "artifact": probe.artifact,
            }
            for probe in probes
        ],
    }
    _atomic_write(destination, document)
    if {probe.batch_size for probe in probes} == set(CONTROL_BATCHES):
        load_fit_evidence(destination)
