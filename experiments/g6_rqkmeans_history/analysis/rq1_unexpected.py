from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dcn.data.features import FeatureValues
from dcn.models import SemanticHistoryTokenizer, SequenceRetrievalModel
from dcn.models.two_tower import TwoTowerLoss
from dcn.nn.sampled_softmax import StreamingInBatchSoftmax
from dcn.nn.semantic_embedding import (
    CombinedSemanticIdEmbedding,
    SemanticIdEmbedding,
)
from dcn.nn.types import ModuleWithDim
from dcn.semantic import ResidualCodebooks, SemanticCodes
from experiments.g6_rqkmeans_history.analysis.learning_curves import (
    load_validation_curve,
)
from experiments.g6_rqkmeans_history.configs.rq1 import (
    project_centroids_with_pca,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
G6_ROOT = PROJECT_ROOT / "experiments/g6_rqkmeans_history"
SURFACE = G6_ROOT / "evidence/rq1_rq3_surface_native50m.json"
CONFIRMATION = G6_ROOT / "evidence/rq1_rq3_confirmation_native50m.json"
LOGS_ROOT = PROJECT_ROOT / "generated/logs"
OUTPUT = G6_ROOT / "evidence/rq1_unexpected_native50m.json"
CODEBOOK_PATTERN = "generated/preprocessed/dataset/*/semantic/{cache_key}/codebooks.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    content = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(content).hexdigest()


def projection_reconstruction(
    centroids: torch.Tensor, *, output_dim: int
) -> list[dict[str, int | float]]:
    if centroids.ndim != 3:
        raise ValueError("codebooks must have shape [levels, codes, width]")
    if output_dim < 1 or output_dim > min(centroids.shape[1:]):
        raise ValueError("invalid PCA output width")
    rows = []
    for level, level_centroids in enumerate(centroids.double()):
        centered = level_centroids - level_centroids.mean(dim=0, keepdim=True)
        _, singular_values, right_vectors = torch.linalg.svd(
            centered, full_matrices=False
        )
        components = right_vectors[:output_dim]
        reconstructed = centered @ components.T @ components
        rows.append(
            {
                "level": level,
                "input_width": level_centroids.shape[1],
                "output_width": output_dim,
                "retained_variance_fraction": float(
                    singular_values[:output_dim].square().sum()
                    / singular_values.square().sum()
                ),
                "centered_reconstruction_mse": float(
                    (centered - reconstructed).square().mean()
                ),
            }
        )
    return rows


class _IdentitySequence(ModuleWithDim):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self._out_dim = dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del cumulative_lens, timestamps, kwargs
        return embeddings


def _packed_batch(item_ids: list[int], lengths: list[int]) -> dict[str, object]:
    values = torch.tensor(item_ids)
    return {
        "int_columns": {
            "compact_item_id": FeatureValues(
                values,
                torch.arange(len(values) + 1, dtype=torch.int64),
            )
        },
        "float_columns": {},
        "cumulative_lens": torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
        ),
        "timestamp": torch.arange(len(values), dtype=torch.int64),
    }


def gradient_probe() -> dict[str, int | str | bool]:
    torch.manual_seed(29)
    codes = SemanticCodes.with_collision_suffix(
        item_ids=torch.tensor([1, 2, 3, 4]),
        codes=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
        num_codes=2,
    )
    codebooks = ResidualCodebooks(
        torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            ]
        )
    )
    learned = SemanticIdEmbedding.learned(codes, num_items=4, embedding_dim=3)
    frozen = SemanticIdEmbedding.from_codebooks(
        codes, codebooks, num_items=4, train_collision_suffix=True
    )
    combined = CombinedSemanticIdEmbedding([learned, frozen])
    item_embedding = nn.Embedding(5, 4)
    tokenizer = SemanticHistoryTokenizer.item_learned_frozen_sid_event(
        item_embedding,
        combined,
        "compact_item_id",
        model_dim=4,
        encoder_hidden_dim=3,
    )
    model = SequenceRetrievalModel(
        tokenizer=tokenizer,
        sequence_model=_IdentitySequence(4),
        item_embedding=item_embedding,
        item_id_column="compact_item_id",
    )
    criterion = TwoTowerLoss(
        model,
        StreamingInBatchSoftmax(hash_size=5, num_in_batch_negatives=4),
    )
    frozen_weights = next(
        buffer for name, buffer in frozen.named_buffers() if name.endswith("weights")
    )
    frozen_centroid_parameters = [
        name for name, _ in frozen.named_parameters() if name.endswith("weights")
    ]
    frozen_before = _tensor_sha256(frozen_weights)
    criterion(_packed_batch([1, 2, 3, 4], [4]))["loss"].backward()
    gradient = learned.embedding.weight.grad
    if gradient is None:
        raise RuntimeError("RQ1 learned SID lookup received no gradient")
    base_mask = torch.zeros(len(gradient), dtype=torch.bool)
    for level in range(codebooks.num_levels):
        first, last = codes.vocabulary.level_range(level)
        base_mask[first:last] = True
    rows_with_gradient = int(gradient[base_mask].abs().sum(dim=1).gt(0).sum())
    frozen_after = _tensor_sha256(frozen_weights)
    return {
        "learned_base_gradient_nonzero": rows_with_gradient > 0,
        "learned_base_rows_with_gradient": rows_with_gradient,
        "frozen_centroid_trainable_parameters": len(frozen_centroid_parameters),
        "frozen_centroid_sha256_before": frozen_before,
        "frozen_centroid_sha256_after": frozen_after,
        "frozen_centroid_unchanged": frozen_before == frozen_after,
    }


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return document


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _authenticate_run(
    row: dict[str, Any], logs_root: Path
) -> tuple[dict[str, Any], Path]:
    directory = logs_root / row["run_name"]
    metadata_path = directory / "training_metadata.json"
    sweep_path = directory / "sweep.log"
    expected = row["artifact_sha256"]
    for name, path in (
        ("training_metadata", metadata_path),
        ("sweep_log", sweep_path),
    ):
        if not path.is_file() or _sha256(path) != expected[name]:
            raise ValueError(f"{row['run_name']}: frozen {name} changed")
    return _read_json(metadata_path), sweep_path


def _load_sources(
    surface_path: Path, confirmation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    surface = _read_json(surface_path)
    confirmation = _read_json(confirmation_path)
    if surface.get("schema") != "g6-rq1-rq3-surface/v1":
        raise ValueError("unexpected RQ1 surface schema")
    if confirmation.get("schema") != "g6-rq1-rq3-confirmation/v1":
        raise ValueError("unexpected RQ1 confirmation schema")
    if confirmation.get("surface_sha256") != _sha256(surface_path):
        raise ValueError("RQ1 confirmation no longer binds its surface")
    return surface, confirmation


def _paired_curves(surface: dict[str, Any], logs_root: Path) -> list[dict[str, Any]]:
    rows_by_trial: dict[int, dict[str, tuple[dict[str, Any], Path]]] = {}
    for row in surface["rq1"]["rows"]:
        _, sweep_path = _authenticate_run(row, logs_root)
        trial = int(row["job_id"].rsplit("_", 1)[1])
        initialization = row["parameters"]["sid_lookup_initialization"]
        rows_by_trial.setdefault(trial, {})[initialization] = (row, sweep_path)
    if sorted(rows_by_trial) != list(range(16)):
        raise ValueError("RQ1 paired surface is incomplete")
    summaries = []
    for trial, pair in sorted(rows_by_trial.items()):
        random_row, random_path = pair["random"]
        content_row, content_path = pair["content_pca"]
        random_parameters = random_row["parameters"]
        content_parameters = content_row["parameters"]
        for name in ("embedding_learning_rate", "deep_learning_rate"):
            if random_parameters[name] != content_parameters[name]:
                raise ValueError(f"RQ1 trial {trial:02d} is not LR-paired")
        random_curve = load_validation_curve(random_path)
        content_curve = load_validation_curve(content_path)
        random_raw_auc = sum(random_curve.recall_at_100) / len(
            random_curve.recall_at_100
        )
        content_raw_auc = sum(content_curve.recall_at_100) / len(
            content_curve.recall_at_100
        )
        summaries.append(
            {
                "trial": trial,
                "embedding_learning_rate": random_parameters["embedding_learning_rate"],
                "deep_learning_rate": random_parameters["deep_learning_rate"],
                "random_first_epoch_at_95_percent": (
                    random_curve.first_epoch_at_95_percent
                ),
                "content_first_epoch_at_95_percent": (
                    content_curve.first_epoch_at_95_percent
                ),
                "content_minus_random_first_epoch_at_95_percent": (
                    content_curve.first_epoch_at_95_percent
                    - random_curve.first_epoch_at_95_percent
                ),
                "random_normalized_auc": random_curve.normalized_auc,
                "content_normalized_auc": content_curve.normalized_auc,
                "content_minus_random_normalized_auc": (
                    content_curve.normalized_auc - random_curve.normalized_auc
                ),
                "content_minus_random_raw_recall_auc": (
                    content_raw_auc - random_raw_auc
                ),
                "content_minus_random_recall_by_epoch": [
                    content - random
                    for random, content in zip(
                        random_curve.recall_at_100,
                        content_curve.recall_at_100,
                        strict=True,
                    )
                ],
                "random_sweep_log_sha256": random_curve.source_sha256,
                "content_sweep_log_sha256": content_curve.source_sha256,
            }
        )
    return summaries


def _initialization_identity(
    confirmation: dict[str, Any],
    logs_root: Path,
    codebook_sha256: str,
) -> dict[str, Any]:
    def seed(row: dict[str, Any]) -> int:
        return row.get("seed", row.get("parameters", {}).get("training_seed"))

    rows_by_mode = {
        mode: {seed(row): row for row in confirmation["rq1"][mode]["rows"]}
        for mode in ("random", "content_pca")
    }
    comparisons = []
    for seed in (43, 44, 45):
        random_metadata, _ = _authenticate_run(rows_by_mode["random"][seed], logs_root)
        content_metadata, _ = _authenticate_run(
            rows_by_mode["content_pca"][seed], logs_root
        )
        random = random_metadata["sid_initialization_diagnostics"]
        content = content_metadata["sid_initialization_diagnostics"]
        random_rms = [level["initialized_rms"] for level in random["levels"]]
        content_rms = [level["initialized_rms"] for level in content["levels"]]
        scale_matches = [
            math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)
            for left, right in zip(random_rms, content_rms, strict=True)
        ]
        checks = {
            "base_rows_before_match": (
                random["base_rows_before_sha256"] == content["base_rows_before_sha256"]
            ),
            "random_base_rows_unchanged": (
                random["base_rows_before_sha256"] == random["base_rows_after_sha256"]
            ),
            "content_base_rows_changed": (
                content["base_rows_before_sha256"] != content["base_rows_after_sha256"]
            ),
            "non_base_rows_match": (
                random["non_base_rows_sha256"] == content["non_base_rows_sha256"]
            ),
            "centroid_hashes_match": (
                random["codebook_centroids_sha256"]
                == content["codebook_centroids_sha256"]
                == codebook_sha256
            ),
            "rng_nonadvancing": (
                random["rng_nonadvancing"] is True
                and content["rng_nonadvancing"] is True
            ),
            "per_level_rms_match": all(scale_matches),
        }
        comparisons.append(
            {
                "seed": seed,
                "checks": checks,
                "random_rms_by_level": random_rms,
                "content_initialized_rms_by_level": content_rms,
                "maximum_absolute_rms_difference": max(
                    abs(left - right)
                    for left, right in zip(random_rms, content_rms, strict=True)
                ),
                "random_base_rows_before_sha256": random["base_rows_before_sha256"],
                "content_base_rows_after_sha256": content["base_rows_after_sha256"],
                "non_base_rows_sha256": content["non_base_rows_sha256"],
                "codebook_centroids_sha256": content["codebook_centroids_sha256"],
            }
        )
    return {
        "paired_seeds": [43, 44, 45],
        "rms_reference": "matched random arm initialized RMS",
        "comparisons": comparisons,
        "all_checks_passed": all(
            all(comparison["checks"].values()) for comparison in comparisons
        ),
    }


def _resolve_codebooks(surface: dict[str, Any]) -> Path:
    cache_key = surface["rq1"]["selected"]["content_pca"]["diagnostics"][
        "semantic_cache_key"
    ]
    matches = sorted(PROJECT_ROOT.glob(CODEBOOK_PATTERN.format(cache_key=cache_key)))
    if len(matches) != 1:
        raise ValueError(f"expected one codebook cache for {cache_key}, got {matches}")
    return matches[0]


def _duplicated_frozen_information(
    codebook_path: Path,
    codebooks: torch.Tensor,
    *,
    learned_width: int,
    initialization: dict[str, Any],
) -> dict[str, Any]:
    codes_path = codebook_path.with_name("codes.pt")
    codes = SemanticCodes.load(codes_path)
    frozen = SemanticIdEmbedding.from_codebooks(
        codes,
        ResidualCodebooks(codebooks),
        num_items=int(codes.item_ids.max()),
        train_collision_suffix=True,
    )
    frozen_weights = next(
        buffer for name, buffer in frozen.named_buffers() if name.endswith("weights")
    )
    frozen_errors = []
    projection_errors = []
    for level, centroids in enumerate(codebooks.double()):
        first, last = codes.vocabulary.level_range(level)
        frozen_errors.append(
            float((frozen_weights[first:last].double() - centroids).abs().max())
        )
        projected = project_centroids_with_pca(centroids, learned_width)
        projected_from_frozen = project_centroids_with_pca(
            frozen_weights[first:last], learned_width
        )
        projection_errors.append(float((projected_from_frozen - projected).abs().max()))
    source_hashes_match = all(
        comparison["codebook_centroids_sha256"] == _tensor_sha256(codebooks)
        for comparison in initialization["comparisons"]
    )
    maximum_frozen_error = max(frozen_errors)
    maximum_projection_error = max(projection_errors)
    frozen_exact = maximum_frozen_error == 0
    deterministic_projection = maximum_projection_error == 0
    if not (frozen_exact and deterministic_projection and source_hashes_match):
        raise ValueError("RQ1 frozen-centroid duplication check failed")
    return {
        "base_levels": int(codebooks.shape[0]),
        "frozen_width": int(codebooks.shape[2]),
        "learned_width": learned_width,
        "codes_path": _relative(codes_path),
        "codes_file_sha256": _sha256(codes_path),
        "frozen_view_tensor_sha256": _tensor_sha256(codebooks),
        "initialization_source_tensor_sha256": _tensor_sha256(codebooks),
        "maximum_frozen_centroid_error": maximum_frozen_error,
        "maximum_projected_frozen_view_error": maximum_projection_error,
        "frozen_base_rows_are_codebook_centroids": frozen_exact,
        "deterministic_linear_projection_of_frozen_view": (
            deterministic_projection and source_hashes_match
        ),
    }


def collect_rq1_unexpected_evidence(
    *,
    surface_path: Path = SURFACE,
    confirmation_path: Path = CONFIRMATION,
    logs_root: Path = LOGS_ROOT,
    codebook_path: Path | None = None,
) -> dict[str, Any]:
    surface, confirmation = _load_sources(surface_path, confirmation_path)
    curves = _paired_curves(surface, logs_root)
    for mode in ("random", "content_pca"):
        for row in confirmation["rq1"][mode]["rows"]:
            _authenticate_run(row, logs_root)
    resolved_codebooks = codebook_path or _resolve_codebooks(surface)
    codebooks = torch.load(resolved_codebooks, map_location="cpu", weights_only=True)
    if not isinstance(codebooks, torch.Tensor):
        raise ValueError("RQ1 codebook cache is not a tensor")
    codebook_sha256 = _tensor_sha256(codebooks)
    initialization = _initialization_identity(confirmation, logs_root, codebook_sha256)
    if not initialization["all_checks_passed"]:
        raise ValueError("RQ1 initialization identity checks failed")
    selected = surface["rq1"]["selected"]["content_pca"]
    learned_width = selected["parameters"]["representation_width"]
    reconstruction = projection_reconstruction(codebooks, output_dim=learned_width)
    duplication = _duplicated_frozen_information(
        resolved_codebooks,
        codebooks,
        learned_width=learned_width,
        initialization=initialization,
    )
    selected_deep_rate = surface["rq1"]["selected"]["random"]["parameters"][
        "deep_learning_rate"
    ]
    fixed_deep = [
        row
        for row in curves
        if math.isclose(
            row["deep_learning_rate"], selected_deep_rate, rel_tol=0, abs_tol=1e-15
        )
    ]
    fixed_deep.sort(key=lambda row: row["embedding_learning_rate"])
    if [row["trial"] for row in fixed_deep] != [0, 6, 7, 8, 9]:
        raise ValueError("RQ1 fixed-deep LR series changed")
    raw_advantages = [row["content_minus_random_raw_recall_auc"] for row in fixed_deep]
    monotonic_erasure = all(
        left >= right for left, right in zip(raw_advantages, raw_advantages[1:])
    )
    return {
        "schema": "g6-rq1-unexpected/v1",
        "dataset_size": "native-50m",
        "source_artifacts": {
            "surface": {
                "path": _relative(surface_path),
                "sha256": _sha256(surface_path),
            },
            "confirmation": {
                "path": _relative(confirmation_path),
                "sha256": _sha256(confirmation_path),
            },
            "codebooks": {
                "path": _relative(resolved_codebooks),
                "file_sha256": _sha256(resolved_codebooks),
                "tensor_sha256": codebook_sha256,
            },
            "authenticated_surface_rq1_runs": 32,
            "authenticated_confirmation_rq1_runs": 8,
        },
        "initialization_identity": initialization,
        "projection_reconstruction": {
            "input_width": int(codebooks.shape[2]),
            "output_width": learned_width,
            "levels": reconstruction,
        },
        "duplicated_frozen_centroid_information": duplication,
        "lr_warm_start": {
            "paired_curves": curves,
            "fixed_deep_learning_rate": selected_deep_rate,
            "fixed_deep_trials": [row["trial"] for row in fixed_deep],
            "fixed_deep_raw_auc_advantage_by_embedding_lr": [
                {
                    "embedding_learning_rate": row["embedding_learning_rate"],
                    "content_minus_random_raw_recall_auc": row[
                        "content_minus_random_raw_recall_auc"
                    ],
                }
                for row in fixed_deep
            ],
            "monotonic_erasure_criterion": (
                "content-minus-random raw Recall AUC is nonincreasing across "
                "the fixed-deep embedding-LR series"
            ),
            "monotonic_erasure_supported": monotonic_erasure,
        },
        "gradient_probe": gradient_probe(),
    }


def write_evidence(document: dict[str, Any], output: Path = OUTPUT) -> None:
    content = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise RuntimeError(f"evidence already differs: {output}")
    output.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    write_evidence(collect_rq1_unexpected_evidence(), arguments.output)


if __name__ == "__main__":
    main()
