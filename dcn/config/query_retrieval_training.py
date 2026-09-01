from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Literal

import torch

from dcn.models.cross_attention_training import (
    AuxiliaryNtpCrossAttentionRetrievalModel,
    CandidateAuxiliaryNtpLoss,
    FirstStageCheckpointError,
    first_stage_initialization_manifest,
    load_first_stage_checkpoint,
    save_first_stage_checkpoint,
)
from dcn.models.two_tower import TwoTowerLoss
from neuralrec.run.train import TrainRunner

from .generation import MuTransferGenerationExperiment
from .query_retrieval import MuTransferCrossAttentionGenerationExperiment


Rq15TrainingMethod = Literal[
    "scratch_candidate_only", "pretrained_finetune", "auxiliary_ntp"
]


@dataclass
class MuTransferRq15CrossAttentionGenerationExperiment(
    MuTransferCrossAttentionGenerationExperiment
):
    training_method: Rq15TrainingMethod = "auxiliary_ntp"
    first_stage_checkpoint: Path | None = None
    first_stage_checkpoint_metadata: dict[str, object] = field(default_factory=dict)
    auxiliary_ntp_weight: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.training_method not in {
            "scratch_candidate_only",
            "pretrained_finetune",
            "auxiliary_ntp",
        }:
            raise ValueError("unknown RQ15 training method")
        if (
            self.query_architecture != "decoder_decoder"
            or self.query_slots_shared
            or self.include_history_memory
        ):
            raise ValueError("RQ15 uses four distinct CLS slots and CLS-only memory")
        if self.training_method == "pretrained_finetune":
            if self.first_stage_checkpoint is None:
                raise ValueError("pretrained fine-tuning requires a first-stage checkpoint")
            if self.auxiliary_ntp_weight != 0:
                raise ValueError("pretrained fine-tuning is downstream-only")
        elif self.first_stage_checkpoint is not None:
            raise ValueError("scratch training cannot load a checkpoint")
        elif (
            self.training_method == "scratch_candidate_only"
            and self.auxiliary_ntp_weight != 0
        ):
            raise ValueError("scratch candidate-only training has no NTP loss")
        elif self.training_method == "auxiliary_ntp" and self.auxiliary_ntp_weight <= 0:
            raise ValueError("auxiliary NTP weight must be positive")

    def _create_model(self):
        model = super()._create_model()
        if self.training_method != "auxiliary_ntp":
            return model
        first_stage_projection = self.create_query_projection()
        return AuxiliaryNtpCrossAttentionRetrievalModel(
            tokenizer=model.tokenizer,
            memory_encoder=model.memory_encoder,
            decoder=model.decoder,
            item_embedding=model.item_embedding,
            item_id_column=model.item_id_column,
            query_projection=model.query_projection,
            query_slots=model.query_slots,
            include_history_memory=model.include_history_memory,
            query_multiplier=model.query_multiplier,
            first_stage_query_projection=first_stage_projection,
        )

    @cached_property
    def base_model(self):
        expected_initialization = self.first_stage_initialization
        model = super().base_model
        if self.training_method == "pretrained_finetune":
            assert self.first_stage_checkpoint is not None
            actual_initialization = load_first_stage_checkpoint(
                model,
                self.first_stage_checkpoint,
                expected_metadata=self.first_stage_checkpoint_metadata,
                history_position_count=self.max_seq_len,
            )
            if actual_initialization != expected_initialization:
                raise FirstStageCheckpointError(
                    "first-stage checkpoint changed after initialization was selected"
                )
            self.__dict__["first_stage_load_report"] = actual_initialization
        return model

    @cached_property
    def first_stage_initialization(self) -> str | dict[str, object]:
        if self.training_method != "pretrained_finetune":
            return "scratch"
        assert self.first_stage_checkpoint is not None
        return first_stage_initialization_manifest(
            self.first_stage_checkpoint,
            source_metadata=self.first_stage_checkpoint_metadata,
            history_position_count=self.max_seq_len,
        )

    def create_criterion(self):
        candidate = super().create_criterion()
        if self.training_method != "auxiliary_ntp":
            return candidate
        auxiliary = super().create_criterion()
        if not isinstance(candidate, TwoTowerLoss) or not isinstance(
            auxiliary, TwoTowerLoss
        ):
            raise TypeError("RQ15 requires sampled-softmax retrieval losses")
        if not isinstance(self.base_model, AuxiliaryNtpCrossAttentionRetrievalModel):
            raise TypeError("RQ15 auxiliary recipe requires the NTP-capable model")
        return CandidateAuxiliaryNtpLoss(
            self.base_model,
            candidate_loss=candidate.loss,
            auxiliary_ntp_loss=auxiliary.loss,
            auxiliary_ntp_weight=self.auxiliary_ntp_weight,
        )

    @cached_property
    def auxiliary_ntp_targets_per_epoch(self) -> int:
        if self.training_method != "auxiliary_ntp":
            return 0
        dataset = self.sequence_train_loader.dataset
        return dataset.event_count - len(dataset)

    def _training_counts(self) -> tuple[int, int]:
        candidate_targets, input_tokens = super()._training_counts()
        return candidate_targets + self.auxiliary_ntp_targets_per_epoch, input_tokens

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata = super().generation_architecture_metadata()
        examples = len(self.sequence_train_loader.dataset)
        metadata.update(
            {
                "training_method": self.training_method,
                "candidate_targets_per_epoch": examples,
                "ntp_targets_per_epoch": self.auxiliary_ntp_targets_per_epoch,
                "auxiliary_ntp_weight": self.auxiliary_ntp_weight,
                "loss_normalization": "candidate_and_ntp_separately_mean_normalized",
                "first_stage_initialization": self.first_stage_initialization,
            }
        )
        return metadata

    def training_count_architecture_invariants(self) -> tuple[object, ...]:
        return (
            *super().training_count_architecture_invariants(),
            self.training_method,
            self.auxiliary_ntp_weight,
        )


@dataclass
class MuTransferRq14PretrainedCrossAttentionGenerationExperiment(
    MuTransferRq15CrossAttentionGenerationExperiment
):
    training_method: Rq15TrainingMethod = "pretrained_finetune"
    auxiliary_ntp_weight: float = 0.0

    def __post_init__(self) -> None:
        MuTransferCrossAttentionGenerationExperiment.__post_init__(self)
        if self.query_architecture != "decoder_decoder" or self.num_query_slots != 4:
            raise ValueError("pretrained RQ14 uses four end slots and two decoders")
        if self.training_method != "pretrained_finetune":
            raise ValueError("pretrained RQ14 requires NTP initialization")
        if self.first_stage_checkpoint is None:
            raise ValueError("pretrained RQ14 requires a first-stage checkpoint")
        if self.auxiliary_ntp_weight != 0:
            raise ValueError("pretrained RQ14 fine-tuning is candidate-only")


def query_change_diagnostics(
    reference_user_ids: torch.Tensor,
    reference_queries: torch.Tensor,
    lesion_user_ids: torch.Tensor,
    lesion_queries: torch.Tensor,
) -> dict[str, float | int]:
    if not torch.equal(reference_user_ids, lesion_user_ids):
        raise ValueError("lesion query user identity or order changed")
    if (
        reference_queries.shape != lesion_queries.shape
        or reference_queries.ndim != 2
    ):
        raise ValueError("lesion query shape changed")
    if reference_queries.shape[0] != reference_user_ids.shape[0]:
        raise ValueError("query rows do not align with users")
    if reference_user_ids.numel() == 0:
        raise ValueError("query diagnostics require at least one user")
    if not bool(torch.isfinite(reference_queries).all()) or not bool(
        torch.isfinite(lesion_queries).all()
    ):
        raise ValueError("query representations must be finite")
    differences = (lesion_queries - reference_queries).float()
    l2 = torch.linalg.vector_norm(differences, dim=1)
    reference_l2 = torch.linalg.vector_norm(reference_queries.float(), dim=1)
    cosine = torch.nn.functional.cosine_similarity(
        reference_queries.float(), lesion_queries.float(), dim=1, eps=1e-12
    )
    return {
        "num_users": int(reference_user_ids.shape[0]),
        "changed_user_fraction": float((l2 > 1e-8).float().mean()),
        "mean_l2_change": float(l2.mean()),
        "max_l2_change": float(l2.max()),
        "mean_relative_l2_change": float((l2 / reference_l2.clamp_min(1e-12)).mean()),
        "mean_cosine_distance": float((1.0 - cosine).clamp_min(0).mean()),
    }


@dataclass
class MuTransferRq14LesionDiagnosticExperiment(
    MuTransferRq14PretrainedCrossAttentionGenerationExperiment
):
    diagnostic_lesions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        expected = tuple(
            ["remove_history"] if self.include_history_memory else []
        ) + tuple(f"drop_cls_{slot}" for slot in range(self.num_query_slots))
        if self.diagnostic_lesions != expected:
            raise ValueError("RQ14 diagnostic lesion set is incomplete or reordered")

    def generation_architecture_metadata(self) -> dict[str, object]:
        metadata = super().generation_architecture_metadata()
        metadata.update(
            {
                "diagnostic_protocol": "rq14_selected_cell_inference_lesions_v1",
                "diagnostic_lesions": list(self.diagnostic_lesions),
                "diagnostic_full_user_evaluation": True,
                "diagnostic_after_best_checkpoint_restore": True,
            }
        )
        return metadata

    def finish(self, runner: TrainRunner) -> None:
        super().finish(runner)
        if not self.callbacks.best_weights.restore(runner.model):
            raise RuntimeError("RQ14 diagnostics require a restorable best checkpoint")
        reference_user_ids, reference_queries = (
            self.true_metric.full_user_query_snapshot()
        )
        normal_metrics = self._required_diagnostic_metrics(
            json.loads(
                (
                    Path(self.base_path) / "logs" / self.run_name / "final_metrics.json"
                ).read_text()
            )
        )
        lesions: dict[str, object] = {}
        for name in self.diagnostic_lesions:
            remove_history = name == "remove_history"
            dropped_slot = (
                None if remove_history else int(name.removeprefix("drop_cls_"))
            )
            with self.base_model.inference_memory_lesion(
                remove_history=remove_history,
                dropped_query_slot=dropped_slot,
            ):
                metrics = self.true_metric.score(max_users=None)
                if metrics is None:
                    raise RuntimeError(f"{name}: no full-user diagnostic metrics")
                user_ids, queries = self.true_metric.full_user_query_snapshot()
            lesions[name] = {
                "metrics": self._required_diagnostic_metrics(metrics),
                "query_change": query_change_diagnostics(
                    reference_user_ids,
                    reference_queries,
                    user_ids,
                    queries,
                ),
            }
        initialization = self.first_stage_initialization
        if not isinstance(initialization, dict):
            raise RuntimeError("RQ14 diagnostics require checkpoint initialization")
        document = {
            "schema_version": 1,
            "run_name": self.run_name,
            "dataset_size": "500m",
            "treatment": {
                "query_slots_shared": self.query_slots_shared,
                "include_history_memory": self.include_history_memory,
            },
            "training_protocol": {
                "seed": self.seed,
                "effective_batch_size": self.dataloader.effective_batch_size,
                "embedding_learning_rate": self.embedding_learning_rate,
                "deep_learning_rate": self.deep_learning_rate,
                "horizon_epochs": self.num_epochs,
                "lr_schedule": self.lr_schedule.shape,
                "lr_schedule_optimizer_group_scope": (
                    self.lr_schedule.optimizer_group_scope
                ),
                "training_method": self.training_method,
                "auxiliary_ntp_weight": self.auxiliary_ntp_weight,
                "source_checkpoint_sha256": initialization["checkpoint_sha256"],
                "best_checkpoint_restored": True,
            },
            "best_model_state_sha256": self._model_state_sha256(self.base_model),
            "normal_metrics": normal_metrics,
            "lesions": lesions,
        }
        self._write_diagnostic_json(document)

    @staticmethod
    def _required_diagnostic_metrics(metrics: object) -> dict[str, float]:
        if not isinstance(metrics, dict):
            raise RuntimeError("RQ14 diagnostic metrics must be an object")
        required = (
            "recall@10",
            "ndcg@10",
            "recall@100",
            "ndcg@100",
            "coverage@100",
        )
        result: dict[str, float] = {}
        for name in required:
            value = metrics.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise RuntimeError(f"RQ14 diagnostic metric {name} is absent or invalid")
            result[name] = float(value)
        return result

    @staticmethod
    def _model_state_sha256(model: torch.nn.Module) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _write_diagnostic_json(self, document: dict[str, object]) -> None:
        directory = Path(self.base_path) / "logs" / self.run_name
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(directory / "rq14_lesion_diagnostics.json")


@dataclass
class FirstStageCheckpointExportExperiment(MuTransferGenerationExperiment):
    checkpoint_export_metadata: dict[str, object] = field(default_factory=dict)
    checkpoint_export_history_positions: int = 128
    checkpoint_export_filename: str = "rq15_first_stage_checkpoint.pt"

    def finish(self, runner: TrainRunner) -> None:
        super().finish(runner)
        destination = (
            Path(self.base_path)
            / "logs"
            / self.run_name
            / self.checkpoint_export_filename
        )
        save_first_stage_checkpoint(
            self.base_model,
            destination,
            metadata=self.checkpoint_export_metadata,
            history_position_count=self.checkpoint_export_history_positions,
        )
