from pathlib import Path

import pytest
import torch
from torch import nn

from dcn.models.cross_attention_retrieval import CrossAttentionRetrievalModel
from dcn.models.cross_attention_training import (
    AuxiliaryNtpCrossAttentionRetrievalModel,
    CandidateAuxiliaryNtpLoss,
    FirstStageCheckpointError,
    first_stage_initialization_manifest,
    load_first_stage_checkpoint,
    save_first_stage_checkpoint,
)
from dcn.models.history_tokens import EndQuerySlots, ItemTokenizer
from dcn.models.sequence_retrieval import SequenceRetrievalModel
from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.sampled_softmax import InBatchSampledSoftmaxLoss
from dcn.nn.types import ModuleWithDim
from dcn.tests.helpers import ITEM_COLUMN, packed_batch
from neuralrec.utils import LOSS_DENOMINATOR


DIM = 8
NUM_ITEMS = 32


class LinearSequence(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)

    @property
    def out_dim(self) -> int:
        return DIM

    def forward(
        self,
        embeddings: torch.Tensor,
        cumulative_lens: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.linear(embeddings)


class MemoryMeanDecoder(ModuleWithDim):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(DIM, DIM)

    @property
    def out_dim(self) -> int:
        return DIM

    def forward(
        self,
        query: torch.Tensor,
        query_cumulative_lens: torch.Tensor,
        memory: torch.Tensor,
        memory_cumulative_lens: torch.Tensor,
    ) -> torch.Tensor:
        means = [
            memory[start:end].mean(dim=0)
            for start, end in zip(
                memory_cumulative_lens[:-1], memory_cumulative_lens[1:]
            )
        ]
        return self.linear(query + torch.stack(means))


class MeanSquaredPairLoss(InBatchSampledSoftmaxLoss):
    def __init__(self) -> None:
        super().__init__(num_in_batch_negatives=0)

    def _log_q(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(ids, dtype=torch.float32)

    def _q_total(self) -> torch.Tensor:
        return torch.tensor(1.0)

    def logits(
        self,
        query_repr: torch.Tensor,
        positive_item_repr: torch.Tensor,
        positive_item_ids: torch.Tensor,
        group_sizes: torch.Tensor,
        negatives=None,
    ) -> torch.Tensor:
        del positive_item_ids, group_sizes, negatives
        squared = (query_repr - positive_item_repr).square().mean(dim=1)
        return torch.stack([-squared, torch.zeros_like(squared)], dim=1)

    def loss_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return -logits[:, 0].mean()


def _auxiliary_model() -> AuxiliaryNtpCrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(NUM_ITEMS, DIM)
    return AuxiliaryNtpCrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        memory_encoder=LinearSequence(),
        decoder=MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        query_slots=EndQuerySlots(DIM, num_slots=4, shared=False),
        include_history_memory=False,
        first_stage_query_projection=nn.Linear(DIM, DIM, bias=False),
    )


def _source_model() -> SequenceRetrievalModel:
    item_embedding = nn.Embedding(NUM_ITEMS, DIM)
    return SequenceRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        sequence_model=LinearSequence(),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        query_projection=nn.Linear(DIM, DIM, bias=False),
    )


def _target_model() -> CrossAttentionRetrievalModel:
    item_embedding = nn.Embedding(NUM_ITEMS, DIM)
    return CrossAttentionRetrievalModel(
        tokenizer=ItemTokenizer(item_embedding, item_id_column=ITEM_COLUMN),
        memory_encoder=LinearSequence(),
        decoder=MemoryMeanDecoder(),
        item_embedding=item_embedding,
        item_id_column=ITEM_COLUMN,
        query_projection=nn.Linear(DIM, DIM, bias=False),
        query_slots=EndQuerySlots(DIM, num_slots=4, shared=False),
        include_history_memory=False,
    )


def test_auxiliary_ntp_pairs_cover_each_history_transition_and_final_target() -> None:
    model = _auxiliary_model()
    batch = packed_batch([1, 2, 3, 4, 10, 11, 12], [4, 3])

    candidate_output, ntp_output = model.forward_training_tasks(batch)
    candidate_pairs = NextItemTargets()(candidate_output)
    ntp_pairs = NextItemTargets()(ntp_output)

    assert candidate_pairs.positive_ids.tolist() == [4, 12]
    assert candidate_pairs.group_sizes.tolist() == [1, 1]
    assert ntp_pairs.positive_ids.tolist() == [2, 3, 4, 11, 12]
    assert ntp_pairs.group_sizes.tolist() == [3, 2]


def test_joint_loss_normalizes_candidate_and_ntp_targets_separately() -> None:
    model = _auxiliary_model()
    criterion = CandidateAuxiliaryNtpLoss(
        model,
        candidate_loss=MeanSquaredPairLoss(),
        auxiliary_ntp_loss=MeanSquaredPairLoss(),
        auxiliary_ntp_weight=1.0,
    )
    batch = packed_batch([1, 2, 3, 4, 10, 11, 12], [4, 3])

    output = criterion(batch)

    torch.testing.assert_close(
        output["loss"], output["candidate_loss"] + output["auxiliary_ntp_loss"]
    )
    assert output["candidate_targets"] == 2
    assert output["auxiliary_ntp_targets"] == 5
    assert output[LOSS_DENOMINATOR] == 2
    assert criterion.accumulation_spec(batch) == {
        "candidate_loss": (1.0, 2),
        "auxiliary_ntp_loss": (1.0, 5),
    }


def test_joint_loss_reaches_both_decoders_every_slot_and_history() -> None:
    model = _auxiliary_model()
    criterion = CandidateAuxiliaryNtpLoss(
        model,
        candidate_loss=MeanSquaredPairLoss(),
        auxiliary_ntp_loss=MeanSquaredPairLoss(),
        auxiliary_ntp_weight=1.0,
    )

    criterion(packed_batch([1, 2, 3, 4, 10, 11, 12], [4, 3]))["loss"].backward()

    assert model.query_slots is not None
    assert model.query_slots.embeddings.grad is not None
    assert model.query_slots.embeddings.grad.abs().sum(dim=1).gt(0).tolist() == [
        True
    ] * 4
    assert model.memory_encoder.linear.weight.grad is not None
    assert model.memory_encoder.linear.weight.grad.abs().sum() > 0
    assert model.decoder.linear.weight.grad is not None
    assert model.decoder.linear.weight.grad.abs().sum() > 0
    assert model.first_stage_query_projection.weight.grad is not None
    assert model.first_stage_query_projection.weight.grad.abs().sum() > 0


def test_checkpoint_load_copies_only_the_first_stage(tmp_path: Path) -> None:
    torch.manual_seed(1)
    source = _source_model()
    torch.manual_seed(2)
    target = _target_model()
    untouched = {
        "slots": target.query_slots.embeddings.detach().clone(),
        "decoder": target.decoder.linear.weight.detach().clone(),
        "readout": target.query_projection.weight.detach().clone(),
    }
    checkpoint = tmp_path / "first_stage.pt"
    metadata = {
        "dataset_size": "500m",
        "source_recipe_run_name": "rq8-standard",
    }
    save_first_stage_checkpoint(
        source,
        checkpoint,
        metadata=metadata,
        history_position_count=128,
    )
    expected_initialization = first_stage_initialization_manifest(
        checkpoint,
        source_metadata=metadata,
        history_position_count=128,
    )

    report = load_first_stage_checkpoint(
        target,
        checkpoint,
        expected_metadata=metadata,
        history_position_count=128,
    )

    torch.testing.assert_close(
        target.item_embedding.weight, source.item_embedding.weight
    )
    torch.testing.assert_close(
        target.memory_encoder.linear.weight, source.sequence_model.linear.weight
    )
    torch.testing.assert_close(target.query_slots.embeddings, untouched["slots"])
    torch.testing.assert_close(target.decoder.linear.weight, untouched["decoder"])
    torch.testing.assert_close(target.query_projection.weight, untouched["readout"])
    assert report["copied_modules"] == [
        "item_embedding",
        "memory_encoder",
        "tokenizer",
    ]
    assert report["history_position_count"] == 128
    assert report == expected_initialization


def test_checkpoint_load_rejects_wrong_recipe(tmp_path: Path) -> None:
    checkpoint = tmp_path / "first_stage.pt"
    save_first_stage_checkpoint(
        _source_model(),
        checkpoint,
        metadata={"dataset_size": "500m", "source_recipe_run_name": "wrong"},
        history_position_count=128,
    )

    with pytest.raises(FirstStageCheckpointError, match="source_recipe_run_name"):
        load_first_stage_checkpoint(
            _target_model(),
            checkpoint,
            expected_metadata={
                "dataset_size": "500m",
                "source_recipe_run_name": "rq8-standard",
            },
            history_position_count=128,
        )
