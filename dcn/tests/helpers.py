import torch
from torch import nn

from collections.abc import Sequence

from dcn.data.features import FeatureValues
from dcn.models.two_tower import Tower, TowerInputEncoder, TwoTowerLoss, TwoTowerModel
from dcn.nn.ffn import SwiGLU
from dcn.nn.resnet import ResNet1D
from dcn.nn.sampled_softmax import StreamingInBatchSoftmax
from dcn.semantic import SemanticCodes
from dcn.nn.transformer import (
    ReverseRelativePositionInput,
    TransformerBlock,
    TransformerEncoder,
)

ITEM_COLUMN = "compact_item_id"
ACTION_COLUMN = "event_type_id"


class TinyFFN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_weights(self, in_range: float, out_range: float) -> None:
        nn.init.trunc_normal_(self.linear.weight, std=in_range)
        nn.init.zeros_(self.linear.bias)


def packed_lens(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )


def packed_batch(
    item_ids: list[int], lengths: list[int], actions: list[int] | None = None
) -> dict:
    """A collated batch of packed sequences carrying nothing but item ids."""
    int_columns = {ITEM_COLUMN: scalar_feature(torch.tensor(item_ids))}
    if actions is not None:
        int_columns[ACTION_COLUMN] = scalar_feature(torch.tensor(actions))
    return {
        "int_columns": int_columns,
        "float_columns": {},
        "cumulative_lens": packed_lens(lengths),
        "timestamp": torch.arange(len(item_ids), dtype=torch.int64),
    }


class WidenCounters(nn.Module):
    """Stand-in for the fitted piecewise-linear encoder used in real training."""

    def __init__(self, num_counters: int, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_counters, num_counters * width)

    @property
    def out_dim(self) -> int:
        return self.linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def scalar_feature(values: torch.Tensor) -> FeatureValues:
    return FeatureValues(
        values=values,
        offsets=torch.arange(
            values.shape[0] + 1, dtype=torch.int64, device=values.device
        ),
    )


def tiny_encoder(
    dim: int, num_layers: int = 1, nhead: int = 2, num_kv_heads: int = 1
) -> TransformerEncoder:
    return TransformerEncoder(
        blocks=[
            TransformerBlock(
                dim=dim,
                nhead=nhead,
                num_kv_heads=num_kv_heads,
                ffn_factory=TinyFFN,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ],
        final_norm=nn.LayerNorm(dim),
    )


CATEGORICAL = ["compact_item_id", "artist_id"]
HISTORY_COUNTERS = ["history_count"]
ITEM_COUNTERS = ["item_count"]
MODEL_DIM = 16
EMBEDDING_DIM = 8
COUNTER_ENCODER_WIDTH = 4


def tower_encoder(
    counter_columns: Sequence[str], *, num_hashes: int = 1
) -> TowerInputEncoder:
    return TowerInputEncoder(
        num_embeddings=64,
        embedding_dim=EMBEDDING_DIM,
        categorical_columns=CATEGORICAL,
        body_factory=lambda input_dim: ResNet1D(
            input_dim=input_dim,
            hidden_dims=[16, MODEL_DIM],
            norm_factory=nn.LayerNorm,
            dropout=0.0,
        ),
        counter_encoder=(
            WidenCounters(len(counter_columns), COUNTER_ENCODER_WIDTH)
            if counter_columns
            else None
        ),
        num_hashes=num_hashes,
    )


def two_tower_model(
    item_counter_columns: Sequence[str] = ITEM_COUNTERS, *, num_hashes: int = 1
) -> TwoTowerModel:
    """A query tower reading a packed history and an item tower scoring rows."""
    return TwoTowerModel(
        Tower(
            tower_encoder(HISTORY_COUNTERS, num_hashes=num_hashes),
            categorical_columns=CATEGORICAL,
            counter_columns=HISTORY_COUNTERS,
            sequence_model=TransformerEncoder(
                blocks=[
                    TransformerBlock(
                        dim=MODEL_DIM,
                        nhead=2,
                        num_kv_heads=2,
                        ffn_factory=lambda dim: SwiGLU(dim, 32),
                        dropout=0.0,
                        use_alibi=False,
                    )
                    for _ in range(2)
                ],
                final_norm=nn.LayerNorm(MODEL_DIM),
                position_inputs=[ReverseRelativePositionInput(MODEL_DIM, 16)],
            ),
        ),
        Tower(
            tower_encoder(item_counter_columns, num_hashes=num_hashes),
            categorical_columns=CATEGORICAL,
            counter_columns=item_counter_columns,
        ),
        item_id_column=ITEM_COLUMN,
    )


def two_tower_loss(
    model: TwoTowerModel, *, hash_size: int = 64, num_in_batch_negatives: int = 4
) -> TwoTowerLoss:
    return TwoTowerLoss(
        model,
        StreamingInBatchSoftmax(
            hash_size=hash_size,
            num_in_batch_negatives=num_in_batch_negatives,
            alpha=0.05,
        ),
    )


# Four items over a 2x2 code space, so every tuple in the space exists.
CODES = SemanticCodes(
    item_ids=torch.tensor([1, 2, 3, 4]),
    codes=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
    codes_per_level=(2, 2),
)
