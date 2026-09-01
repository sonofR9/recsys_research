"""The architectures this project trains, assembled from :mod:`dcn.nn` blocks."""

from .criterions import CriterionSpec, MultiCriterion, TargetExtractionWrapper
from .cross_attention_retrieval import CrossAttentionRetrievalModel
from .history_tokens import (
    ActionTokenizer,
    EventTokenizer,
    BosTokenizer,
    ItemTokenizer,
    SemanticHistoryTokenizer,
    SemanticIdTokenizer,
    TimestampDeltaTokenizer,
    TokenizedHistory,
)
from .loss_wrapper import LossWrapper
from .multi_head_network import MultiHeadNetwork
from .semantic_constraint import SemanticIdConstraint
from .sequence_retrieval import ClsTokenMode, SequenceRetrievalModel
from .sequence_targets import (
    NextItemTargets,
    SequenceTargets,
    TargetPairs,
    TimeWindowTargets,
)
from .token_generation import (
    CausalTokenDecoder,
    Seq2SeqTokenDecoder,
    TokenConstraint,
    TokenDecoder,
    TokenPredictionLoss,
)
from .two_tower import Tower, TowerInputEncoder, TwoTowerLoss, TwoTowerModel

__all__ = [
    "ActionTokenizer",
    "CausalTokenDecoder",
    "ClsTokenMode",
    "CrossAttentionRetrievalModel",
    "CriterionSpec",
    "EventTokenizer",
    "BosTokenizer",
    "ItemTokenizer",
    "LossWrapper",
    "MultiCriterion",
    "MultiHeadNetwork",
    "NextItemTargets",
    "Seq2SeqTokenDecoder",
    "SemanticIdConstraint",
    "SemanticHistoryTokenizer",
    "SemanticIdTokenizer",
    "SequenceRetrievalModel",
    "SequenceTargets",
    "TargetExtractionWrapper",
    "TargetPairs",
    "TimeWindowTargets",
    "TimestampDeltaTokenizer",
    "TokenConstraint",
    "TokenDecoder",
    "TokenPredictionLoss",
    "TokenizedHistory",
    "Tower",
    "TowerInputEncoder",
    "TwoTowerLoss",
    "TwoTowerModel",
]
