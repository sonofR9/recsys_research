"""Reusable layers, in the spirit of ``torch.nn``."""

from .layer_registry import layer_registry, build_layer
from .crossnet import CrossHeadDescription, CrossNetwork
from .dcnv2 import DcnV2
from .densenet import DenseNet
from .esasrec import LiGRBlock, SASRecBlock
from .ffn import GEGLU, ReGLU, RegularMLP, SwiGLU
from .layer_item_features import (
    ConcatenatedItemFeatureResidual,
    DirectAddItemFeature,
    GemmaItemFeatureResidual,
    LayerItemFeatureFusion,
)
from .multi_task_embedding import (
    BaseMultiTaskEmbeddingLayer,
    MultiTaskEmbeddingLayer,
    MultiTaskEmbeddingLayerTorchRec,
)
from .precomputed_embeddings import PrecomputedEmbeddingLookup
from .resnet import ResNet1D
from .sampled_softmax import (
    GeneralizedBCELoss,
    InBatchSampledSoftmaxLoss,
    OfflineInBatchSoftmax,
    RandomCatalogNegatives,
    StreamingInBatchSoftmax,
)
from .types import ModuleWithDim, OutputDims

__all__ = [
    "CrossHeadDescription",
    "CrossNetwork",
    "DcnV2",
    "DenseNet",
    "LiGRBlock",
    "SASRecBlock",
    "GEGLU",
    "ReGLU",
    "RegularMLP",
    "SwiGLU",
    "ConcatenatedItemFeatureResidual",
    "DirectAddItemFeature",
    "GemmaItemFeatureResidual",
    "LayerItemFeatureFusion",
    "GeneralizedBCELoss",
    "InBatchSampledSoftmaxLoss",
    "OfflineInBatchSoftmax",
    "RandomCatalogNegatives",
    "StreamingInBatchSoftmax",
    "BaseMultiTaskEmbeddingLayer",
    "MultiTaskEmbeddingLayer",
    "MultiTaskEmbeddingLayerTorchRec",
    "PrecomputedEmbeddingLookup",
    "ResNet1D",
    "ModuleWithDim",
    "OutputDims",
    "layer_registry",
    "build_layer",
]
