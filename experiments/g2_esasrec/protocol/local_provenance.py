from __future__ import annotations

import hashlib
from pathlib import Path


LOCAL_EXECUTION_SOURCES = {
    "entry_and_protocol": (
        "dcn/__init__.py",
        "dcn/main.py",
        "experiments/g2_esasrec/__init__.py",
        "experiments/g2_esasrec/configs/__init__.py",
        "experiments/generation_protocol.py",
        "experiments/g2_esasrec/configs/local.py",
        "experiments/g2_esasrec/launchers/compiled.py",
        "experiments/g2_esasrec/launchers/cost.py",
        "experiments/g2_esasrec/launchers/run_local.py",
        "experiments/g2_esasrec/launchers/__init__.py",
        "experiments/g2_esasrec/protocol/__init__.py",
        "experiments/g2_esasrec/protocol/local_provenance.py",
        "experiments/g2_esasrec/protocol/manifest.py",
    ),
    "configuration": (
        "dcn/config/__init__.py",
        "dcn/config/experiment.py",
        "dcn/config/generation.py",
        "dcn/config/networks.py",
        "dcn/config/ranking.py",
        "dcn/config/retrieval.py",
        "dcn/config/sasrec.py",
        "dcn/config/semantic.py",
        "dcn/config/sequence.py",
        "dcn/config/settings.py",
        "dcn/config/yambda.py",
        "dcn/config/yambda_base.py",
        "dcn/training_metadata.py",
    ),
    "data_and_dataset": (
        "data/__init__.py",
        "data/counters/__init__.py",
        "data/counters/config.py",
        "data/counters/counter.py",
        "data/preprocessing.py",
        "data/split_by_day.py",
        "data/utils.py",
        "dcn/data/dataset.py",
        "dcn/data/__init__.py",
        "dcn/data/dataset_manager.py",
        "dcn/data/features.py",
        "dcn/data/packed.py",
        "dcn/data/sequence_dataset.py",
        "dcn/datasets/base.py",
        "dcn/datasets/__init__.py",
        "dcn/datasets/remap.py",
        "dcn/datasets/yambda.py",
    ),
    "model_targets_and_loss": (
        "dcn/models/criterions.py",
        "dcn/models/__init__.py",
        "dcn/models/history_tokens.py",
        "dcn/models/loss_wrapper.py",
        "dcn/models/multi_head_network.py",
        "dcn/models/semantic_constraint.py",
        "dcn/models/sequence_retrieval.py",
        "dcn/models/sequence_targets.py",
        "dcn/models/two_tower.py",
        "dcn/models/token_generation.py",
        "dcn/nn/crossnet.py",
        "dcn/nn/dcnv2.py",
        "dcn/nn/densenet.py",
        "dcn/nn/esasrec.py",
        "dcn/nn/__init__.py",
        "dcn/nn/ffn.py",
        "dcn/nn/history_encoder.py",
        "dcn/nn/layer_item_features.py",
        "dcn/nn/layer_registry.py",
        "dcn/nn/multi_task_embedding.py",
        "dcn/nn/ple.py",
        "dcn/nn/precomputed_embeddings.py",
        "dcn/nn/resnet.py",
        "dcn/nn/sampled_softmax.py",
        "dcn/nn/semantic_embedding.py",
        "dcn/nn/transformer.py",
        "dcn/nn/types.py",
        "dcn/semantic/__init__.py",
        "dcn/semantic/artifacts.py",
        "dcn/semantic/codes.py",
        "dcn/semantic/residual_kmeans.py",
        "dcn/semantic/rq_vae.py",
        "dcn/semantic/trie.py",
    ),
    "evaluation": (
        "dcn/eval/__init__.py",
        "dcn/eval/base.py",
        "dcn/eval/callback.py",
        "dcn/eval/generation.py",
        "dcn/eval/pairwise.py",
        "dcn/eval/ranking_metrics.py",
        "dcn/eval/true_metric.py",
    ),
    "training_and_optimizer": (
        "dcn/training/__init__.py",
        "dcn/training/combined_optimizer.py",
        "dcn/training/epoch_trainer.py",
        "dcn/training/optimizer_groups.py",
        "dcn/training/pretrain_callback.py",
        "dcn/training/trainer.py",
        "neuralrec/__init__.py",
        "neuralrec/data/dataloader.py",
        "neuralrec/data/transforms.py",
        "neuralrec/data/__init__.py",
        "neuralrec/nn/autocast.py",
        "neuralrec/nn/__init__.py",
        "neuralrec/nn/metrics/__init__.py",
        "neuralrec/nn/metrics/base.py",
        "neuralrec/nn/metrics/classification.py",
        "neuralrec/nn/metrics/recall.py",
        "neuralrec/nn/metrics/regression.py",
        "neuralrec/run/callbacks/__init__.py",
        "neuralrec/run/callbacks/base.py",
        "neuralrec/run/callbacks/best_weights.py",
        "neuralrec/run/callbacks/checkpoint.py",
        "neuralrec/run/callbacks/clipping.py",
        "neuralrec/run/callbacks/early_stopping.py",
        "neuralrec/run/callbacks/logging.py",
        "neuralrec/run/callbacks/lr_schedule.py",
        "neuralrec/run/callbacks/resources.py",
        "neuralrec/run/callbacks/tensorboard.py",
        "neuralrec/run/callbacks/validation.py",
        "neuralrec/run/callbacks/wandb.py",
        "neuralrec/run/train.py",
        "neuralrec/run/__init__.py",
        "neuralrec/utils/__init__.py",
        "neuralrec/utils/stateful.py",
        "neuralrec/utils/utils.py",
    ),
    "runtime_support": (
        "utils/__init__.py",
        "utils/global_config.py",
        "utils/locks.py",
    ),
}

LOCAL_IMPLEMENTATION_SOURCES = tuple(
    source for sources in LOCAL_EXECUTION_SOURCES.values() for source in sources
)


def local_source_manifest(project_root: Path) -> dict[str, str]:
    if len(LOCAL_IMPLEMENTATION_SOURCES) != len(set(LOCAL_IMPLEMENTATION_SOURCES)):
        raise RuntimeError("local G2 provenance contains duplicate source paths")
    manifest = {}
    for relative in LOCAL_IMPLEMENTATION_SOURCES:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"invalid local G2 source path: {relative}")
        source = project_root / path
        if not source.is_file():
            raise RuntimeError(f"local G2 source is missing: {relative}")
        manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    return manifest
