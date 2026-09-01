from __future__ import annotations

from dcn.config import CollisionPolicy, SemanticHistoryExperiment, SemanticIdConfig
from neuralrec.run.train import TrainRunner
from experiments.g6_rqkmeans_history.configs.rq0 import _common
from experiments.g6_rqkmeans_history.protocol.collision_policy import (
    CollisionSearchJob,
    validate_collision_search_job,
    validate_collision_symbol_cap,
)


class CollisionPolicyExperiment(SemanticHistoryExperiment):
    def create_runner(self) -> TrainRunner:
        validate_collision_symbol_cap(
            self.semantic_codes,
            policy=self.semantic.collision_policy,
            base_levels=self.semantic.num_levels,
            require_suffix_feasibility=True,
        )
        return super().create_runner()


def build_collision_policy_experiment(
    job: CollisionSearchJob,
) -> SemanticHistoryExperiment:
    validate_collision_search_job(job)
    coordinate = job.coordinate
    return build_collision_policy(
        policy=job.policy,
        num_levels=coordinate.num_levels,
        num_codes=coordinate.num_codes,
        kmeans_iterations=coordinate.kmeans_iterations,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        run_name=job.run_name,
    )


def build_collision_policy(
    *,
    policy: CollisionPolicy,
    num_levels: int,
    num_codes: int,
    kmeans_iterations: int,
    embedding_learning_rate: float,
    deep_learning_rate: float,
    run_name: str,
    training_seed: int = 42,
) -> SemanticHistoryExperiment:
    common = _common(
        "best_g1",
        batch_size=256,
        validation_batch_size=8192,
        embedding_learning_rate=embedding_learning_rate,
        deep_learning_rate=deep_learning_rate,
        run_name=run_name,
    )
    common["seed"] = training_seed
    return CollisionPolicyExperiment(
        **common,
        history_representation="item_frozen_sid_event",
        representation_width=128,
        semantic=SemanticIdConfig(
            quantizer="kmeans",
            num_levels=num_levels,
            num_codes=num_codes,
            kmeans_iterations=kmeans_iterations,
            collision_policy=policy,
            seed=42,
        ),
    )
