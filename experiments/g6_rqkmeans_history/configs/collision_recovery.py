from __future__ import annotations

from dcn.config import SemanticHistoryExperiment
from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy,
)
from experiments.g6_rqkmeans_history.protocol.collision_recovery import (
    CollisionRecoveryJob,
    recovery_job,
)


def build_collision_recovery_experiment(
    job: CollisionRecoveryJob,
) -> SemanticHistoryExperiment:
    if job != recovery_job():
        raise ValueError("job is outside the approved collision recovery")
    source = job.source_job
    coordinate = source.coordinate
    return build_collision_policy(
        policy=source.policy,
        num_levels=coordinate.num_levels,
        num_codes=coordinate.num_codes,
        kmeans_iterations=coordinate.kmeans_iterations,
        embedding_learning_rate=coordinate.embedding_learning_rate,
        deep_learning_rate=coordinate.deep_learning_rate,
        run_name=job.run_name,
    )
