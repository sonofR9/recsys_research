from dcn.config import SemanticHistoryExperiment
from experiments.g6_rqkmeans_history.configs.collision_policy import (
    build_collision_policy,
)
from experiments.g6_rqkmeans_history.configs.rq1 import build_rq1_initialization
from experiments.g6_rqkmeans_history.protocol.confirmation import ConfirmationJob


def build_confirmation_experiment(
    job: ConfirmationJob,
) -> SemanticHistoryExperiment:
    if job.family == "rq1":
        initialization = job.variant.rsplit("_t", 1)[0]
        return build_rq1_initialization(
            initialization,
            embedding_learning_rate=job.embedding_learning_rate,
            deep_learning_rate=job.deep_learning_rate,
            run_name=job.run_name,
            training_seed=job.seed,
        )
    policy = job.variant.split("_", 1)[0]
    if policy not in {"suffix", "none"}:
        raise ValueError("confirmation collision policy is invalid")
    return build_collision_policy(
        policy=policy,
        num_levels=job.num_levels,
        num_codes=job.num_codes,
        kmeans_iterations=job.kmeans_iterations,
        embedding_learning_rate=job.embedding_learning_rate,
        deep_learning_rate=job.deep_learning_rate,
        run_name=job.run_name,
        training_seed=job.seed,
    )
