from dataclasses import replace

from dcn.datasets.yambda import UserSample
from experiments.g2_esasrec.configs.local import (
    ComponentMethod,
    LocalG2Experiment,
    build_component,
)

SMOKE_METHODS: tuple[ComponentMethod, ...] = (
    "standard_sampled_softmax",
    "standard_gbce",
    "ligr_sampled_softmax",
    "ligr_gbce",
)


def build_smoke(method: ComponentMethod) -> LocalG2Experiment:
    if method not in SMOKE_METHODS:
        raise ValueError(f"{method!r} is not an approved G2 smoke method")
    experiment = build_component(
        method,
        batch_size=128,
        embedding_learning_rate=0.001,
        deep_learning_rate=0.001,
        ligr_multiplier=6,
        gbce_t=0.75 if method.endswith("_gbce") else None,
    )
    return replace(
        experiment,
        run_name=f"g2_smoke_{method}_2000users_seed42",
        user_sample=UserSample(max_users=2_000, seed=42),
        num_epochs=1,
        early_stopping_patience=None,
    )
