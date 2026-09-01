from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ControlReference:
    manifest_sha256: str
    run_name: str
    embedding_learning_rate: float
    deep_learning_rate: float
    horizon_epochs: int
    best_epoch: int
    recall_at_100: float


@dataclass(frozen=True)
class ApprovedProtocol:
    schema_version: int
    approved_on: str
    batch_size: int
    seed: int
    dataset_sizes: tuple[str, ...]
    main_dataset_size: str
    embedding_lr_bounds: tuple[float, float]
    deep_lr_bounds: tuple[float, float]
    horizon_epochs: tuple[int, ...]
    content_sha256: tuple[tuple[str, str], ...]
    relative_dispersions: tuple[tuple[str, tuple[tuple[str, float], ...]], ...]
    initial_opportunity_budget: int
    conditional_opportunity_budget: int
    control: ControlReference

    @property
    def maximum_opportunity_budget(self) -> int:
        return self.initial_opportunity_budget + self.conditional_opportunity_budget

    def content_hash(self, dataset_size: str) -> str:
        try:
            return dict(self.content_sha256)[dataset_size]
        except KeyError as error:
            raise ValueError(f"unapproved dataset size {dataset_size!r}") from error

    def relative_dispersion(self, dataset_size: str, metric: str) -> float:
        datasets = dict(self.relative_dispersions)
        try:
            return dict(datasets[dataset_size])[metric]
        except KeyError as error:
            raise ValueError(
                f"no approved relative dispersion for {dataset_size!r} {metric!r}"
            ) from error

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


_METRIC_NAMES = (
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
    "mrr@10",
    "mrr@50",
    "mrr@100",
    "capped_recall@10",
    "capped_recall@50",
    "capped_recall@100",
    "coverage@10",
    "coverage@50",
    "coverage@100",
)


APPROVED_PROTOCOL = ApprovedProtocol(
    schema_version=1,
    approved_on="2026-08-29",
    batch_size=512,
    seed=42,
    dataset_sizes=("native-50m", "native-500m"),
    main_dataset_size="native-50m",
    embedding_lr_bounds=(0.0368614745, 0.5897835914),
    deep_lr_bounds=(0.0081084848, 0.1297357573),
    horizon_epochs=(15, 25, 40),
    content_sha256=(
        (
            "native-50m",
            "aa14c76ea36d5a9b8730bd856ba0f0e90bc7230a7179e04650b22d5a9572dd64",
        ),
        (
            "native-500m",
            "647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c",
        ),
    ),
    relative_dispersions=(
        (
            "native-50m",
            tuple(
                zip(
                    _METRIC_NAMES,
                    (
                        0.25924,
                        0.24172,
                        0.19414,
                        0.27546,
                        0.24805,
                        0.21427,
                        0.26918,
                        0.25274,
                        0.24280,
                        0.26000,
                        0.24190,
                        0.19413,
                        1.02046,
                        0.91862,
                        0.85178,
                    ),
                    strict=True,
                )
            ),
        ),
        (
            "native-500m",
            tuple(
                zip(
                    _METRIC_NAMES,
                    (
                        0.03152,
                        0.02116,
                        0.01685,
                        0.02680,
                        0.02272,
                        0.01966,
                        0.02393,
                        0.02157,
                        0.02085,
                        0.02955,
                        0.02107,
                        0.01683,
                        0.16765,
                        0.15102,
                        0.13429,
                    ),
                    strict=True,
                )
            ),
        ),
    ),
    initial_opportunity_budget=178,
    conditional_opportunity_budget=27,
    control=ControlReference(
        manifest_sha256=(
            "c30fb4eafcea2cefa1099631a40ca1531245e412c1cedcdbd02d9f7fea7aafd6"
        ),
        run_name="g4_control_trial_16_native50m",
        embedding_learning_rate=0.1474458978470563,
        deep_learning_rate=0.032433939334700325,
        horizon_epochs=25,
        best_epoch=20,
        recall_at_100=0.10435560161495364,
    ),
)

APPROVED_PROTOCOL_SHA256 = APPROVED_PROTOCOL.sha256

APPROVED_UNTIED_CONTROL_LEDGER_SHA256 = (
    "83b288364f57c79a5320fb4a53680c8d775eeba8b9a25805b9df4bdd7c8246a9"
)
