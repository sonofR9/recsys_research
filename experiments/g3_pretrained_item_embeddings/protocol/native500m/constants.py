from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


FamilyDesign = Literal[
    "nine_cell",
    "capacity",
    "rq5_global",
    "rq5_frequency",
]


@dataclass(frozen=True)
class LearningRateAnchor:
    source_id: str
    embedding_learning_rate: float
    deep_learning_rate: float


@dataclass(frozen=True)
class FamilySpec:
    id: str
    research_question: str
    code: int
    search_predecessor_id: str
    promotion_predecessor_id: str
    design: FamilyDesign
    budget: int
    capacities: tuple[int, ...] = ()
    conditional: bool = False


@dataclass(frozen=True)
class ArtifactIdentity:
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Native500MProtocol:
    schema_version: int
    approved_on: str
    dataset_size: str
    data_group: str
    batch_size: int
    seed: int
    horizon_epochs: tuple[int, ...]
    validation_interval_seconds: int
    minimum_query_events: int
    num_items: int
    filtered_event_count: int
    filtered_user_count: int
    remapped_event_count: int
    remapped_user_count: int
    training_interaction_count: int
    training_user_count: int
    evaluation_user_count: int
    validation_cutoff_timestamp: int
    content_width: int
    content_sha256: str
    local_learning_rate_factor: float
    hard_learning_rate_factor: float
    warmup_fraction: float
    initial_opportunity_budget: int
    with_conditional_opportunity_budget: int
    maximum_opportunity_budget: int

    @property
    def conditional_opportunity_budget(self) -> int:
        return (
            self.with_conditional_opportunity_budget - self.initial_opportunity_budget
        )

    @property
    def evaluable_user_count(self) -> int:
        return self.evaluation_user_count

    @property
    def sha256(self) -> str:
        document = {
            "protocol": asdict(self),
            "baseline_anchor": asdict(BASELINE_ANCHOR),
            "families": [asdict(spec) for spec in FAMILY_SPECS],
            "content_artifacts": [
                asdict(identity) for identity in CONTENT_ARTIFACT_IDENTITIES
            ],
            "dataset_artifacts": [
                asdict(identity) for identity in DATASET_ARTIFACT_IDENTITIES
            ],
            "feature_artifacts": [
                asdict(identity) for identity in FEATURE_ARTIFACT_IDENTITIES
            ],
        }
        payload = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


BASELINE_ANCHOR = LearningRateAnchor(
    source_id="g1_aggregate_selected",
    embedding_learning_rate=0.0468526465053628,
    deep_learning_rate=0.032703745675187676,
)


FAMILY_SPECS = (
    FamilySpec(
        "baseline", "baseline", 0, BASELINE_ANCHOR.source_id, "baseline", "nine_cell", 9
    ),
    FamilySpec("untied_control", "rq1", 10, "baseline", "baseline", "nine_cell", 9),
    FamilySpec("rq1_content_input", "rq1", 11, "baseline", "baseline", "nine_cell", 9),
    FamilySpec(
        "rq2_content_concat",
        "rq2",
        20,
        "baseline",
        "baseline",
        "capacity",
        12,
        (64, 128, 256),
    ),
    FamilySpec(
        "rq3_output_learned",
        "rq3",
        31,
        "rq2_content_concat",
        "rq3_local_control",
        "nine_cell",
        9,
    ),
    FamilySpec(
        "rq3_output_frozen_content",
        "rq3",
        32,
        "rq2_content_concat",
        "rq3_output_learned",
        "nine_cell",
        9,
    ),
    FamilySpec(
        "rq3_output_trainable_content",
        "rq3",
        33,
        "rq2_content_concat",
        "rq3_output_learned",
        "nine_cell",
        9,
    ),
    FamilySpec(
        "rq3_output_learned_frozen_content",
        "rq3",
        34,
        "rq2_content_concat",
        "rq3_output_learned",
        "nine_cell",
        9,
    ),
    FamilySpec(
        "rq3_output_learned_trainable_content",
        "rq3",
        35,
        "rq2_content_concat",
        "rq3_output_learned",
        "nine_cell",
        9,
    ),
    FamilySpec(
        "rq4_artist", "rq4", 41, "baseline", "baseline", "capacity", 12, (16, 32, 64)
    ),
    FamilySpec(
        "rq4_album", "rq4", 42, "baseline", "baseline", "capacity", 12, (16, 32, 64)
    ),
    FamilySpec(
        "rq4_artist_album",
        "rq4",
        43,
        "baseline",
        "baseline",
        "capacity",
        12,
        (16, 32, 64),
    ),
    FamilySpec(
        "rq5_global_gate",
        "rq5",
        51,
        "rq2_content_concat",
        "rq2_content_concat",
        "rq5_global",
        9,
    ),
    FamilySpec(
        "rq5_frequency_gate",
        "rq5",
        52,
        "rq2_content_concat",
        "rq5_global_gate",
        "rq5_frequency",
        11,
        (32, 64, 96),
    ),
    FamilySpec(
        "bridge_rq3_output",
        "aggregate",
        61,
        "aggregate_selected_input",
        "aggregate_selected_input_with_learned_output",
        "nine_cell",
        9,
        conditional=True,
    ),
    FamilySpec(
        "bridge_rq4_metadata",
        "aggregate",
        62,
        "aggregate_selected_input_output",
        "aggregate_selected_input_output_without_metadata",
        "nine_cell",
        9,
        conditional=True,
    ),
    FamilySpec(
        "aggregate",
        "aggregate",
        63,
        "most_specific_compatible_predecessor",
        "baseline",
        "nine_cell",
        9,
        conditional=True,
    ),
)


CONTENT_ARTIFACT_IDENTITIES = (
    ArtifactIdentity(
        "compact_output",
        "generated/datasets/yambda/500m_like_core5_knownitems/embeddings_compact.parquet",
        154290290,
        "647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c",
    ),
    ArtifactIdentity(
        "compact_remap",
        "generated/datasets/yambda/500m_like_core5_knownitems/item_id_remap.parquet",
        678369,
        "ec5d5e3e1e8045609c39b87a89ce8d05c7544622a4d9824ff2e47a8d258115e1",
    ),
    ArtifactIdentity(
        "compaction_implementation",
        "dcn/datasets/remap.py",
        2285,
        "bea65b5f1f563758b7cafb7f31dbf458642baa312bb9a9d62d74db06f68df9a3",
    ),
    ArtifactIdentity(
        "content_source",
        "generated/yambda_data/embeddings.parquet",
        13814230943,
        "c8959a584257473ab9f5dab7e88b05ea3d03b4daf7c16e9a617e8e216d811c83",
    ),
)


FEATURE_ARTIFACT_IDENTITIES = (
    ArtifactIdentity(
        "album_vocab",
        "generated/g3_pretrained_item_embeddings/native-500m/album_vocab.parquet",
        347509,
        "cf52152a787ab46bc063325d5d2c3fb2343bda0483c38209b1e14be832e0d3cc",
    ),
    ArtifactIdentity(
        "artist_vocab",
        "generated/g3_pretrained_item_embeddings/native-500m/artist_vocab.parquet",
        75774,
        "8d74781c46f1de4b05a94e6226d75ef31507becfd67c48f7554a8e565983a85e",
    ),
    ArtifactIdentity(
        "compact_remap",
        "generated/datasets/yambda/500m_like_core5_knownitems/item_id_remap.parquet",
        678369,
        "ec5d5e3e1e8045609c39b87a89ce8d05c7544622a4d9824ff2e47a8d258115e1",
    ),
    ArtifactIdentity(
        "events_source",
        "generated/datasets/yambda/500m_like_core5_knownitems/events_remapped.parquet",
        166290379,
        "1316037b5ac01f666aa85ad6c6bf59258139aa37c96d75dcdeab30ebe16b8dd6",
    ),
    ArtifactIdentity(
        "item_features",
        "generated/g3_pretrained_item_embeddings/native-500m/item_features.parquet",
        1710106,
        "2a39ca5eb9e8cced2a2a7bec89b53d45d1d16eb99f31054a5f11c1a7b64006fd",
    ),
    ArtifactIdentity(
        "materialization_implementation",
        "experiments/g3_pretrained_item_embeddings/data.py",
        6141,
        "03b75fc1d42b0a664e28babfa726e8850c7f3233b0eb23bf55a22b1a53d7a0c4",
    ),
    ArtifactIdentity(
        "training_user_histories",
        "generated/g3_pretrained_item_embeddings/native-500m/training_user_histories.parquet",
        330678,
        "2ad7139faec2803540d56d7ca1f9e2cf0cde5a67c79a3e99f67b364497743faa",
    ),
)


DATASET_ARTIFACT_IDENTITIES = (
    ArtifactIdentity(
        "compact_remap",
        "generated/datasets/yambda/500m_like_core5_knownitems/item_id_remap.parquet",
        678369,
        "ec5d5e3e1e8045609c39b87a89ce8d05c7544622a4d9824ff2e47a8d258115e1",
    ),
    ArtifactIdentity(
        "filtered_events",
        "generated/datasets/yambda/500m_like_core5_knownitems/events.parquet",
        222345846,
        "8aac12250ec4f5fb6972f82705d9c8001aa52d78191fef68ec38829ee3b39657",
    ),
    ArtifactIdentity(
        "protocol_implementation",
        "experiments/generation_protocol.py",
        906,
        "ba3699efc9d8b58e858822e428e9ee35c4e89932bdf7d579c86378a164fc7f05",
    ),
    ArtifactIdentity(
        "remapped_events",
        "generated/datasets/yambda/500m_like_core5_knownitems/events_remapped.parquet",
        166290379,
        "1316037b5ac01f666aa85ad6c6bf59258139aa37c96d75dcdeab30ebe16b8dd6",
    ),
)


PROTOCOL = Native500MProtocol(
    schema_version=1,
    approved_on="2026-08-31",
    dataset_size="native-500m",
    data_group="g3-native500m-likes",
    batch_size=512,
    seed=42,
    horizon_epochs=(10, 20, 40),
    validation_interval_seconds=604800,
    minimum_query_events=2,
    num_items=157357,
    filtered_event_count=8304589,
    filtered_user_count=81926,
    remapped_event_count=8013866,
    remapped_user_count=81635,
    training_interaction_count=7755722,
    training_user_count=81020,
    evaluation_user_count=37018,
    validation_cutoff_timestamp=25395195,
    content_width=128,
    content_sha256=("647b62ccc6cb214181e6aa44768fe94abd69e840b7758824f8e521dfe040043c"),
    local_learning_rate_factor=2.0,
    hard_learning_rate_factor=16.0,
    warmup_fraction=0.05,
    initial_opportunity_budget=140,
    with_conditional_opportunity_budget=167,
    maximum_opportunity_budget=295,
)


PROTOCOL_SHA256 = PROTOCOL.sha256


def family_spec(family_id: str) -> FamilySpec:
    try:
        return next(spec for spec in FAMILY_SPECS if spec.id == family_id)
    except StopIteration as error:
        raise ValueError(f"unknown native-500M G3 family {family_id!r}") from error


def boundary_extension_budget(spec: FamilySpec) -> int:
    if spec.id == "rq5_global_gate":
        return 4
    if spec.id == "rq5_frequency_gate":
        return 7
    if spec.design == "capacity":
        return 10
    return 7


def _validate_constants() -> None:
    ids = [spec.id for spec in FAMILY_SPECS]
    codes = [spec.code for spec in FAMILY_SPECS]
    if len(set(ids)) != len(ids) or len(set(codes)) != len(codes):
        raise RuntimeError("native-500M family IDs and codes must be unique")
    if sum(spec.budget for spec in FAMILY_SPECS if not spec.conditional) != 140:
        raise RuntimeError("native-500M initial budget drifted")
    if sum(spec.budget for spec in FAMILY_SPECS if spec.conditional) != 27:
        raise RuntimeError("native-500M conditional budget drifted")
    if any(width % 16 for spec in FAMILY_SPECS for width in spec.capacities):
        raise RuntimeError("native-500M learned widths must be divisible by 16")
    maximum = PROTOCOL.with_conditional_opportunity_budget + sum(
        boundary_extension_budget(spec) for spec in FAMILY_SPECS
    )
    if maximum != PROTOCOL.maximum_opportunity_budget:
        raise RuntimeError("native-500M maximum budget drifted")


_validate_constants()
