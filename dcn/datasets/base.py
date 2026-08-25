from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSourceArtifacts:
    """What a prepared `DatasetSource` hands to the training pipeline.

    `columns` lists every column the pipeline reads off the parquet — features,
    targets and masks alike. The dataset buckets each one by its parquet dtype
    (int/bool -> int columns, float -> float columns); semantic roles are not
    encoded here but applied downstream by name (the model picks its feature
    columns, the criterion picks its target/mask columns).

    The `main_parquet` file (or each entry in a multi-day split) MUST contain:
      - every column listed in `columns`; categorical ids must be int64-castable
        and remapped to compact ids 1..N for any feature with precomputed
        embeddings (0 is reserved for "unknown")
      - the timestamp column referenced by `timestamp_column` (int64-castable)

    `precomputed_embeddings` maps a feature column name (e.g. "item_id",
    "album_id") to the parquet file holding the compact embedding table for
    that feature. The parquet must follow the schema documented on
    `PrecomputedEmbeddingLookup.from_parquet`.

    `user_column` and `item_id_column` name the two columns the pipeline itself
    has to know: the one histories are grouped by, and the compact item id every
    per-item artifact (embeddings, semantic ids, the catalog a retrieval model
    scores) is keyed on. The source names them because the source is what wrote
    them.
    """

    main_parquet: Path
    columns: list[str]
    precomputed_embeddings: dict[str, Path]  # should also contain column name/ mapping?
    timestamp_column: str
    user_column: str
    item_id_column: str


class DatasetSource(ABC):
    @property
    @abstractmethod
    def artifacts(self) -> DatasetSourceArtifacts: ...
