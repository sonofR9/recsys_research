from pathlib import Path

from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3GenerationExperiment,
    G3Representation,
    build_g3_experiment,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq4 import (
    Rq4CapacitySurface,
    Rq4ExtraIdRow,
    Rq4ExtraIdSurface,
    Rq4HorizonFollowup,
    Rq4MetadataRow,
    resolve_rq4_feature_data,
)


Rq4Surface = Rq4CapacitySurface | Rq4HorizonFollowup | Rq4ExtraIdSurface
Rq4Row = Rq4MetadataRow | Rq4ExtraIdRow


def build_rq4_training_experiment(
    surface: Rq4Surface,
    row: Rq4Row,
    *,
    root: Path,
) -> G3GenerationExperiment:
    if isinstance(surface, Rq4ExtraIdSurface):
        compiled_rows = surface.rows
    else:
        compiled_rows = surface.rows_by_family.get(row.family_id, ())
    if row not in compiled_rows:
        raise ValueError("RQ4 row is not part of the authenticated staged surface")
    if (
        row.batch_size != 512
        or row.seed != 42
        or row.id.split(":", 1)[0] != row.family_id
        or row.horizon_epochs < 1
    ):
        raise ValueError("compiled RQ4 row violates the approved training contract")
    if isinstance(row, Rq4MetadataRow):
        if row.reused_from is not None:
            raise ValueError(
                "reused RQ4 horizon cells must not launch duplicate training"
            )
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=surface.predecessor.history_hidden_dim,
            catalog_representation=surface.predecessor.catalog_representation,
            metadata=row.metadata,
            metadata_dim=row.metadata_dim,
        )
    else:
        representation = G3Representation(
            history_representation="id_content",
            history_hidden_dim=surface.predecessor.history_hidden_dim,
            catalog_representation=surface.predecessor.catalog_representation,
            extra_item_id_dim=row.extra_item_id_dim,
        )
    feature_data_path = resolve_rq4_feature_data(root=root, surface=surface)
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=row.horizon_epochs,
        seed=row.seed,
        representation=representation,
        feature_data_path=feature_data_path,
    )
