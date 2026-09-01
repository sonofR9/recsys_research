from pathlib import Path

from experiments.g3_pretrained_item_embeddings.configs.model import (
    build_g3_experiment,
    build_rq3_representation,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import Rq3OutputRow
from experiments.g3_pretrained_item_embeddings.protocol.rq3 import (
    Rq3OutputSurface,
    resolve_rq3_feature_data,
)


def build_rq3_training_experiment(
    surface: Rq3OutputSurface,
    row: Rq3OutputRow,
    *,
    root: Path,
):
    compiled_rows = surface.rows_by_family.get(row.family_id, ())
    if row not in compiled_rows:
        raise ValueError("RQ3 row is not part of the authenticated output surface")
    if row.reused_from is not None:
        raise ValueError("reused RQ2 rows must not launch duplicate RQ3 training")
    if (
        row.batch_size != 512
        or row.seed != 42
        or row.id.split(":", 1)[0] != row.family_id
        or row.history_hidden_dim < 1
    ):
        raise ValueError("compiled RQ3 row violates the approved training contract")
    feature_data_path = resolve_rq3_feature_data(root=root, surface=surface)
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=row.horizon_epochs,
        seed=row.seed,
        representation=build_rq3_representation(
            row.family_id,
            history_hidden_dim=row.history_hidden_dim,
        ),
        feature_data_path=feature_data_path,
    )
