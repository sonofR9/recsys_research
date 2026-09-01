from pathlib import Path

from experiments.g3_pretrained_item_embeddings.configs.model import (
    G3GenerationExperiment,
    G3Representation,
    build_g3_experiment,
)
from experiments.g3_pretrained_item_embeddings.protocol.rq5 import (
    Rq5GateRow,
    Rq5GateSurface,
    resolve_rq5_feature_data,
)


def build_rq5_training_experiment(
    surface: Rq5GateSurface,
    row: Rq5GateRow,
    *,
    root: Path,
) -> G3GenerationExperiment:
    feature_data_path = resolve_rq5_feature_data(root=root, surface=surface)
    if row == surface.fixed_gate:
        raise ValueError(
            "selected fixed-gate reuse must not launch duplicate RQ5 training"
        )
    if row not in (*surface.global_gate_rows, *surface.frequency_gate_rows):
        raise ValueError("RQ5 row is not part of the authenticated gate surface")
    if row.reused_from is not None:
        raise ValueError("reused RQ5 horizon rows must not launch duplicate training")
    if (
        row.batch_size != 512
        or row.seed != 42
        or row.history_hidden_dim != surface.selected_history_hidden_dim
        or row.id.split(":", 1)[0] != row.family_id
        or (row.family_id == "rq5_global_gate" and row.content_gate != "global")
        or (row.family_id == "rq5_frequency_gate" and row.content_gate != "frequency")
        or (row.content_gate == "global" and row.gate_hidden_dim is not None)
        or (row.content_gate == "frequency" and row.gate_hidden_dim not in {4, 8, 16})
    ):
        raise ValueError("compiled RQ5 row violates the approved training contract")
    return build_g3_experiment(
        run_name=row.run_name,
        dataset_size="native-50m",
        embedding_learning_rate=row.embedding_learning_rate,
        deep_learning_rate=row.deep_learning_rate,
        lr_schedule_horizon_epochs=row.horizon_epochs,
        seed=row.seed,
        representation=G3Representation(
            history_representation="id_content",
            catalog_representation="learned_id",
            history_hidden_dim=row.history_hidden_dim,
            content_gate=row.content_gate,
            gate_hidden_dim=row.gate_hidden_dim,
        ),
        feature_data_path=feature_data_path,
    )
