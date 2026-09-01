import json

import pytest

from experiments.g4_future_items.report.artifacts import read_recommender_trial


def test_recommender_artifact_uses_restored_best_validation_epoch(tmp_path) -> None:
    (tmp_path / "g4_job.json").write_text(
        json.dumps(
            {
                "row_id": "control_tuning:01",
                "job": {
                    "run_name": "g4_control_trial_01_native50m",
                    "dataloader": {"batch_size": 512},
                    "embedding_learning_rate": 0.01,
                    "deep_learning_rate": 0.02,
                    "lr_schedule_horizon_epochs": 20,
                },
            }
        )
    )
    (tmp_path / "training_metadata.json").write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "epochs_trained": 20,
                "lr_schedule_horizon_epochs": 20,
                "num_epochs": 20,
                "max_epochs": 20,
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
                "lr_horizon_complete": True,
                "selection_resolved": True,
            }
        )
    )
    (tmp_path / "sweep.log").write_text(
        "epoch 0 finished epoch/val.loss=1.2000 epoch/val_true.recall@100=0.1000\n"
        "epoch 1 finished epoch/val.loss=1.1000 epoch/val_true.recall@100=0.1200\n"
        "epoch 2 finished epoch/val.loss=1.0000 epoch/val_true.recall@100=0.1100\n"
    )

    trial = read_recommender_trial(tmp_path)

    assert trial.row_id == "control_tuning:01"
    assert trial.validation_recall_at_100 == pytest.approx(0.12)
    assert trial.validation_loss == pytest.approx(1.1)
    assert trial.epochs_trained == 20


def test_recommender_artifact_rejects_unresolved_or_missing_best_epoch(
    tmp_path,
) -> None:
    (tmp_path / "g4_job.json").write_text(
        json.dumps(
            {
                "row_id": "x",
                "job": {
                    "run_name": "x",
                    "dataloader": {"batch_size": 512},
                    "embedding_learning_rate": 0.01,
                    "deep_learning_rate": 0.02,
                    "lr_schedule_horizon_epochs": 20,
                },
            }
        )
    )
    (tmp_path / "training_metadata.json").write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "epochs_trained": 19,
                "lr_schedule_horizon_epochs": 20,
                "num_epochs": 20,
                "max_epochs": 20,
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
                "lr_horizon_complete": False,
                "selection_resolved": False,
            }
        )
    )
    (tmp_path / "sweep.log").write_text("")

    trial = read_recommender_trial(tmp_path)

    assert trial.usable is False
    assert trial.validation_recall_at_100 != trial.validation_recall_at_100


@pytest.mark.parametrize(
    ("contract_batch", "contract_horizon", "metadata_horizon"),
    [(128, 20, 20), (512, 5, 20)],
)
def test_recommender_artifact_rejects_batch_or_horizon_mismatch(
    tmp_path, contract_batch, contract_horizon, metadata_horizon
) -> None:
    (tmp_path / "g4_job.json").write_text(
        json.dumps(
            {
                "row_id": "control_tuning:01",
                "job": {
                    "run_name": "g4_control_trial_01_native50m",
                    "dataloader": {"batch_size": contract_batch},
                    "embedding_learning_rate": 0.01,
                    "deep_learning_rate": 0.02,
                    "lr_schedule_horizon_epochs": contract_horizon,
                },
            }
        )
    )
    (tmp_path / "training_metadata.json").write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "epochs_trained": metadata_horizon,
                "lr_schedule_horizon_epochs": metadata_horizon,
                "num_epochs": metadata_horizon,
                "max_epochs": metadata_horizon,
                "batch_size": 512,
                "embedding_learning_rate": 0.01,
                "deep_learning_rate": 0.02,
                "lr_horizon_complete": True,
                "selection_resolved": True,
            }
        )
    )
    (tmp_path / "sweep.log").write_text(
        "epoch 1 finished epoch/val.loss=1.1 epoch/val_true.recall@100=0.12\n"
    )

    with pytest.raises(ValueError, match="batch|horizon"):
        read_recommender_trial(tmp_path)
