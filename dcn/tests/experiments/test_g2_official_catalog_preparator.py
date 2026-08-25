from pathlib import Path
import subprocess

import pytest


RECTOOLS_PYTHON = Path("/home/sashanovak/envs/esasrec/bin/python")


@pytest.mark.skipif(not RECTOOLS_PYTHON.is_file(), reason="RecTools environment absent")
def test_catalog_preparator_maps_candidates_without_adding_interactions() -> None:
    script = """
import pandas as pd
from rectools import Columns
from rectools.dataset import Dataset
from rectools.models.nn.item_net import IdEmbeddingsItemNet

from experiments.g2_esasrec.official.catalog_data import CatalogCompleteSASRecDataPreparator

interactions = pd.DataFrame({
    Columns.User: [1, 1, 2, 2],
    Columns.Item: [10, 20, 10, 20],
    Columns.Weight: [1.0] * 4,
    Columns.Datetime: pd.to_datetime([1, 2, 1, 2], unit="s"),
})
dataset = Dataset.construct(interactions)
preparator = CatalogCompleteSASRecDataPreparator(
    session_max_len=10,
    batch_size=2,
    dataloader_num_workers=0,
    candidate_item_ids=[10, 20, 30],
    expected_candidate_count=3,
)
preparator.process_dataset_train(dataset)
item_net = IdEmbeddingsItemNet.from_dataset(
    preparator.train_dataset, n_factors=4, dropout_rate=0.0
)

assert preparator.train_dataset.interactions.df.shape[0] == 4
assert set(preparator.get_known_item_ids()) == {10, 20, 30}
assert preparator.item_id_map.convert_to_internal([30]).item() == 3
assert preparator.item_id_map.size == 4
assert item_net.n_items == 4
"""

    subprocess.run(
        [str(RECTOOLS_PYTHON), "-c", script],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )
