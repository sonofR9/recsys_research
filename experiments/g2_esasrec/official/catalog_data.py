from collections.abc import Sequence

import numpy as np
from rectools.dataset import Dataset
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator


class CatalogCompleteSASRecDataPreparator(SASRecDataPreparator):
    def __init__(
        self,
        *args: object,
        candidate_item_ids: Sequence[int],
        expected_candidate_count: int,
        **kwargs: object,
    ) -> None:
        candidates = np.asarray(candidate_item_ids)
        if candidates.ndim != 1 or candidates.size != expected_candidate_count:
            raise ValueError("candidate item ids do not match the expected count")
        if np.unique(candidates).size != candidates.size:
            raise ValueError("candidate item ids must be unique")
        self.candidate_item_ids = candidates
        self.expected_candidate_count = expected_candidate_count
        super().__init__(*args, **kwargs)

    def process_dataset_train(self, dataset: Dataset) -> None:
        super().process_dataset_train(dataset)
        if self.train_dataset.item_features is not None:
            raise ValueError("catalog completion supports ID embeddings only")
        training_item_ids = self.get_known_item_ids()
        if not np.isin(training_item_ids, self.candidate_item_ids).all():
            raise ValueError("training items are absent from the candidate catalog")

        item_id_map = self.item_id_map.add_ids(self.candidate_item_ids)
        known_count = item_id_map.size - self.n_item_extra_tokens
        if known_count != self.expected_candidate_count:
            raise ValueError("completed candidate catalog has an unexpected size")
        self.train_dataset = Dataset(
            user_id_map=self.train_dataset.user_id_map,
            item_id_map=item_id_map,
            interactions=self.train_dataset.interactions,
            user_features=self.train_dataset.user_features,
        )
        self.item_id_map = item_id_map
        self._init_extra_token_ids()
