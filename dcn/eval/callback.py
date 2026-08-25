"""Epoch-end hook that runs the full-catalog future-day eval on a live model."""

from typing import Iterable, Sequence

import torch
from torch import nn

from dcn.eval.base import EpochEvalCallback
from dcn.eval.true_metric import PreparedRanking, evaluate_true_ndcg, prepare_ranking
from neuralrec.data.transforms import move_to_device


class TrueMetricCallback(EpochEvalCallback):
    """Rank the whole catalog for every user and log NDCG/Recall/MRR@k."""

    def __init__(
        self,
        *,
        model: nn.Module,
        item_batch: dict,
        query_loader: Iterable[dict],
        relevance: dict[int, set[int]],
        train_seen: dict[int, set[int]],
        user_column: str,
        item_id_column: str,
        ks: Sequence[int] = (10, 50, 100),
        every_n_epochs: int = 1,
        prefix: str = "epoch/val_true",
        user_chunk: int = 256,
        max_users: int | None = None,
        seed: int = 42,
        exclude_seen: bool = True,
        # Encoding runs outside the training model's AutoCast wrapper.
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(model=model, prefix=prefix, every_n_epochs=every_n_epochs)
        self.item_batch = item_batch
        self.query_loader = query_loader
        self.item_id_column = item_id_column
        self.relevance = relevance
        self.train_seen = train_seen
        self.ks = tuple(ks)
        self.user_column = user_column
        self.user_chunk = user_chunk
        self.max_users = max_users
        self.seed = seed
        self.exclude_seen = exclude_seen
        self.dtype = dtype
        self._prepared: dict[int | None, PreparedRanking] = {}
        self._query_batches: list[dict] | None = None
        self._prepared_query_batches: list[dict] | None = None

    def prepare(self) -> None:
        if self._query_batches is not None or self._prepared_query_batches is not None:
            return
        self._prepared_query_batches = list(self.query_loader)

    def _encode_catalog(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = move_to_device(self.item_batch, self._device)
        item_ids = batch["int_columns"][self.item_id_column].dense()
        return item_ids, self.model.encode_items(batch)

    def _batches(self) -> list[dict]:
        """Held on the device after the first epoch: the query set is every
        user's state at one fixed cutoff, so re-reading it each epoch would run
        the loader to rebuild the same tensors."""
        if self._query_batches is None:
            batches = (
                self._prepared_query_batches
                if self._prepared_query_batches is not None
                else self.query_loader
            )
            self._prepared_query_batches = None
            self._query_batches = [
                move_to_device(batch, self._device) for batch in batches
            ]
        return self._query_batches

    def _encode_queries(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_ids: list[torch.Tensor] = []
        query_repr: list[torch.Tensor] = []
        for batch in self._batches():
            last = batch["cumulative_lens"][1:] - 1
            if hasattr(self.model, "encode_cutoff_queries"):
                query_repr.append(self.model.encode_cutoff_queries(batch))
            else:
                query_repr.append(self.model.encode_queries(batch)[last])
            user_ids.append(batch["int_columns"][self.user_column].dense()[last])

        if not query_repr:
            empty = torch.empty(0, dtype=torch.long)
            return empty, torch.empty(0, 0)
        return torch.cat(user_ids), torch.cat(query_repr)

    def score(self, *, max_users: int | None) -> dict[str, float] | None:
        """Rank the catalog outside the epoch cadence -- for the full-population
        numbers a run reports once, after training."""
        was_training = self.model.training
        self.model.eval()
        try:
            return self._score(max_users)
        finally:
            self.model.train(was_training)

    def _evaluate(self) -> dict[str, float] | None:
        return self._score(self.max_users)

    def _score(self, max_users: int | None) -> dict[str, float] | None:
        with (
            torch.inference_mode(),
            torch.autocast(
                self._device.type,
                dtype=self.dtype,
                enabled=self.dtype != torch.float32,
            ),
        ):
            item_ids, item_repr = self._encode_catalog()
            query_user_ids, query_repr = self._encode_queries()

        if query_repr.shape[0] == 0:
            return None

        if max_users not in self._prepared:
            self._prepared[max_users] = prepare_ranking(
                query_user_ids,
                item_ids,
                self.relevance,
                self.train_seen,
                device=self._device,
                user_chunk=self.user_chunk,
                max_users=max_users,
                seed=self.seed,
                exclude_seen=self.exclude_seen,
            )

        # Ranked outside the autocast block: the dot product decides the order
        # of near-tied items, and bf16 carries eight bits of mantissa to do it.
        return evaluate_true_ndcg(
            query_repr=query_repr.float(),
            query_user_ids=query_user_ids,
            item_repr=item_repr.float(),
            item_ids=item_ids,
            relevance=self.relevance,
            train_seen=self.train_seen,
            ks=self.ks,
            device=self._device,
            user_chunk=self.user_chunk,
            max_users=max_users,
            seed=self.seed,
            prepared=self._prepared[max_users],
            exclude_seen=self.exclude_seen,
        )
