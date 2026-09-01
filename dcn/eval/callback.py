"""Epoch-end hook that runs the full-catalog future-day eval on a live model."""

from typing import Iterable, Sequence

import torch
from torch import nn

from dcn.eval.base import EpochEvalCallback
from dcn.eval.ranking_evidence import RankingEvidence
from dcn.eval.true_metric import (
    PreparedRanking,
    RankingDetails,
    evaluate_true_ndcg,
    prepare_ranking,
)
from dcn.semantic import SemanticCodes
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
        semantic_codes: SemanticCodes | None = None,
        semantic_base_levels: int | None = None,
        train_item_frequencies: dict[int, int] | None = None,
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
        if (semantic_codes is None) != (semantic_base_levels is None):
            raise ValueError(
                "semantic codes and the number of base levels must be provided together"
            )
        if semantic_codes is not None and not (
            1 <= semantic_base_levels <= semantic_codes.num_levels
        ):
            raise ValueError("semantic base levels must select existing code levels")
        self.semantic_codes = semantic_codes
        self.semantic_base_levels = semantic_base_levels
        self.train_item_frequencies = train_item_frequencies
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
            result = self._score(max_users, return_evidence=False)
            assert result is None or isinstance(result, dict)
            return result
        finally:
            self.model.train(was_training)

    def full_user_query_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.model.training
        self.model.eval()
        try:
            with (
                torch.inference_mode(),
                torch.autocast(
                    self._device.type,
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ),
            ):
                user_ids, query_repr = self._encode_queries()
            return user_ids.detach().cpu(), query_repr.detach().float().cpu()
        finally:
            self.model.train(was_training)

    def score_with_evidence(
        self, *, max_users: int | None
    ) -> tuple[dict[str, float], RankingEvidence] | None:
        if self.train_item_frequencies is None:
            raise ValueError("train item frequencies are required for ranking evidence")
        was_training = self.model.training
        self.model.eval()
        try:
            result = self._score(max_users, return_evidence=True)
            assert result is None or isinstance(result, tuple)
            return result
        finally:
            self.model.train(was_training)

    def score_with_evidence_and_rankings(
        self, *, max_users: int | None
    ) -> tuple[dict[str, float], RankingEvidence, dict[int, tuple[int, ...]]] | None:
        if self.train_item_frequencies is None:
            raise ValueError("train item frequencies are required for ranking evidence")
        was_training = self.model.training
        self.model.eval()
        try:
            result = self._score(max_users, return_evidence=True, return_rankings=True)
            assert result is None or isinstance(result, tuple)
            return result
        finally:
            self.model.train(was_training)

    def _evaluate(self) -> dict[str, float] | None:
        result = self._score(self.max_users, return_evidence=False)
        assert result is None or isinstance(result, dict)
        return result

    def _score(
        self,
        max_users: int | None,
        *,
        return_evidence: bool,
        return_rankings: bool = False,
    ) -> (
        dict[str, float]
        | tuple[dict[str, float], RankingEvidence]
        | tuple[dict[str, float], RankingEvidence, dict[int, tuple[int, ...]]]
        | None
    ):
        if return_rankings and not return_evidence:
            raise ValueError("rankings require ranking evidence")
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

        item_semantic_codes = None
        if self.semantic_codes is not None:
            item_ids_cpu = item_ids.cpu()
            if not bool(torch.isin(item_ids_cpu, self.semantic_codes.item_ids).all()):
                raise ValueError(
                    "the ranked catalog contains items without semantic codes"
                )
            table = self.semantic_codes.lookup_table(
                max(
                    int(item_ids_cpu.max()),
                    int(self.semantic_codes.item_ids.max()),
                )
            )
            item_semantic_codes = table[item_ids_cpu, : self.semantic_base_levels].to(
                self._device
            )

        # Ranked outside the autocast block: the dot product decides the order
        # of near-tied items, and bf16 carries eight bits of mantissa to do it.
        result = evaluate_true_ndcg(
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
            item_semantic_codes=item_semantic_codes,
            return_relevant_ranks=return_evidence and not return_rankings,
            return_ranking_details=return_rankings,
        )
        if not return_evidence:
            assert isinstance(result, dict)
            return result
        assert isinstance(result, tuple)
        metrics, ranking_result = result
        prepared = self._prepared[max_users]
        if return_rankings:
            assert isinstance(ranking_result, RankingDetails)
            rankings = {
                user.user_id: tuple(map(int, top_item_ids.tolist()))
                for user, top_item_ids in zip(
                    prepared.evaluable,
                    ranking_result.top_item_ids,
                    strict=True,
                )
            }
            return (
                metrics,
                self._ranking_evidence(prepared, ranking_result.relevant_ranks),
                rankings,
            )
        assert isinstance(ranking_result, torch.Tensor)
        return metrics, self._ranking_evidence(prepared, ranking_result)

    def _ranking_evidence(
        self, prepared: PreparedRanking, relevant_ranks: torch.Tensor
    ) -> RankingEvidence:
        histories: dict[int, torch.Tensor] = {}
        for batch in self._batches():
            cumulative_lens = batch["cumulative_lens"].cpu().tolist()
            item_ids = batch["int_columns"][self.item_id_column].dense().cpu()
            user_ids = batch["int_columns"][self.user_column].dense().cpu()
            for start, end in zip(
                cumulative_lens[:-1], cumulative_lens[1:], strict=True
            ):
                user_id = int(user_ids[end - 1])
                if user_id in histories:
                    raise ValueError("ranking evidence requires one query per user")
                histories[user_id] = item_ids[start:end]

        user_ids: list[int] = []
        history_values: list[torch.Tensor] = []
        history_offsets = [0]
        relevant_values: list[int] = []
        relevance_offsets = [0]
        frequencies: list[int] = []
        assert self.train_item_frequencies is not None
        catalog_positions = {
            item_id: position for position, item_id in enumerate(prepared.item_id_list)
        }
        for user in prepared.evaluable:
            user_ids.append(user.user_id)
            history = histories[user.user_id]
            history_values.append(history)
            history_offsets.append(history_offsets[-1] + history.shape[0])
            relevant = sorted(user.relevant, key=catalog_positions.__getitem__)
            relevant_values.extend(relevant)
            relevance_offsets.append(relevance_offsets[-1] + len(relevant))
            frequencies.extend(
                self.train_item_frequencies.get(item_id, 0) for item_id in relevant
            )

        return RankingEvidence(
            user_ids=torch.tensor(user_ids, dtype=torch.int64),
            history_item_ids=(
                torch.cat(history_values)
                if history_values
                else torch.empty(0, dtype=torch.int64)
            ),
            history_offsets=torch.tensor(history_offsets, dtype=torch.int64),
            relevant_item_ids=torch.tensor(relevant_values, dtype=torch.int64),
            relevance_offsets=torch.tensor(relevance_offsets, dtype=torch.int64),
            relevant_train_frequencies=torch.tensor(frequencies, dtype=torch.int64),
            relevant_ranks=relevant_ranks,
            max_k=max(self.ks),
        )
