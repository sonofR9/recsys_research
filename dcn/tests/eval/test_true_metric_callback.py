from pathlib import Path

import polars as pl
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from dcn.data import FeatureValues, SequenceDataset, collate_sequence_batch
from dcn.eval import (
    TrueMetricCallback,
    build_catalog_batch,
    build_interaction_sets,
    build_item_snapshot,
)
from dcn.models import Tower, TowerInputEncoder, TwoTowerModel
from dcn.semantic import SemanticCodes
from neuralrec.utils import EXTRA_METRICS

SECONDS_IN_DAY = 86_400
FIRST_ITEM = 10
CATALOG = [10, 11, 12, 13, 14]

# uid -> [(day, item), ...]. The *last* training item of users 1-3 decides
# what they touch on the future day; user 4 only puts item 14 in the catalog.
TRAIN_HISTORY = {
    1: [(0, 10), (1, 11)],
    2: [(0, 11), (1, 12)],
    3: [(0, 13), (1, 13)],
    4: [(1, 14)],
}
FUTURE = {1: 12, 2: 13, 3: 14}


def _write_days(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    rows: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for uid, history in TRAIN_HISTORY.items():
        for position, (day, item) in enumerate(history):
            rows[day].append(
                {
                    "uid": uid,
                    "compact_item_id": item,
                    "timestamp": day * SECONDS_IN_DAY + position,
                }
            )
    for uid, item in FUTURE.items():
        rows[2].append(
            {
                "uid": uid,
                "compact_item_id": item,
                "timestamp": 2 * SECONDS_IN_DAY + uid,
            }
        )

    paths = {}
    for day, day_rows in rows.items():
        path = tmp_path / f"day_{day}.parquet"
        pl.DataFrame(day_rows).write_parquet(path)
        paths[day] = path
    return [paths[0], paths[1]], [paths[2]]


class _NextItemStub(nn.Module):
    """Scores the token's own item 2.0 and the following catalog item 1.0.

    So the relevant future item only reaches the top when the callback both
    takes the *last* token as the query and masks the train-seen items -- the
    user's own last item would otherwise outrank it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def _one_hot(self, item_ids: torch.Tensor) -> torch.Tensor:
        index = item_ids - FIRST_ITEM
        table = 2.0 * torch.eye(len(CATALOG))
        table = table + torch.roll(torch.eye(len(CATALOG)), shifts=1, dims=1)
        return table[index]

    def encode_items(self, batch: dict) -> torch.Tensor:
        item_ids = batch["int_columns"]["compact_item_id"].dense()
        return torch.eye(len(CATALOG))[item_ids - FIRST_ITEM]

    def encode_queries(self, batch: dict) -> torch.Tensor:
        return self._one_hot(batch["int_columns"]["compact_item_id"].dense())


class _CutoffStub(_NextItemStub):
    def encode_queries(self, batch: dict) -> torch.Tensor:
        raise AssertionError("token positions are model-specific")

    def encode_cutoff_queries(self, batch: dict) -> torch.Tensor:
        last = batch["cumulative_lens"][1:] - 1
        item_ids = batch["int_columns"]["compact_item_id"].dense()
        return self._one_hot(item_ids[last])


class _FixedRankingStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def encode_items(self, batch: dict) -> torch.Tensor:
        item_ids = batch["int_columns"]["compact_item_id"].dense()
        return torch.eye(len(CATALOG))[item_ids - FIRST_ITEM]

    def encode_queries(self, batch: dict) -> torch.Tensor:
        scores = torch.arange(1, len(CATALOG) + 1, dtype=torch.float32)
        return scores.repeat(batch["cumulative_lens"][-1], 1)


def _build_callback(
    tmp_path: Path, model: nn.Module, **overrides: object
) -> TrueMetricCallback:
    train_files, val_files = _write_days(tmp_path)

    dataset = SequenceDataset(
        train_files,
        ["compact_item_id"],
        tmp_path / "seq_cache",
        user_column="uid",
        max_seq_len=8,
        min_seq_len=2,
        emit_user_column=True,
    )

    arguments: dict = {
        "model": model,
        "item_batch": build_item_snapshot(
            train_files, item_id_column="compact_item_id"
        ),
        "query_loader": DataLoader(
            dataset, batch_size=2, collate_fn=collate_sequence_batch
        ),
        "relevance": build_interaction_sets(
            val_files, user_column="uid", item_id_column="compact_item_id"
        ),
        "train_seen": build_interaction_sets(
            train_files, user_column="uid", item_id_column="compact_item_id"
        ),
        "user_column": "uid",
        "item_id_column": "compact_item_id",
        "ks": (3,),
    }
    return TrueMetricCallback(**{**arguments, **overrides})


def _metrics(state: dict) -> dict[str, float]:
    return state[EXTRA_METRICS]["epoch/val_true"]


def test_perfect_next_item_model_scores_one(tmp_path: Path) -> None:
    callback = _build_callback(tmp_path, _NextItemStub())
    state: dict = {}

    callback.on_epoch_end(state)

    metrics = _metrics(state)
    assert metrics["num_users"] == 3.0
    assert metrics["ndcg@3"] == pytest.approx(1.0)
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert metrics["mrr@3"] == pytest.approx(1.0)


def test_model_can_resolve_its_own_cutoff_token_positions(tmp_path: Path) -> None:
    callback = _build_callback(tmp_path, _CutoffStub())

    metrics = callback.score(max_users=None)

    assert metrics is not None
    assert metrics["recall@3"] == pytest.approx(1.0)


def test_semantic_codes_add_base_sid_metrics_without_the_collision_suffix(
    tmp_path: Path,
) -> None:
    codes = SemanticCodes(
        item_ids=torch.tensor([*CATALOG, 15]),
        codes=torch.tensor(
            [
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [2, 0, 1],
                [2, 1, 1],
            ]
        ),
        codes_per_level=(3, 2, 2),
    )
    callback = _build_callback(
        tmp_path,
        _NextItemStub(),
        semantic_codes=codes,
        semantic_base_levels=2,
    )

    metrics = callback.score(max_users=None)

    assert metrics is not None
    assert metrics["sid_exact_recall@3"] == pytest.approx(1.0)
    assert metrics["sid_prefix_recall@3_l1"] == pytest.approx(1.0)
    assert "sid_prefix_recall@3_l3" not in metrics


def test_catalog_comes_from_the_snapshot_not_the_future(tmp_path: Path) -> None:
    train_files, _ = _write_days(tmp_path)
    batch = build_item_snapshot(train_files, item_id_column="compact_item_id")

    item_ids = sorted(batch["int_columns"]["compact_item_id"].dense().tolist())
    assert item_ids == CATALOG


def test_an_id_only_catalog_scores_the_same_as_the_snapshot(tmp_path: Path) -> None:
    callback = _build_callback(
        tmp_path,
        _NextItemStub(),
        item_batch=build_catalog_batch(CATALOG, item_id_column="compact_item_id"),
    )
    state: dict = {}

    callback.on_epoch_end(state)

    metrics = _metrics(state)
    assert metrics["num_users"] == 3.0
    assert metrics["ndcg@3"] == pytest.approx(1.0)


def test_max_users_caps_the_epoch_eval(tmp_path: Path) -> None:
    callback = _build_callback(tmp_path, _NextItemStub(), max_users=2)
    state: dict = {}

    callback.on_epoch_end(state)

    assert _metrics(state)["num_users"] == 2.0


def test_score_reports_every_user_regardless_of_the_epoch_cap(tmp_path: Path) -> None:
    callback = _build_callback(tmp_path, _NextItemStub(), max_users=2)

    metrics = callback.score(max_users=None)

    assert metrics is not None
    assert metrics["num_users"] == 3.0


def test_full_user_query_snapshot_is_ordered_and_leaves_model_mode_unchanged(
    tmp_path: Path,
) -> None:
    model = _CutoffStub()
    model.train()
    callback = _build_callback(tmp_path, model, max_users=2)

    user_ids, queries = callback.full_user_query_snapshot()

    assert user_ids.tolist() == [1, 2, 3]
    assert queries.shape == (3, len(CATALOG))
    assert model.training


def test_score_with_evidence_preserves_histories_targets_frequencies_and_ranks(
    tmp_path: Path,
) -> None:
    callback = _build_callback(
        tmp_path,
        _NextItemStub(),
        train_item_frequencies={10: 1, 11: 2, 12: 1, 13: 2, 14: 1},
    )

    scored = callback.score_with_evidence(max_users=None)

    assert scored is not None
    metrics, evidence = scored
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert evidence.user_ids.tolist() == [1, 2, 3]
    assert evidence.history_offsets.tolist() == [0, 2, 4, 6]
    assert evidence.history_item_ids.tolist() == [10, 11, 11, 12, 13, 13]
    assert evidence.relevant_item_ids.tolist() == [12, 13, 14]
    assert evidence.relevant_train_frequencies.tolist() == [1, 2, 1]
    assert evidence.relevant_ranks.tolist() == [1, 1, 1]


def test_score_with_evidence_and_rankings_encodes_and_ranks_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _NextItemStub()
    model.train()
    callback = _build_callback(
        tmp_path,
        model,
        train_item_frequencies={10: 1, 11: 2, 12: 1, 13: 2, 14: 1},
    )
    calls = {"catalog": 0, "queries": 0, "topk": 0}
    encode_catalog = callback._encode_catalog
    encode_queries = callback._encode_queries
    topk = torch.topk

    def counted_catalog():
        calls["catalog"] += 1
        return encode_catalog()

    def counted_queries():
        calls["queries"] += 1
        return encode_queries()

    def counted_topk(*args, **kwargs):
        calls["topk"] += 1
        return topk(*args, **kwargs)

    monkeypatch.setattr(callback, "_encode_catalog", counted_catalog)
    monkeypatch.setattr(callback, "_encode_queries", counted_queries)
    monkeypatch.setattr(torch, "topk", counted_topk)

    scored = callback.score_with_evidence_and_rankings(max_users=None)

    assert scored is not None
    metrics, evidence, rankings = scored
    assert metrics["recall@3"] == pytest.approx(1.0)
    assert evidence.relevant_ranks.tolist() == [1, 1, 1]
    assert rankings == {
        1: (12, 13, 14),
        2: (13, 14, 10),
        3: (14, 11, 12),
    }
    assert calls == {"catalog": 1, "queries": 1, "topk": 1}
    assert model.training


def test_ranking_evidence_aligns_multi_target_ranks_in_catalog_order(
    tmp_path: Path,
) -> None:
    descending_catalog = torch.tensor(list(reversed(CATALOG)))
    callback = _build_callback(
        tmp_path,
        _FixedRankingStub(),
        item_batch={
            "int_columns": {
                "compact_item_id": FeatureValues(
                    descending_catalog,
                    torch.arange(len(CATALOG) + 1),
                )
            },
            "float_columns": {},
        },
        train_item_frequencies={item_id: item_id for item_id in CATALOG},
    )
    callback.relevance[1].add(14)

    scored = callback.score_with_evidence(max_users=None)

    assert scored is not None
    _, evidence = scored
    first_user_end = int(evidence.relevance_offsets[1])
    assert dict(
        zip(
            evidence.relevant_item_ids[:first_user_end].tolist(),
            evidence.relevant_ranks[:first_user_end].tolist(),
            strict=True,
        )
    ) == {14: 1, 12: 3}
    assert evidence.relevant_train_frequencies[:first_user_end].tolist() == [14, 12]


def test_the_catalog_is_ranked_in_full_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encoding runs in the run's own dtype; ranking must not, or eight bits of
    mantissa decide the order of near-tied items inside the top-k."""
    ranked_dtypes: list[torch.dtype] = []
    original_topk = torch.topk

    def recording_topk(scores: torch.Tensor, *args, **kwargs):
        ranked_dtypes.append(scores.dtype)
        return original_topk(scores, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", recording_topk)
    callback = _build_callback(tmp_path, _NextItemStub(), dtype=torch.bfloat16)

    callback.score(max_users=None)

    assert ranked_dtypes == [torch.float32]


def test_score_leaves_the_model_in_training_mode(tmp_path: Path) -> None:
    model = _NextItemStub()
    model.train()
    callback = _build_callback(tmp_path, model)

    callback.score(max_users=None)

    assert model.training


def test_every_n_epochs_skips_intermediate_epochs(tmp_path: Path) -> None:
    callback = _build_callback(tmp_path, _NextItemStub())
    callback.every_n_epochs = 2
    state: dict = {}

    callback.on_epoch_end(state)
    assert EXTRA_METRICS not in state

    callback.on_epoch_end(state)
    assert _metrics(state)["num_users"] == 3.0


def test_runs_against_the_real_simple_two_tower_model(tmp_path: Path) -> None:
    torch.manual_seed(0)

    def make_tower(column: str) -> Tower:
        return Tower(
            TowerInputEncoder(
                num_embeddings=32,
                embedding_dim=4,
                categorical_columns=[column],
                num_hashes=2,
            ),
            categorical_columns=[column],
        )

    model = TwoTowerModel(
        make_tower("uid"),
        make_tower("compact_item_id"),
        item_id_column="compact_item_id",
    )
    callback = _build_callback(tmp_path, model)
    state: dict = {}

    callback.on_epoch_end(state)

    metrics = _metrics(state)
    assert metrics["num_users"] == 3.0
    assert 0.0 <= metrics["ndcg@3"] <= 1.0
    assert model.training
