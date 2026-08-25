import threading
import time
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from data.counters import DecayConfig, EmaCounter, FieldConfig
from dcn.data.dataset_manager import DatasetManager
from utils.global_config import config as global_config

SECONDS_IN_DAY = 86_400
COLUMNS = ["item_id", "uid", "target_like"]


@pytest.fixture
def main_parquet(tmp_path: Path) -> Path:
    rows = []
    for day in range(3):
        for position, (user, item, event) in enumerate(
            [(100, 1, "like"), (101, 2, "listen")]
        ):
            rows.append(
                {
                    "item_id": item + day,
                    "uid": user,
                    "event_type": event,
                    "timestamp": day * SECONDS_IN_DAY + position,
                    "target_like": float(event == "like"),
                    "day": date(2023, 1, day + 1),
                }
            )

    path = tmp_path / "main.parquet"
    pl.DataFrame(rows).write_parquet(path)
    global_config.initialize(tmp_path)
    return path


def _like_counter(keys: list[str], half_life_days: float) -> EmaCounter:
    return EmaCounter(
        keys=keys,
        fields=[
            FieldConfig(
                name="like",
                condition=pl.col("event_type") == "like",
                decays=[DecayConfig(half_life_days=half_life_days)],
            )
        ],
        cache_dir=global_config.counters_path,
    )


def _manager(
    main_parquet: Path,
    *,
    cache_dir: Path | None = None,
    counters: list[EmaCounter] = [],
) -> DatasetManager:
    counter_columns = [c for counter in counters for c in counter.get_output_columns()]
    return DatasetManager(
        main_parquet=main_parquet,
        columns=COLUMNS,
        counter_columns=counter_columns,
        counters=counters,
        cache_dir=cache_dir or (main_parquet.parent / "cache"),
    )


def test_days_are_split_and_addressable(main_parquet: Path) -> None:
    manager = _manager(main_parquet)

    assert manager.get_available_days() == [0, 1, 2]
    assert len(manager.create_dataset([0, 1])) == 4
    assert len(manager.create_dataset(0)) == 2


def test_concurrent_managers_build_a_shared_cold_cache_once(
    main_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = DatasetManager._prepare_data
    active = 0
    maximum = 0
    builds = 0
    guard = threading.Lock()

    def prepare(
        manager: DatasetManager, source: Path, invalidate_cache: bool = False
    ) -> None:
        nonlocal active, maximum, builds
        with guard:
            active += 1
            maximum = max(maximum, active)
            builds += 1
        try:
            time.sleep(0.05)
            original(manager, source, invalidate_cache)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(DatasetManager, "_prepare_data", prepare)
    managers: list[DatasetManager] = []
    errors: list[BaseException] = []

    def build() -> None:
        try:
            managers.append(_manager(main_parquet))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(managers) == 2
    assert builds == 1
    assert maximum == 1


def test_dataloader_batches_the_requested_days(main_parquet: Path) -> None:
    manager = _manager(main_parquet)

    batches = list(manager.create_dataloader(days=[0, 1], batch_size=2, num_workers=0))

    assert len(batches) == 2


def test_worker_processes_can_read_their_batches(main_parquet: Path) -> None:
    manager = _manager(main_parquet)

    batches = list(
        manager.create_dataloader(
            days=[0, 1], batch_size=2, num_workers=2, prefetch_factor=2
        )
    )

    assert len(batches) == 2


def test_invalid_day_raises(main_parquet: Path) -> None:
    manager = _manager(main_parquet)

    with pytest.raises(AssertionError, match="Day 99 not found"):
        manager.create_dataset(99)


def test_counters_reach_training_as_one_packed_column(main_parquet: Path) -> None:
    counter = _like_counter(["uid"], 7)

    manager = _manager(main_parquet, counters=[counter])
    batch = next(iter(manager.create_dataloader(days=[1], batch_size=2, num_workers=0)))

    assert manager.dense_columns == ["counters"]
    packed = batch["float_columns"]["counters"].matrix()
    assert packed.shape == (2, len(counter.get_output_columns()))


def test_different_counter_sets_do_not_share_a_cache(main_parquet: Path) -> None:
    cache_dir = main_parquet.parent / "shared_cache"

    first = _manager(
        main_parquet, cache_dir=cache_dir, counters=[_like_counter(["uid"], 7)]
    )
    second = _manager(
        main_parquet, cache_dir=cache_dir, counters=[_like_counter(["item_id"], 30)]
    )

    assert first.metadata_file != second.metadata_file
    assert first.day_to_path[0] != second.day_to_path[0]
    assert first.original_day_to_path == second.original_day_to_path


def test_state_dict_round_trip(main_parquet: Path) -> None:
    manager = _manager(main_parquet)

    restored = _manager(main_parquet)
    restored.load_state_dict(manager.state_dict())

    assert restored.day_to_path == manager.day_to_path
