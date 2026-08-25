from dcn.datasets.yambda import UserSample
from experiments.generation_protocol import (
    CORE_MIN_INTERACTIONS_PER_ITEM,
    FINAL_YAMBDA_SIZE,
    TEST_INTERVAL_SECONDS,
    generation_protocol,
)


def test_generation_protocol_defaults_to_full_homework_dataset() -> None:
    protocol = generation_protocol(event_type_filter="like", window="next_item")

    assert FINAL_YAMBDA_SIZE == "500m"
    assert CORE_MIN_INTERACTIONS_PER_ITEM == 5
    assert TEST_INTERVAL_SECONDS == 7 * 24 * 60 * 60
    assert protocol == {
        "size": "500m",
        "user_sample": None,
        "event_type_filter": "like",
        "min_item_interactions_per_item": 5,
        "drop_unmapped_items": True,
        "window": "next_item",
        "validation_interval_seconds": 7 * 24 * 60 * 60,
        "day_range": protocol["day_range"],
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
    }
    assert protocol["day_range"].start_day == 0
    assert protocol["day_range"].end_day == 300


def test_generation_protocol_allows_explicit_tuning_scale() -> None:
    sample = UserSample(max_users=1_000)

    protocol = generation_protocol(
        event_type_filter="like",
        window="next_item",
        size="50m",
        user_sample=sample,
    )

    assert protocol["size"] == "50m"
    assert protocol["user_sample"] == sample
