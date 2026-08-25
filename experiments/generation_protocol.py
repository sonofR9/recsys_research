from typing import Any

from dcn.config.settings import DayRangeConfig
from dcn.datasets.yambda import UserSample, YambdaSize


FINAL_YAMBDA_SIZE: YambdaSize = "500m"
CORE_MIN_INTERACTIONS_PER_ITEM = 5
TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60


def generation_protocol(
    *,
    event_type_filter: str | None,
    window: str,
    size: YambdaSize = FINAL_YAMBDA_SIZE,
    user_sample: UserSample | None = None,
) -> dict[str, Any]:
    return {
        "size": size,
        "user_sample": user_sample,
        "event_type_filter": event_type_filter,
        "min_item_interactions_per_item": CORE_MIN_INTERACTIONS_PER_ITEM,
        "drop_unmapped_items": True,
        "window": window,
        "validation_interval_seconds": TEST_INTERVAL_SECONDS,
        "day_range": DayRangeConfig(start_day=0, end_day=300),
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
    }
