from typing import Any

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate

from neuralrec.data.transforms import Transform


class ItemIdLast(Transform):
    def __init__(self, history_key: str = "item_id", max_len: int = 128):
        self._history_key = history_key
        self._max_len = max_len

    def __call__(self, obj):
        if not isinstance(obj, dict):
            return obj
        history = obj.get(self._history_key, obj.get("item_id"))
        if history is None:
            return {"item_id": []}
        if isinstance(history, (list, tuple)):
            history = list(history)[-self._max_len :]
        else:
            history = [history]
        return {"item_id": history}


class RemapItemIds(Transform):
    def __init__(self, mapping: dict[int, int] | dict[str, int]):
        self._mapping = mapping

    def __call__(self, obj):
        if not isinstance(obj, dict) or "item_id" not in obj:
            return obj
        history = obj["item_id"]
        if isinstance(history, (list, tuple)):
            new_history = [self._mapping.get(int(x), 0) for x in history]
        else:
            new_history = [self._mapping.get(int(history), 0)]
        return {**obj, "item_id": new_history}


def pad_collate_item_id(
    batch: list[dict[str, Any]],
    key: str = "item_id",
    pad_value: int = 0,
    batch_first: bool = True,
) -> dict[str, Any]:
    if not batch:
        return default_collate(batch)

    if key not in batch[0]:
        return default_collate(batch)

    sequences = [sample[key] for sample in batch]
    padded = pad_sequence(
        sequences,
        batch_first=batch_first,
        padding_value=pad_value,
    )

    result = {key: padded}
    for k in batch[0].keys():
        if k != key:
            result[k] = default_collate([sample[k] for sample in batch])
    return result
