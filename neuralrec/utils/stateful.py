from abc import ABC
from typing import Any


class Stateful(ABC):
    _state_fields: tuple[str, ...] = ()

    def state_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._state_fields}

    def load_state_dict(self, state_dict: dict[str, Any]):
        for k in self._state_fields:
            if k in state_dict:
                setattr(self, k, state_dict[k])
            else:
                raise KeyError(f"Missing key: {k}")

        extra = set(state_dict) - set(self._state_fields)
        if extra:
            raise KeyError(f"Unexpected keys: {extra}")
