import numpy as np
import torch


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class ToNumpy(Transform):
    """Convert all lists to numpy"""

    def __init__(self, dtype=np.int64):
        super().__init__()
        self._dtype = dtype

    def __call__(self, sample):
        res = {}
        for key, value in sample.items():
            if isinstance(value, dict):
                res[key] = self.__call__(value)
            elif isinstance(value, list):
                res[key] = np.array(value, dtype=self._dtype)
            else:
                res[key] = value
        return res


class ToTorch(Transform):
    """Convert all lists or numpy arrays in torch tensors."""

    def __call__(self, obj):
        if isinstance(obj, dict):
            return {key: self.__call__(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return torch.tensor(obj)
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        else:
            return obj


def move_to_device(obj, device: torch.device | str, non_blocking: bool = False):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # NamedTuple (e.g. FeatureValues) takes its fields positionally.
        return type(obj)(*(move_to_device(x, device, non_blocking) for x in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device, non_blocking) for x in obj)
    return obj


class ToDevice(Transform):
    """Move obj to device."""

    def __init__(
        self,
        device: torch.device | str,
        non_blocking: bool = False,
    ):
        self._device = device
        self._non_blocking = non_blocking

    def __call__(self, obj):
        return move_to_device(obj, self._device, self._non_blocking)
