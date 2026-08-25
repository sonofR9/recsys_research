from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable

from torch import nn


# What callers actually pass is a raw nn.Module subclass (nn.GELU, nn.BatchNorm1d),
# called to build the layer — these aliases say so.
ActivationFactory = Callable[[], nn.Module]
NormalizationFactory = Callable[[int], nn.Module]
FFNFactory = Callable[[int], nn.Module]
# FIXME(sashanova): ffn factory


@dataclass
class OutputDims:
    dims: dict[str, int]

    # FIXME: attribute access for dims hides typos behind AttributeError;
    # consider exposing them as plain keys instead of dynamic attributes.
    def __getattr__(self, name: str) -> int:
        # Out of __dict__, not off self: `dims` itself misses while copy and
        # pickle rebuild the object, and __getattr__ would recurse.
        dims = self.__dict__.get("dims", {})
        if name in dims:
            return dims[name]
        raise AttributeError(f"No dimension named '{name}'")


class ModuleWithDim(nn.Module):
    @property
    @abstractmethod
    def out_dim(self) -> "int | OutputDims": ...
