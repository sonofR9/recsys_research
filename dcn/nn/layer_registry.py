from typing import (
    Protocol,
    Self,
    Type,
)
import inspect


class ConfigurableLayer(Protocol):
    @classmethod
    def from_config(cls, config: dict) -> Self: ...


class LayerRegistry:
    def __init__(self):
        self._registry: dict[str, Type[ConfigurableLayer]] = {}

    def register(self, cls: Type[ConfigurableLayer]) -> Type[ConfigurableLayer]:
        name = cls.__name__

        if name in self._registry:
            raise ValueError(f"Layer with name '{name}' is already registered!")

        self._registry[name] = cls
        return cls

    def get(self, name: str) -> Type[ConfigurableLayer]:
        if name not in self._registry:
            raise KeyError(
                f"Layer '{name}' is not found in registry. "
                f"Available layers: {list(self._registry.keys())}"
            )
        return self._registry[name]


layer_registry = LayerRegistry()


def build_layer(type: str, config: dict):
    layer_class = layer_registry.get(type)

    sig = inspect.signature(layer_class.__init__)

    resolved_kwargs = {
        name: config[name]
        for name in sig.parameters
        if name != "self" and name in config
    }

    return layer_class(**resolved_kwargs)


# FIXME(sashanovak): add to all classes (see SwiGlu)
