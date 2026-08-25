import torch


class CombinedOptimizer(torch.optim.Optimizer):
    """Steps several optimizers as one."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]):
        assert optimizers, "CombinedOptimizer needs at least one optimizer"
        self.optimizers = optimizers
        self._enabled = [True] * len(optimizers)
        super().__init__([{"params": []}], defaults={})
        self._adopt_param_groups()

    def _adopt_param_groups(self) -> None:
        self.param_groups = [
            group for opt in self.optimizers for group in opt.param_groups
        ]

    def set_enabled(self, index: int, enabled: bool) -> None:
        self._enabled[index] = enabled

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        for opt, enabled in zip(self.optimizers, self._enabled):
            if enabled:
                opt.step()

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts: list[dict]):
        for opt, state_dict in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state_dict)
        # torch replaces a group dict rather than updating it, so the groups
        # published here stop being the ones the sub-optimizers step.
        self._adopt_param_groups()
