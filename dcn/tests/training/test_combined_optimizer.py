import torch

from dcn.training.combined_optimizer import CombinedOptimizer


def test_disabled_optimizer_does_not_update_its_parameters() -> None:
    enabled_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    disabled_parameter = torch.nn.Parameter(torch.tensor([10.0, 20.0, 30.0]))

    enabled_optimizer = torch.optim.SGD([enabled_parameter], lr=0.1)
    disabled_optimizer = torch.optim.SGD([disabled_parameter], lr=0.1)

    combined = CombinedOptimizer([enabled_optimizer, disabled_optimizer])
    combined.set_enabled(1, False)

    enabled_parameter.grad = torch.ones_like(enabled_parameter)
    disabled_parameter.grad = torch.ones_like(disabled_parameter)

    enabled_before = enabled_parameter.detach().clone()
    disabled_before = disabled_parameter.detach().clone()

    combined.step()

    assert not torch.equal(enabled_parameter.detach(), enabled_before)
    assert torch.equal(disabled_parameter.detach(), disabled_before)


def test_a_rate_set_after_loading_a_checkpoint_still_reaches_the_parameters() -> None:
    """Whoever schedules the learning rate writes it through `param_groups`."""
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    combined = CombinedOptimizer([torch.optim.SGD([parameter], lr=0.1)])
    combined.load_state_dict(combined.state_dict())

    combined.param_groups[0]["lr"] = 0.5

    assert combined.optimizers[0].param_groups[0]["lr"] == 0.5
