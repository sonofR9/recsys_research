import torch

from dcn.training import OPTIMIZER_GROUP_ID, register_stable_optimizer_groups


def _optimizer(
    groups: tuple[tuple[str, torch.nn.Parameter, float], ...],
) -> torch.optim.Optimizer:
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter], "lr": rate, "schedule_group": name}
            for name, parameter, rate in groups
        ],
        momentum=0.9,
    )
    register_stable_optimizer_groups(optimizer)
    return optimizer


def test_checkpoint_groups_restore_by_identity_not_position() -> None:
    source_embedding = torch.nn.Parameter(torch.tensor(1.0))
    source_deep = torch.nn.Parameter(torch.tensor(2.0))
    source = _optimizer(
        (
            ("embedding", source_embedding, 0.064),
            ("deep", source_deep, 0.006),
        )
    )
    source_embedding.grad = torch.tensor(3.0)
    source_deep.grad = torch.tensor(7.0)
    source.step()

    target_embedding = torch.nn.Parameter(torch.tensor(1.0))
    target_deep = torch.nn.Parameter(torch.tensor(2.0))
    target = _optimizer(
        (
            ("deep", target_deep, 0.1),
            ("embedding", target_embedding, 0.1),
        )
    )
    target.load_state_dict(source.state_dict())

    assert [group["schedule_group"] for group in target.param_groups] == [
        "deep",
        "embedding",
    ]
    assert [group["lr"] for group in target.param_groups] == [0.006, 0.064]
    assert target.state[target_deep]["momentum_buffer"] == 7.0
    assert target.state[target_embedding]["momentum_buffer"] == 3.0


def test_duplicate_logical_schedule_roles_get_unique_checkpoint_identities() -> None:
    parameter_a = torch.nn.Parameter(torch.tensor(1.0))
    parameter_b = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [parameter_a], "schedule_group": "deep"},
            {"params": [parameter_b], "schedule_group": "deep"},
        ],
        lr=0.1,
    )

    register_stable_optimizer_groups(optimizer)

    assert [group["schedule_group"] for group in optimizer.param_groups] == [
        "deep",
        "deep",
    ]
    identities = [group[OPTIMIZER_GROUP_ID] for group in optimizer.param_groups]
    assert len(set(identities)) == 2


def test_checkpoint_restores_reordered_multiple_deep_groups() -> None:
    source_parameters = [
        torch.nn.Parameter(torch.tensor(value)) for value in (1.0, 2.0, 3.0)
    ]
    source = _optimizer(
        (
            ("embedding", source_parameters[0], 0.064),
            ("deep", source_parameters[1], 0.006),
            ("deep", source_parameters[2], 0.003),
        )
    )
    for parameter, gradient in zip(source_parameters, (5.0, 7.0, 11.0)):
        parameter.grad = torch.tensor(gradient)
    source.step()

    target_parameters = [
        torch.nn.Parameter(torch.tensor(value)) for value in (1.0, 2.0, 3.0)
    ]
    target = _optimizer(
        (
            ("embedding", target_parameters[0], 0.1),
            ("deep", target_parameters[1], 0.1),
            ("deep", target_parameters[2], 0.1),
        )
    )
    target.param_groups[:] = [
        target.param_groups[2],
        target.param_groups[0],
        target.param_groups[1],
    ]
    target.load_state_dict(source.state_dict())

    assert [group["lr"] for group in target.param_groups] == [0.003, 0.064, 0.006]
    assert target.state[target_parameters[2]]["momentum_buffer"] == 11.0
    assert target.state[target_parameters[0]]["momentum_buffer"] == 5.0
    assert target.state[target_parameters[1]]["momentum_buffer"] == 7.0
