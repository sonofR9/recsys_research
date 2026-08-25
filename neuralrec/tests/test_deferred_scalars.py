import torch

from neuralrec.utils import DeferredScalars


def test_drains_the_values_it_was_given() -> None:
    scalars = DeferredScalars()

    scalars.add(0, {"loss": torch.tensor(2.0), "hit_rate": torch.tensor(0.25)})
    scalars.add(1, {"loss": torch.tensor(1.0), "hit_rate": torch.tensor(0.75)})

    assert scalars.drain() == [
        (0, {"loss": 2.0, "hit_rate": 0.25}),
        (1, {"loss": 1.0, "hit_rate": 0.75}),
    ]


def test_holds_tensors_until_drained() -> None:
    scalars = DeferredScalars()
    value = torch.tensor(3.0, requires_grad=True) * 2

    scalars.add(0, {"loss": value})

    assert not scalars.resolved


def test_buffered_tensors_carry_no_graph() -> None:
    scalars = DeferredScalars()
    value = torch.tensor(3.0, requires_grad=True) * 2

    scalars.add(0, {"loss": value})

    assert all(
        not tensor.requires_grad
        for _, values in scalars.pending
        for tensor in values.values()
    )


def test_drops_what_is_not_a_scalar() -> None:
    scalars = DeferredScalars()

    scalars.add(0, {"loss": torch.tensor(1.0), "ranked": torch.arange(4), "name": "x"})

    assert scalars.drain() == [(0, {"loss": 1.0})]


def test_keeps_plain_numbers() -> None:
    scalars = DeferredScalars()

    scalars.add(0, {"lr": 0.001, "loss": torch.tensor(1.0)})

    assert scalars.drain() == [(0, {"lr": 0.001, "loss": 1.0})]


def test_ignores_named_keys() -> None:
    scalars = DeferredScalars(to_ignore=["hit_rate"])

    scalars.add(0, {"loss": torch.tensor(1.0), "hit_rate": torch.tensor(0.5)})

    assert scalars.drain() == [(0, {"loss": 1.0})]


def test_a_step_with_nothing_to_record_is_not_kept() -> None:
    scalars = DeferredScalars()

    scalars.add(0, {"ranked": torch.arange(4)})

    assert scalars.drain() == []


def test_draining_empties_the_buffer() -> None:
    scalars = DeferredScalars()
    scalars.add(0, {"loss": torch.tensor(1.0)})

    scalars.drain()

    assert scalars.drain() == []


def test_means_average_each_key_over_the_steps_that_reported_it() -> None:
    scalars = DeferredScalars()
    scalars.add(0, {"loss": torch.tensor(2.0), "hit_rate": torch.tensor(0.5)})
    scalars.add(1, {"loss": torch.tensor(4.0)})

    assert DeferredScalars.means(scalars.drain()) == {"loss": 3.0, "hit_rate": 0.5}
