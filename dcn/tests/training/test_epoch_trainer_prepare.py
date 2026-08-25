from itertools import chain

import torch
from torch.utils.data import DataLoader, Dataset

from dcn.training import EpochTrainer


class _Dataset(Dataset):
    def __init__(self) -> None:
        self.read: list[int] = []

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> torch.Tensor:
        self.read.append(index)
        return torch.tensor([float(index)])


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.batches = 0

    def forward(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        self.batches += 1
        return {"loss": (batch * self.weight).sum()}


def test_prepare_loads_first_batch_without_consuming_it() -> None:
    dataset = _Dataset()
    loader = DataLoader(dataset, batch_size=2)
    model = _Model()
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=loader,
        num_epochs=1,
    )

    trainer.prepare()

    assert dataset.read == [0, 1]
    assert model.batches == 0

    trainer.train()

    assert dataset.read == [0, 1, 2, 3]
    assert model.batches == 2


def test_prepare_keeps_a_first_batch_loaded_before_trainer_construction() -> None:
    dataset = _Dataset()
    loader = DataLoader(dataset, batch_size=2)
    iterator = iter(loader)
    prepared_iterator = chain((next(iterator),), iterator)
    model = _Model()
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=loader,
        prepared_train_iterator=prepared_iterator,
        num_epochs=1,
    )

    trainer.prepare()

    assert dataset.read == [0, 1]
    trainer.train()
    assert dataset.read == [0, 1, 2, 3]


def test_discarding_prepared_resources_reloads_after_gpu_wait() -> None:
    dataset = _Dataset()
    loader = DataLoader(dataset, batch_size=2)
    model = _Model()
    trainer = EpochTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=loader,
        num_epochs=1,
    )
    trainer.prepare()

    trainer.discard_prepared_resources()
    trainer.prepare()
    trainer.train()

    assert dataset.read == [0, 1, 0, 1, 2, 3]
    assert model.batches == 2
