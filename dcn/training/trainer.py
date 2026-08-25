import random
from time import perf_counter

from torch.utils.data import DataLoader, Dataset

from dcn.data.dataset_manager import DatasetManager
from neuralrec.run.callbacks import Callback, ValidationCallback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import add_metrics


class EmptyDataset(Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("This dataset is empty")


class DayByDayTrainer(TrainRunner):
    _state_fields: tuple[str, ...] = (
        *TrainRunner._state_fields,
        "current_day",
        "current_pretrain_epoch",
        "end_day",
        "last_available_day",
        "train_batch_size",
        "val_batch_size",
        "num_workers",
        "prefetch_factor",
        "pretrain_days",
        "pretrain_num_epochs",
    )

    def __init__(
        self,
        dataset_manager: DatasetManager,
        start_day: int,
        end_day: int,
        val_callback: ValidationCallback,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        num_workers: int = 16,
        prefetch_factor: int = 2,
        pretrain_days: int = 0,
        pretrain_num_epochs: int = 1,
        pretrain_shuffle_days: bool = True,
        callbacks: list[Callback] | None = None,
        **kwargs,
    ):
        super().__init__(callbacks=[val_callback, *(callbacks or [])], **kwargs)

        self.state["dataset_manager"] = dataset_manager
        self.dataset_manager = dataset_manager

        available_days = self.dataset_manager.get_available_days()
        assert start_day >= available_days[0] and end_day <= available_days[-1]

        self.start_day = start_day
        self.current_day = start_day
        self.end_day = end_day
        self.val_callback = val_callback
        self.last_available_day = available_days[-1]

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.pretrain_days = pretrain_days
        self.pretrain_num_epochs = pretrain_num_epochs
        self.pretrain_shuffle_days = pretrain_shuffle_days
        self.current_pretrain_epoch = 0

    def _create_dataloader(
        self, day: int, batch_size: int, shuffle: bool
    ) -> DataLoader:
        if day > self.last_available_day:
            return DataLoader(EmptyDataset(), batch_size=1)
        return self.dataset_manager.create_dataloader(
            day,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def _train_day(self, day: int, epoch_id: int) -> None:
        dataset_start = perf_counter()

        self.current_day = day
        self.state["current_day"] = day
        add_metrics(self.state, "epoch", {"current_day": day})

        train_loader = self._create_dataloader(day, self.train_batch_size, shuffle=True)

        add_metrics(
            self.state,
            "timing",
            {"dataset_creation_time": perf_counter() - dataset_start},
        )

        self.train_epoch(epoch_id, train_loader)

    def _run_pretrain_epoch(
        self, pretrain_epoch: int, pretrain_day_list: list[int], pretrain_end_day: int
    ):
        self.current_pretrain_epoch = pretrain_epoch
        self.state["current_pretrain_epoch"] = pretrain_epoch
        self.state["is_pretrain_epoch_end"] = False
        add_metrics(self.state, "epoch", {"current_pretrain_epoch": pretrain_epoch})

        # FIXME(sonofr): add freeze to part of the model
        days_order = pretrain_day_list.copy()
        if self.pretrain_shuffle_days:
            random.shuffle(days_order)

        self.val_callback.val_loader = None
        for day in days_order:
            self._train_day(day, self.global_step)

        self.state["is_pretrain_epoch_end"] = True
        self.val_callback.val_loader = self._create_dataloader(
            pretrain_end_day + 1, self.val_batch_size, shuffle=False
        )
        self._fire_callbacks("on_epoch_end", self.state)

    def _run_pretrain_phase(self) -> int:
        if self.pretrain_days <= 0:
            return self.start_day

        pretrain_end_day = self.start_day + self.pretrain_days - 1
        pretrain_day_list = list(range(self.start_day, pretrain_end_day + 1))

        for pretrain_epoch in range(
            self.current_pretrain_epoch, self.pretrain_num_epochs
        ):
            self._run_pretrain_epoch(
                pretrain_epoch, pretrain_day_list, pretrain_end_day
            )

        return pretrain_end_day + 1

    def _run_online_phase(self, online_start_day: int):
        self.state["is_pretrain_epoch_end"] = True
        for day in range(online_start_day, self.end_day + 1):
            val_day = min(day + 1, self.last_available_day + 1)
            self.val_callback.val_loader = self._create_dataloader(
                val_day, self.val_batch_size, shuffle=False
            )
            self._train_day(day, day)

    def train(self) -> None:
        self._fire_callbacks("on_train_begin", self.state)
        online_start_day = self._run_pretrain_phase()
        self._run_online_phase(online_start_day)
        self._fire_callbacks("on_train_end", self.state)
