import torch
from torch import nn

from neuralrec.run.callbacks import ResourceUsageCallback
from neuralrec.utils import EXTRA_METRICS


def _model() -> nn.Module:
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    model[1].bias.requires_grad_(False)
    return model


def _usage(state: dict) -> dict[str, float]:
    return state[EXTRA_METRICS]["resources"]


def test_counts_parameters_of_the_model_it_watches() -> None:
    state: dict = {}

    ResourceUsageCallback(model=_model()).on_epoch_end(state)

    assert _usage(state)["params_total"] == 4 * 3 + 3 + 3 * 2 + 2
    assert _usage(state)["params_trainable"] == 4 * 3 + 3 + 3 * 2


def test_separates_the_lookup_tables_from_the_rest_of_the_model() -> None:
    """An item table can be three orders of magnitude larger than the stack that
    reads it, and a total that mixes them says nothing about either."""
    model = _model()
    state: dict = {}

    ResourceUsageCallback(
        model=model, embedding_parameters=list(model[0].parameters())
    ).on_epoch_end(state)

    assert _usage(state)["params_embedding"] == 4 * 3 + 3
    assert _usage(state)["params_deep"] == 3 * 2 + 2


def test_reports_peak_memory_only_where_it_can_be_measured() -> None:
    state: dict = {}

    ResourceUsageCallback(model=_model()).on_epoch_end(state)

    assert ("peak_memory_gb" in _usage(state)) == torch.cuda.is_available()
