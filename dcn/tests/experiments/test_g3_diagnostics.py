from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dcn.models import TwoTowerLoss
from dcn.nn import (
    FrequencyContentGate,
    GlobalContentGate,
    ItemContentCatalogEncoder,
    PrecomputedEmbeddingLookup,
    PretrainedCatalogEncoder,
)
from dcn.nn.sampled_softmax import StreamingInBatchSoftmax
from experiments.g3_pretrained_item_embeddings.diagnostics import (
    G3DiagnosticTwoTowerLoss,
    G3DiagnosticsCallback,
    G3GateDiagnosticsCallback,
    _Distribution,
)
from neuralrec.utils import LOSS_DENOMINATOR


class _FixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_repr = nn.Parameter(
            torch.tensor(
                [
                    [0.2, -0.1, 0.7],
                    [0.3, 0.5, -0.2],
                    [-0.4, 0.1, 0.6],
                    [0.8, -0.2, 0.1],
                    [-0.1, 0.9, 0.3],
                    [0.5, 0.4, -0.6],
                ]
            )
        )
        self.item_repr = nn.Parameter(
            torch.tensor(
                [
                    [0.1, 0.4, -0.2],
                    [0.7, -0.3, 0.2],
                    [-0.2, 0.6, 0.5],
                    [0.4, 0.1, -0.7],
                    [0.3, 0.8, -0.1],
                    [-0.5, 0.2, 0.9],
                ]
            )
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        return {
            "query_repr": self.query_repr,
            "item_repr": self.item_repr,
            "item_ids": torch.arange(1, 7),
            "lengths": torch.tensor([3, 3]),
        }


class _RecordingSoftmax(StreamingInBatchSoftmax):
    recorded_logits: torch.Tensor

    def logits(self, *args, **kwargs) -> torch.Tensor:
        logits = super().logits(*args, **kwargs)
        self.recorded_logits = logits.detach().clone()
        return logits


def _softmax() -> _RecordingSoftmax:
    return _RecordingSoftmax(
        hash_size=7,
        num_in_batch_negatives=2,
        correction="none",
        mask_false_negatives=False,
        exclude_own_group=False,
    )


def _counts() -> torch.Tensor:
    return torch.tensor([0, 9, 1, 4, 2, 8, 3])


def test_diagnostic_loss_preserves_logits_loss_and_gradients() -> None:
    model = _FixedModel()
    baseline_model = deepcopy(model)
    baseline_softmax = _softmax()
    diagnostic_softmax = deepcopy(baseline_softmax)
    baseline = TwoTowerLoss(baseline_model, baseline_softmax)
    diagnostic = G3DiagnosticTwoTowerLoss(
        model,
        diagnostic_softmax,
        training_counts=_counts(),
    )

    torch.manual_seed(17)
    baseline_output = baseline({})
    torch.manual_seed(17)
    diagnostic_output = diagnostic({})

    assert torch.equal(
        diagnostic_softmax.recorded_logits, baseline_softmax.recorded_logits
    )
    assert torch.equal(diagnostic_output["loss"], baseline_output["loss"])
    assert torch.equal(diagnostic_output["hit_rate"], baseline_output["hit_rate"])
    assert diagnostic_output[LOSS_DENOMINATOR] == baseline_output[LOSS_DENOMINATOR]
    assert diagnostic.state_dict().keys() == baseline.state_dict().keys()
    diagnostic_scalars = {
        name: value
        for name, value in diagnostic_output.items()
        if name.startswith("g3_diagnostics/")
    }
    assert diagnostic_scalars
    assert all(
        value.ndim == 0 and not value.requires_grad
        for value in diagnostic_scalars.values()
    )

    baseline_output["loss"].backward()
    diagnostic_output["loss"].backward()
    for diagnostic_parameter, baseline_parameter in zip(
        model.parameters(), baseline_model.parameters(), strict=True
    ):
        assert torch.equal(diagnostic_parameter.grad, baseline_parameter.grad)


def test_training_frequency_slice_denominators_are_exact() -> None:
    diagnostic = G3DiagnosticTwoTowerLoss(
        _FixedModel(),
        _softmax(),
        training_counts=_counts(),
    )

    torch.manual_seed(3)
    diagnostic({})
    statistics = diagnostic.epoch_statistics()

    assert statistics["global"]["num_examples"] == 4
    assert statistics["tail"]["num_examples"] == 1
    assert statistics["mid"]["num_examples"] == 2
    assert statistics["head"]["num_examples"] == 1
    assert statistics["global"]["query_norm"]["count"] == 4
    assert statistics["global"]["positive_logit"]["count"] == 4
    assert statistics["global"]["negative_logit"]["count"] == 8
    assert statistics["tail"]["negative_logit"]["count"] == 2
    assert statistics["mid"]["negative_logit"]["count"] == 4
    assert statistics["head"]["negative_logit"]["count"] == 2


def test_evaluation_does_not_enter_the_next_training_epoch() -> None:
    diagnostic = G3DiagnosticTwoTowerLoss(
        _FixedModel(),
        _softmax(),
        training_counts=_counts(),
    )

    diagnostic.train()
    diagnostic({})
    assert diagnostic.epoch_statistics(reset=True)["global"]["num_examples"] == 4
    diagnostic.eval()
    diagnostic({})
    diagnostic.train()
    diagnostic({})

    assert diagnostic.epoch_statistics()["global"]["num_examples"] == 4


def test_gate_diagnostics_bind_counts_gradients_outputs_and_content_scaling(
    tmp_path: Path,
) -> None:
    counts = _counts()
    gate = FrequencyContentGate(counts, hidden_dim=2)
    content = PrecomputedEmbeddingLookup(
        torch.eye(6), learnable_default=False, strict=False
    )
    callback = G3GateDiagnosticsCallback(
        gate=gate,
        content_provider=content,
        training_counts=counts,
        run_log_directory=tmp_path,
    )
    gate(torch.arange(1, 7)).sum().backward()

    callback.on_before_optimizer_step({})
    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})
    document = json.loads(callback.path.read_text())
    epoch = document["epochs"][0]

    assert document["frequency_input_parity"] is True
    assert document["training_count_reference"]["length"] == 7
    assert epoch["gate_parameter_gradient_norm"]["mean"] > 0
    assert epoch["gate_output"]["global"]["count"] == 6
    assert epoch["gate_output"]["tail"]["count"] == 2
    assert epoch["gate_output"]["mid"]["count"] == 2
    assert epoch["gate_output"]["head"]["count"] == 2
    assert epoch["gate_output"]["global"]["fraction_at_or_above_0_95"] == 1
    assert epoch["raw_content_norm"]["global"]["mean"] == 1
    assert epoch["content_scaling_ratio"]["global"]["mean"] == pytest.approx(
        epoch["gate_output"]["global"]["mean"]
    )


def test_gate_diagnostics_reject_mismatched_frequency_buffer(tmp_path: Path) -> None:
    counts = _counts()
    gate = FrequencyContentGate(counts, hidden_dim=2)
    gate.standardized_log_counts[1] += 1
    content = PrecomputedEmbeddingLookup(
        torch.eye(6), learnable_default=False, strict=False
    )

    with pytest.raises(ValueError, match="standardized counts"):
        G3GateDiagnosticsCallback(
            gate=gate,
            content_provider=content,
            training_counts=counts,
            run_log_directory=tmp_path,
        )


def test_global_gate_diagnostics_mark_frequency_input_not_applicable(
    tmp_path: Path,
) -> None:
    counts = _counts()
    gate = GlobalContentGate()
    content = PrecomputedEmbeddingLookup(
        torch.eye(6), learnable_default=False, strict=False
    )
    callback = G3GateDiagnosticsCallback(
        gate=gate,
        content_provider=content,
        training_counts=counts,
        run_log_directory=tmp_path,
    )

    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})

    assert json.loads(callback.path.read_text())["frequency_input_parity"] is None


def _content_encoder(trainable: bool) -> PretrainedCatalogEncoder:
    content = PrecomputedEmbeddingLookup(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        learnable_default=False,
        strict=False,
    )
    encoder = PretrainedCatalogEncoder(content, output_dim=2, trainable=trainable)
    with torch.no_grad():
        encoder.projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        )
    return encoder


def _callback(
    run_log_directory: Path,
    catalog_encoder: nn.Module,
    *,
    counts: torch.Tensor | None = None,
    components: dict[str, nn.Module] | None = None,
) -> G3DiagnosticsCallback:
    criterion = G3DiagnosticTwoTowerLoss(
        _FixedModel(),
        _softmax(),
        training_counts=_counts() if counts is None else counts,
    )
    return G3DiagnosticsCallback(
        criterion=criterion,
        catalog_encoder=catalog_encoder,
        components={} if components is None else components,
        run_log_directory=run_log_directory,
    )


@pytest.mark.parametrize("trainable", [False, True])
def test_content_drift_distinguishes_frozen_and_trainable_tables(
    tmp_path: Path, trainable: bool
) -> None:
    encoder = _content_encoder(trainable)
    callback = _callback(
        tmp_path / str(trainable),
        encoder,
        counts=torch.tensor([0, 1, 2, 3]),
    )
    if trainable:
        assert encoder.content.embedding is not None
        with torch.no_grad():
            encoder.content.embedding.weight[2].add_(torch.tensor([1.0, 0.0, 0.0]))

    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})
    epoch = json.loads(callback.path.read_text())["epochs"][0]
    content = epoch["pretrained_content"]

    assert content["available"] is True
    assert content["trainable"] is trainable
    if trainable:
        assert content["drift_l2"]["global"]["mean"] > 0
        assert content["cosine_to_initial"]["global"]["mean"] < 1
    else:
        assert content["drift_l2"]["global"]["mean"] == 0
        assert content["cosine_to_initial"]["global"]["mean"] == 1


def test_restart_rejects_a_different_content_drift_reference(tmp_path: Path) -> None:
    encoder = _content_encoder(trainable=True)
    run_log = tmp_path / "resume"
    callback = _callback(
        run_log,
        encoder,
        counts=torch.tensor([0, 1, 2, 3]),
    )
    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})
    assert encoder.content.embedding is not None
    with torch.no_grad():
        encoder.content.embedding.weight[1].add_(1)

    with pytest.raises(ValueError, match="content drift reference"):
        _callback(
            run_log,
            encoder,
            counts=torch.tensor([0, 1, 2, 3]),
        )


def test_restart_binds_exact_training_counts_and_slice_membership(
    tmp_path: Path,
) -> None:
    encoder = _content_encoder(trainable=False)
    run_log = tmp_path / "frequency-identity"
    callback = _callback(
        run_log,
        encoder,
        counts=torch.tensor([0, 1, 2, 3]),
    )
    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})
    document = json.loads(callback.path.read_text())
    assert document["training_count_reference"]["sha256"]
    assert document["slice_membership_reference"]["sha256"]

    document["slice_membership_reference"]["sha256"] = "0" * 64
    callback.path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="slice membership reference"):
        _callback(
            run_log,
            encoder,
            counts=torch.tensor([0, 1, 2, 3]),
        )

    callback.path.unlink()
    callback = _callback(
        run_log,
        encoder,
        counts=torch.tensor([0, 1, 2, 3]),
    )
    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})
    with pytest.raises(ValueError, match="training-count reference"):
        _callback(
            run_log,
            encoder,
            counts=torch.tensor([0, 1, 2, 4]),
        )


def _serialize_epochs(path: Path, order: tuple[int, ...]) -> bytes:
    catalog = nn.Embedding(7, 2)
    component = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        catalog.weight.copy_(torch.arange(14, dtype=torch.float32).reshape(7, 2))
        component.weight.zero_()
    callback = _callback(path, catalog, components={"projection": component})
    for epoch in order:
        component.weight.grad = torch.tensor([[3.0, 4.0]])
        callback.on_before_optimizer_step({})
        callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=epoch)})
    return callback.path.read_bytes()


def test_epoch_json_is_stable_and_records_gradients_and_catalog_denominators(
    tmp_path: Path,
) -> None:
    reverse = _serialize_epochs(tmp_path / "reverse", (1, 0))
    forward = _serialize_epochs(tmp_path / "forward", (0, 1))

    assert reverse == forward
    document = json.loads(reverse)
    assert [epoch["epoch"] for epoch in document["epochs"]] == [0, 1]
    first = document["epochs"][0]
    assert first["component_gradient_norms"]["projection"]["mean"] == 5
    assert first["component_gradient_norms"]["projection"]["count"] == 1
    assert first["catalog_representation_norm"]["global"]["count"] == 6
    assert {
        name: first["catalog_representation_norm"][name]["count"]
        for name in ("tail", "mid", "head")
    } == {"tail": 2, "mid": 2, "head": 2}


def test_static_catalog_identities_are_not_recomputed_per_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = _callback(
        tmp_path,
        _content_encoder(trainable=False),
        counts=torch.tensor([0, 5, 1, 4]),
    )
    expected_manifest = callback.criterion.frequency_terciles.manifest()
    expected_training_reference = callback._training_count_reference()
    expected_slice_reference = callback._slice_membership_reference()
    expected_content_reference = callback._content_reference()

    def unexpected_recomputation(*args, **kwargs):
        raise AssertionError("static diagnostic identity was recomputed")

    monkeypatch.setattr(
        type(callback.criterion.frequency_terciles),
        "manifest",
        unexpected_recomputation,
    )
    monkeypatch.setattr(callback, "_content_reference", unexpected_recomputation)

    for epoch in range(2):
        callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=epoch)})

    document = json.loads(callback.path.read_text())
    assert document["frequency_terciles"] == expected_manifest
    assert document["training_count_reference"] == expected_training_reference
    assert document["slice_membership_reference"] == expected_slice_reference
    assert document["content_drift_reference"] == expected_content_reference


def test_cached_slice_positions_preserve_exact_distribution_statistics(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path, nn.Embedding(7, 2))
    values = torch.tensor([1.5, float("nan"), -2.0, 4.0, float("inf"), 3.0])

    actual = callback._sliced_distributions(values)
    expected = {}
    for name, selected in {
        "global": values,
        **{
            name: values[
                callback.criterion.frequency_terciles.item_ids(
                    name, torch.device("cpu")
                )
                - 1
            ]
            for name in ("tail", "mid", "head")
        },
    }.items():
        distribution = _Distribution()
        distribution.update(selected)
        expected[name] = distribution.statistics()

    assert actual == expected


def test_catalog_table_gradient_norms_are_split_by_item_frequency(
    tmp_path: Path,
) -> None:
    content = PrecomputedEmbeddingLookup(
        torch.eye(3), learnable_default=False, strict=False
    )
    catalog = ItemContentCatalogEncoder(
        num_items=3,
        item_dim=2,
        content=content,
        output_dim=2,
        trainable_content=True,
    )
    callback = _callback(
        tmp_path,
        catalog,
        counts=torch.tensor([0, 1, 2, 3]),
        components={"catalog_encoder": catalog},
    )
    catalog.item_embedding.weight.grad = torch.tensor(
        [[0.0, 0.0], [3.0, 4.0], [0.0, 2.0], [0.0, 0.0]]
    )
    assert catalog.content.embedding is not None
    catalog.content.embedding.weight.grad = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    callback.on_before_optimizer_step({})
    callback.on_epoch_end({"train_runner": SimpleNamespace(current_epoch=0)})

    gradients = json.loads(callback.path.read_text())["epochs"][0][
        "catalog_table_gradient_norms"
    ]
    item = gradients["item_embedding.weight"]
    pretrained = gradients["content.embedding.weight"]
    assert item["global"]["all_row_exposure_weighted_norm"]["count"] == 3
    assert item["tail"]["all_row_exposure_weighted_norm"]["mean"] == 5
    assert item["mid"]["all_row_exposure_weighted_norm"]["mean"] == 2
    assert item["head"]["all_row_exposure_weighted_norm"]["mean"] == 0
    assert item["global"]["active_row_count"]["mean"] == 2
    assert item["global"]["active_row_fraction"]["mean"] == pytest.approx(2 / 3)
    assert item["global"]["conditional_on_active_row_norm"]["count"] == 2
    assert item["global"]["conditional_on_active_row_norm"]["mean"] == 3.5
    assert item["head"]["active_row_count"]["mean"] == 0
    assert item["head"]["conditional_on_active_row_norm"]["count"] == 0
    assert pretrained["tail"]["conditional_on_active_row_norm"]["mean"] == 6
    assert pretrained["mid"]["conditional_on_active_row_norm"]["mean"] == 8
    assert pretrained["head"]["active_row_fraction"]["mean"] == 0
