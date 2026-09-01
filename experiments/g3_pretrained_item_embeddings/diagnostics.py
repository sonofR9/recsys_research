from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dcn.models.sequence_targets import NextItemTargets
from dcn.nn.sampled_softmax import InBatchSampledSoftmaxLoss
from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import LOSS_DENOMINATOR

_SLICE_NAMES = ("tail", "mid", "head")
_SCOPES = ("global", *_SLICE_NAMES)
_DIAGNOSTICS_FILENAME = "g3_training_diagnostics.json"
_DIAGNOSTICS_SCHEMA_VERSION = 2
_GATE_DIAGNOSTICS_FILENAME = "g3_gate_diagnostics.json"


def build_frequency_identity(training_counts: torch.Tensor) -> dict[str, object]:
    terciles = _FrequencyTerciles.from_tensor(training_counts)
    return {
        "frequency_terciles": terciles.manifest(),
        "training_count_reference": _integer_sequence_reference(terciles.counts),
        "slice_membership_reference": _integer_sequence_reference(
            terciles.slice_indices
        ),
    }


@dataclass(frozen=True)
class _FrequencyTerciles:
    counts: tuple[int, ...]
    slice_indices: tuple[int, ...]

    @classmethod
    def from_tensor(cls, training_counts: torch.Tensor) -> _FrequencyTerciles:
        if training_counts.ndim != 1 or len(training_counts) < 4:
            raise ValueError(
                "training counts need unknown ID 0 and at least three items"
            )
        if training_counts.is_floating_point() or training_counts.dtype == torch.bool:
            raise ValueError("training counts must be integers")
        values = tuple(int(value) for value in training_counts.detach().cpu().tolist())
        if values[0] != 0 or any(value < 0 for value in values):
            raise ValueError(
                "training counts must have unknown count zero and be nonnegative"
            )
        known_ids = range(1, len(values))
        ordered = sorted(known_ids, key=lambda item_id: (values[item_id], item_id))
        indices = [-1] * len(values)
        for rank, item_id in enumerate(ordered):
            indices[item_id] = min(2, 3 * rank // len(ordered))
        return cls(values, tuple(indices))

    @property
    def num_items(self) -> int:
        return len(self.counts) - 1

    def item_ids(self, name: str, device: torch.device) -> torch.Tensor:
        index = _SLICE_NAMES.index(name)
        return torch.tensor(
            [
                item_id
                for item_id in range(1, len(self.slice_indices))
                if self.slice_indices[item_id] == index
            ],
            device=device,
        )

    def manifest(self) -> dict[str, object]:
        slices = {}
        for name in _SLICE_NAMES:
            item_ids = self.item_ids(name, torch.device("cpu")).tolist()
            slices[name] = {
                "num_items": len(item_ids),
                "training_interactions": sum(
                    self.counts[item_id] for item_id in item_ids
                ),
            }
        return {"num_catalog_items": self.num_items, "slices": slices}


class _Distribution:
    def __init__(self) -> None:
        self.count: torch.Tensor | None = None
        self.nonfinite_count: torch.Tensor | None = None
        self.total: torch.Tensor | None = None
        self.squared_total: torch.Tensor | None = None
        self.minimum: torch.Tensor | None = None
        self.maximum: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        flattened = values.detach().reshape(-1)
        if flattened.numel() == 0:
            return
        finite = torch.isfinite(flattened)
        selected = flattened.float()
        finite_count = finite.sum()
        nonfinite_count = finite_count.new_tensor(flattened.numel()) - finite_count
        selected_or_zero = torch.where(finite, selected, 0.0)
        total = selected_or_zero.sum().double()
        squared_total = selected_or_zero.square().sum().double()
        minimum = torch.where(finite, selected, math.inf).amin()
        maximum = torch.where(finite, selected, -math.inf).amax()
        if self.count is None:
            self.count = finite_count
            self.nonfinite_count = nonfinite_count
            self.total = total
            self.squared_total = squared_total
            self.minimum = minimum
            self.maximum = maximum
            return
        assert self.nonfinite_count is not None
        assert self.total is not None
        assert self.squared_total is not None
        assert self.minimum is not None
        assert self.maximum is not None
        self.count += finite_count
        self.nonfinite_count += nonfinite_count
        self.total += total
        self.squared_total += squared_total
        self.minimum = torch.minimum(self.minimum, minimum)
        self.maximum = torch.maximum(self.maximum, maximum)

    def statistics(self) -> dict[str, int | float | None]:
        count = 0 if self.count is None else int(self.count)
        nonfinite_count = (
            0 if self.nonfinite_count is None else int(self.nonfinite_count)
        )
        if count == 0:
            return {
                "count": 0,
                "nonfinite_count": nonfinite_count,
                "mean": None,
                "standard_deviation": None,
                "minimum": None,
                "maximum": None,
            }
        assert self.total is not None
        assert self.squared_total is not None
        assert self.minimum is not None
        assert self.maximum is not None
        mean = float(self.total) / count
        variance = max(0.0, float(self.squared_total) / count - mean * mean)
        return {
            "count": count,
            "nonfinite_count": nonfinite_count,
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


class _TrainingScope:
    def __init__(self) -> None:
        self.num_examples = 0
        self.query_norm = _Distribution()
        self.positive_logit = _Distribution()
        self.negative_logit = _Distribution()

    def update(self, query_repr: torch.Tensor, logits: torch.Tensor) -> None:
        self.num_examples += query_repr.shape[0]
        self.query_norm.update(query_repr.norm(dim=-1))
        self.positive_logit.update(logits[:, 0])
        self.negative_logit.update(logits[:, 1:])

    def statistics(self) -> dict[str, object]:
        return {
            "num_examples": self.num_examples,
            "query_norm": self.query_norm.statistics(),
            "positive_logit": self.positive_logit.statistics(),
            "negative_logit": self.negative_logit.statistics(),
        }


class _RowGradientScope:
    def __init__(self) -> None:
        self.all_row_exposure_weighted_norm = _Distribution()
        self.conditional_on_active_row_norm = _Distribution()
        self.active_row_count = _Distribution()
        self.active_row_fraction = _Distribution()

    def update(self, row_norms: torch.Tensor) -> None:
        self.all_row_exposure_weighted_norm.update(row_norms)
        active = row_norms > 0
        self.conditional_on_active_row_norm.update(row_norms[active])
        active_count = active.sum().reshape(1)
        self.active_row_count.update(active_count)
        self.active_row_fraction.update(
            active_count.float() / max(1, row_norms.numel())
        )

    def statistics(self) -> dict[str, object]:
        return {
            "all_row_exposure_weighted_norm": (
                self.all_row_exposure_weighted_norm.statistics()
            ),
            "conditional_on_active_row_norm": (
                self.conditional_on_active_row_norm.statistics()
            ),
            "active_row_count": self.active_row_count.statistics(),
            "active_row_fraction": self.active_row_fraction.statistics(),
        }


class G3DiagnosticTwoTowerLoss(nn.Module):
    """Two-tower sampled-softmax loss with graph-independent G3 measurements."""

    def __init__(
        self,
        model: nn.Module,
        loss: InBatchSampledSoftmaxLoss,
        *,
        training_counts: torch.Tensor,
        targets: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss = loss
        self.targets = targets if targets is not None else NextItemTargets()
        self.frequency_terciles = _FrequencyTerciles.from_tensor(training_counts)
        self.register_buffer(
            "_frequency_slice_indices",
            torch.tensor(self.frequency_terciles.slice_indices),
            persistent=False,
        )
        self._epoch_scopes = self._new_scopes()

    @staticmethod
    def _new_scopes() -> dict[str, _TrainingScope]:
        return {name: _TrainingScope() for name in _SCOPES}

    def forward(self, batch: dict) -> dict[str, torch.Tensor | int]:
        out = self.model(batch)
        pairs = self.targets(out)
        if pairs.query_repr.shape[0] == 0:
            zero = (out["query_repr"].sum() + out["item_repr"].sum()) * 0.0
            return {
                "loss": zero,
                "hit_rate": zero.detach(),
                LOSS_DENOMINATOR: 0,
            }

        logits = self.loss.logits(
            pairs.query_repr,
            pairs.positive_repr,
            pairs.positive_ids,
            pairs.group_sizes,
        )
        slice_indices = self._frequency_slice_indices[pairs.positive_ids]
        if self.training:
            self._record(pairs.query_repr, logits, slice_indices)
        return {
            "loss": self.loss.loss_from_logits(logits),
            "hit_rate": (logits.detach().argmax(dim=1) == 0).float().mean(),
            LOSS_DENOMINATOR: pairs.query_repr.shape[0],
            **self._scalar_output(pairs.query_repr, logits, slice_indices),
        }

    def _record(
        self,
        query_repr: torch.Tensor,
        logits: torch.Tensor,
        slice_indices: torch.Tensor,
    ) -> None:
        self._epoch_scopes["global"].update(query_repr, logits)
        for index, name in enumerate(_SLICE_NAMES):
            selected = slice_indices == index
            self._epoch_scopes[name].update(query_repr[selected], logits[selected])

    @staticmethod
    def _scalar_output(
        query_repr: torch.Tensor,
        logits: torch.Tensor,
        slice_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output: dict[str, torch.Tensor] = {}
        selections = {"global": torch.ones_like(slice_indices, dtype=torch.bool)}
        selections.update(
            {name: slice_indices == index for index, name in enumerate(_SLICE_NAMES)}
        )
        for scope_name, selected in selections.items():
            distributions = {
                "query_norm": query_repr[selected].detach().norm(dim=-1),
                "positive_logit": logits[selected, 0].detach(),
                "negative_logit": logits[selected, 1:].detach(),
            }
            for distribution_name, values in distributions.items():
                flattened = values.reshape(-1).float()
                finite = torch.isfinite(flattened)
                count = finite.sum()
                selected_or_zero = torch.where(finite, flattened, 0.0)
                denominator = count.clamp_min(1).to(selected_or_zero.dtype)
                mean = selected_or_zero.sum() / denominator
                variance = (
                    selected_or_zero.square().sum() / denominator - mean.square()
                ).clamp_min(0)
                prefix = f"g3_diagnostics/{scope_name}/{distribution_name}"
                output[f"{prefix}/count"] = count.detach()
                output[f"{prefix}/mean"] = mean.detach()
                output[f"{prefix}/standard_deviation"] = variance.sqrt().detach()
        return output

    def epoch_statistics(self, *, reset: bool = False) -> dict[str, object]:
        statistics = {
            name: scope.statistics() for name, scope in self._epoch_scopes.items()
        }
        if reset:
            self._epoch_scopes = self._new_scopes()
        return statistics


class G3DiagnosticsCallback(Callback):
    """Records per-epoch G3 mechanism evidence in one deterministic document."""

    def __init__(
        self,
        *,
        criterion: G3DiagnosticTwoTowerLoss,
        catalog_encoder: nn.Module,
        components: Mapping[str, nn.Module],
        run_log_directory: Path,
        catalog_chunk_size: int = 4096,
    ) -> None:
        if catalog_chunk_size < 1:
            raise ValueError("catalog chunk size must be positive")
        if any(not name or "/" in name for name in components):
            raise ValueError(
                "component names must be nonempty and cannot contain slashes"
            )
        self.criterion = criterion
        self.catalog_encoder = catalog_encoder
        self.components = dict(components)
        self.catalog_chunk_size = catalog_chunk_size
        self.path = Path(run_log_directory) / _DIAGNOSTICS_FILENAME
        self._gradient_norms = {
            name: _Distribution() for name in sorted(self.components)
        }
        self._catalog_table_parameters = self._find_catalog_table_parameters()
        self._catalog_table_gradient_norms = self._new_catalog_table_gradient_norms()
        self._known_item_slice_indices = torch.tensor(
            self.criterion.frequency_terciles.slice_indices[1:], dtype=torch.long
        )
        self._slice_positions = {
            name: torch.where(self._known_item_slice_indices == index)[0]
            for index, name in enumerate(_SLICE_NAMES)
        }
        self._slice_indices_by_device: dict[torch.device, torch.Tensor] = {}
        self._content_provider = self._find_content_provider(catalog_encoder)
        self._initial_content = self._content_snapshot()
        self._frequency_manifest = self.criterion.frequency_terciles.manifest()
        self._training_count_identity = _integer_sequence_reference(
            self.criterion.frequency_terciles.counts
        )
        self._slice_membership_identity = _integer_sequence_reference(
            self.criterion.frequency_terciles.slice_indices
        )
        self._content_drift_identity = self._content_reference()
        self._epochs = self._load_epochs()

    @staticmethod
    def _find_content_provider(catalog_encoder: nn.Module) -> nn.Module | None:
        providers = [
            module
            for module in catalog_encoder.modules()
            if callable(getattr(module, "content_embeddings", None))
        ]
        if len(providers) > 1:
            raise ValueError("catalog encoder exposes multiple content tables")
        return providers[0] if providers else None

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        for name, module in self.components.items():
            squared_norm: torch.Tensor | None = None
            for parameter in module.parameters():
                gradient = parameter.grad
                if gradient is None:
                    continue
                values = (
                    gradient.coalesce().values() if gradient.is_sparse else gradient
                )
                contribution = values.detach().float().square().sum().double()
                squared_norm = (
                    contribution
                    if squared_norm is None
                    else squared_norm + contribution
                )
            if squared_norm is None:
                reference = next(module.parameters(), None)
                device = torch.device("cpu") if reference is None else reference.device
                squared_norm = torch.zeros((), dtype=torch.float64, device=device)
            self._gradient_norms[name].update(squared_norm.sqrt())
        self._record_catalog_table_gradient_norms()

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        runner = state["train_runner"]
        epoch = int(runner.current_epoch)
        entry = {
            "epoch": epoch,
            "training": self.criterion.epoch_statistics(reset=True),
            "component_gradient_norms": {
                name: distribution.statistics()
                for name, distribution in sorted(self._gradient_norms.items())
            },
            "catalog_table_gradient_norms": {
                parameter_name: {
                    scope: gradient_scope.statistics()
                    for scope, gradient_scope in scoped.items()
                }
                for parameter_name, scoped in sorted(
                    self._catalog_table_gradient_norms.items()
                )
            },
            "catalog_representation_norm": self._catalog_representation_norms(),
            "pretrained_content": self._content_diagnostics(),
        }
        self._gradient_norms = {
            name: _Distribution() for name in sorted(self.components)
        }
        self._catalog_table_gradient_norms = self._new_catalog_table_gradient_norms()
        self._epochs[epoch] = entry
        self._write()

    def _find_catalog_table_parameters(self) -> dict[str, nn.Parameter]:
        expected_rows = self.criterion.frequency_terciles.num_items + 1
        parameters = {}
        for module_name, module in self.catalog_encoder.named_modules():
            if not isinstance(module, nn.Embedding):
                continue
            if module.num_embeddings != expected_rows:
                continue
            name = f"{module_name}.weight" if module_name else "weight"
            parameters[name] = module.weight
        return parameters

    def _new_catalog_table_gradient_norms(
        self,
    ) -> dict[str, dict[str, _RowGradientScope]]:
        return {
            name: {scope: _RowGradientScope() for scope in _SCOPES}
            for name in self._catalog_table_parameters
        }

    def _record_catalog_table_gradient_norms(self) -> None:
        num_items = self.criterion.frequency_terciles.num_items
        for name, parameter in self._catalog_table_parameters.items():
            gradient = parameter.grad
            if gradient is None:
                row_norms = parameter.new_zeros(num_items + 1, dtype=torch.float32)
            elif gradient.is_sparse:
                coalesced = gradient.coalesce()
                row_squared_norms = parameter.new_zeros(
                    num_items + 1, dtype=torch.float32
                )
                row_squared_norms.index_add_(
                    0,
                    coalesced.indices()[0],
                    coalesced.values().detach().float().flatten(1).square().sum(dim=1),
                )
                row_norms = row_squared_norms.sqrt()
            else:
                row_norms = (
                    gradient.detach().float().reshape(num_items + 1, -1).norm(dim=1)
                )
            scoped = self._catalog_table_gradient_norms[name]
            known_row_norms = row_norms[1:]
            scoped["global"].update(known_row_norms)
            slice_indices = self._slice_indices_by_device.get(row_norms.device)
            if slice_indices is None:
                slice_indices = self._known_item_slice_indices.to(row_norms.device)
                self._slice_indices_by_device[row_norms.device] = slice_indices
            for index, slice_name in enumerate(_SLICE_NAMES):
                scoped[slice_name].update(known_row_norms[slice_indices == index])

    def _catalog_representation_norms(self) -> dict[str, object]:
        representations = self._encode_known_items(self.catalog_encoder)
        return self._sliced_distributions(representations.norm(dim=-1))

    def _content_snapshot(self) -> torch.Tensor | None:
        if self._content_provider is None:
            return None
        return self._encode_known_items(
            self._content_provider,
            method_name="content_embeddings",
        )

    def _content_diagnostics(self) -> dict[str, object]:
        if self._content_provider is None or self._initial_content is None:
            return {"available": False}
        current = self._content_snapshot()
        assert current is not None
        drift = (current - self._initial_content).norm(dim=-1)
        cosine = F.cosine_similarity(current, self._initial_content, dim=-1)
        parameters = getattr(self._content_provider, "content_parameters", None)
        trainable = (
            any(parameter.requires_grad for parameter in parameters())
            if callable(parameters)
            else None
        )
        return {
            "available": True,
            "trainable": trainable,
            "drift_l2": self._sliced_distributions(drift),
            "cosine_to_initial": self._sliced_distributions(cosine),
        }

    def _encode_known_items(
        self,
        module: nn.Module,
        *,
        method_name: str | None = None,
    ) -> torch.Tensor:
        device = self._module_device(module)
        method = module if method_name is None else getattr(module, method_name)
        was_training = module.training
        module.eval()
        try:
            with torch.inference_mode():
                chunks = []
                item_ids = torch.arange(
                    1,
                    self.criterion.frequency_terciles.num_items + 1,
                    device=device,
                )
                for chunk_ids in item_ids.split(self.catalog_chunk_size):
                    chunks.append(method(chunk_ids).detach().float().cpu())
        finally:
            module.train(was_training)
        return torch.cat(chunks)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        for tensor in (*module.parameters(), *module.buffers()):
            return tensor.device
        return torch.device("cpu")

    def _sliced_distributions(self, values: torch.Tensor) -> dict[str, object]:
        if values.shape != (self.criterion.frequency_terciles.num_items,):
            raise ValueError(
                "catalog diagnostic values must contain one row per known item"
            )
        output = {}
        global_distribution = _Distribution()
        global_distribution.update(values)
        output["global"] = global_distribution.statistics()
        for name in _SLICE_NAMES:
            distribution = _Distribution()
            distribution.update(values[self._slice_positions[name]])
            output[name] = distribution.statistics()
        return output

    def _load_epochs(self) -> dict[int, dict[str, object]]:
        if not self.path.exists():
            return {}
        document = json.loads(self.path.read_text())
        if document.get("schema_version") != _DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError("unsupported G3 diagnostics schema")
        if document.get("training_count_reference") != self._training_count_reference():
            raise ValueError("saved G3 training-count reference differs from this run")
        if (
            document.get("slice_membership_reference")
            != self._slice_membership_reference()
        ):
            raise ValueError(
                "saved G3 slice membership reference differs from this run"
            )
        if document.get("frequency_terciles") != self._frequency_manifest:
            raise ValueError("saved G3 frequency slices differ from this run")
        if document.get("content_drift_reference") != self._content_drift_identity:
            raise ValueError("saved G3 content drift reference differs from this run")
        return {int(entry["epoch"]): entry for entry in document["epochs"]}

    def _content_reference(self) -> dict[str, object]:
        if self._initial_content is None:
            return {"available": False}
        contiguous = self._initial_content.contiguous()
        return {
            "available": True,
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.numpy().tobytes()).hexdigest(),
        }

    def _training_count_reference(self) -> dict[str, object]:
        return self._training_count_identity

    def _slice_membership_reference(self) -> dict[str, object]:
        return self._slice_membership_identity

    def _write(self) -> None:
        document = {
            "schema_version": _DIAGNOSTICS_SCHEMA_VERSION,
            "frequency_terciles": self._frequency_manifest,
            "training_count_reference": self._training_count_reference(),
            "slice_membership_reference": self._slice_membership_reference(),
            "content_drift_reference": self._content_drift_identity,
            "epochs": [self._epochs[epoch] for epoch in sorted(self._epochs)],
        }
        payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(self.path)


class G3GateDiagnosticsCallback(Callback):
    def __init__(
        self,
        *,
        gate: nn.Module,
        content_provider: nn.Module,
        training_counts: torch.Tensor,
        run_log_directory: Path,
    ) -> None:
        self.gate = gate
        self.content_provider = content_provider
        self.terciles = _FrequencyTerciles.from_tensor(training_counts)
        self.path = Path(run_log_directory) / _GATE_DIAGNOSTICS_FILENAME
        self._gradient_norm = _Distribution()
        self._training_count_reference = _integer_sequence_reference(
            self.terciles.counts
        )
        self._slice_membership_reference = _integer_sequence_reference(
            self.terciles.slice_indices
        )
        self._epochs = self._load_epochs()
        self._frequency_input_parity = self._validate_frequency_input()

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        squared = None
        for parameter in self.gate.parameters():
            if parameter.grad is None:
                continue
            contribution = parameter.grad.detach().float().square().sum().double()
            squared = contribution if squared is None else squared + contribution
        if squared is None:
            reference = next(self.gate.parameters())
            squared = torch.zeros((), dtype=torch.float64, device=reference.device)
        self._gradient_norm.update(squared.sqrt())

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = int(state["train_runner"].current_epoch)
        item_ids = torch.arange(
            1,
            self.terciles.num_items + 1,
            device=G3DiagnosticsCallback._module_device(self.gate),
        )
        was_training = self.gate.training
        self.gate.eval()
        try:
            with torch.inference_mode():
                gate_values = self.gate(item_ids).reshape(-1).float().cpu()
                content = self.content_provider.lookup(item_ids).float().cpu()
        finally:
            self.gate.train(was_training)
        if gate_values.shape != (self.terciles.num_items,):
            raise ValueError("content gate must emit one scalar per known item")
        raw_norm = content.norm(dim=-1)
        gated_norm = (content * gate_values.unsqueeze(-1)).norm(dim=-1)
        counts = torch.tensor(self.terciles.counts[1:], dtype=torch.float32)
        logged = torch.log1p(counts)
        entry = {
            "epoch": epoch,
            "gate_parameter_gradient_norm": self._gradient_norm.statistics(),
            "gate_output": self._sliced_gate(gate_values),
            "raw_content_norm": self._sliced_distribution(raw_norm),
            "gated_content_norm": self._sliced_distribution(gated_norm),
            "content_scaling_ratio": self._sliced_distribution(
                gated_norm / raw_norm.clamp_min(torch.finfo(raw_norm.dtype).tiny)
            ),
            "gate_log1p_count_pearson": _pearson(gate_values, logged),
        }
        self._gradient_norm = _Distribution()
        self._epochs[epoch] = entry
        self._write()

    def _validate_frequency_input(self) -> bool | None:
        standardized = getattr(self.gate, "standardized_log_counts", None)
        if standardized is None:
            return None
        counts = torch.tensor(self.terciles.counts[1:], dtype=torch.float32)
        logged = torch.log1p(counts)
        deviation = logged.std(unbiased=False)
        expected = logged - logged.mean()
        if deviation != 0:
            expected = expected / deviation
        expected = torch.cat([expected.new_zeros(1), expected])
        if standardized.shape != expected.shape or not torch.equal(
            standardized.detach().float().cpu(), expected
        ):
            raise ValueError("frequency gate standardized counts differ from source")
        return True

    def _sliced_gate(self, values: torch.Tensor) -> dict[str, object]:
        output = {}
        for name, selected in self._selections().items():
            current = values[selected]
            output[name] = {
                **_distribution(current),
                "p05": float(torch.quantile(current, 0.05)),
                "p50": float(torch.quantile(current, 0.50)),
                "p95": float(torch.quantile(current, 0.95)),
                "fraction_at_or_below_0_05": float((current <= 0.05).float().mean()),
                "fraction_at_or_above_0_95": float((current >= 0.95).float().mean()),
            }
        return output

    def _sliced_distribution(self, values: torch.Tensor) -> dict[str, object]:
        return {
            name: _distribution(values[selected])
            for name, selected in self._selections().items()
        }

    def _selections(self) -> dict[str, torch.Tensor]:
        memberships = torch.tensor(self.terciles.slice_indices[1:])
        return {
            "global": torch.ones(self.terciles.num_items, dtype=torch.bool),
            **{
                name: memberships == index
                for index, name in enumerate(_SLICE_NAMES)
            },
        }

    def _load_epochs(self) -> dict[int, dict[str, object]]:
        if not self.path.exists():
            return {}
        document = json.loads(self.path.read_text())
        if (
            document.get("schema_version") != 1
            or document.get("training_count_reference")
            != self._training_count_reference
            or document.get("slice_membership_reference")
            != self._slice_membership_reference
        ):
            raise ValueError("saved gate diagnostics identity changed")
        return {int(entry["epoch"]): entry for entry in document["epochs"]}

    def _write(self) -> None:
        document = {
            "schema_version": 1,
            "frequency_terciles": self.terciles.manifest(),
            "training_count_reference": self._training_count_reference,
            "slice_membership_reference": self._slice_membership_reference,
            "frequency_input_parity": self._frequency_input_parity,
            "epochs": [self._epochs[epoch] for epoch in sorted(self._epochs)],
        }
        payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(self.path)


def _distribution(values: torch.Tensor) -> dict[str, object]:
    distribution = _Distribution()
    distribution.update(values)
    return distribution.statistics()


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if denominator == 0:
        return None
    return float((left_centered * right_centered).sum() / denominator)


def _integer_sequence_reference(values: tuple[int, ...]) -> dict[str, object]:
    payload = json.dumps(values, separators=(",", ":")).encode()
    return {
        "length": len(values),
        "encoding": "canonical-json-integers",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
