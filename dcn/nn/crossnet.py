from dataclasses import dataclass

import torch
from torch import nn

from .types import ActivationFactory, ModuleWithDim, NormalizationFactory


@dataclass
class CrossHeadDescription:
    num_layers: int
    compression_activation: ActivationFactory
    decompression_activation: ActivationFactory | None = None
    compressed_norm: NormalizationFactory | None = None
    decompressed_norm: NormalizationFactory | None = None


class MultiHeadLayer(ModuleWithDim):
    def __init__(
        self,
        input_dim: int,
        width: int,
        num_heads: int,
        output_dim: int,
        compression_norm_factories: NormalizationFactory
        | list[NormalizationFactory | None]
        | None,
        compression_activations: list[nn.Module],
        decompression_norm_factories: NormalizationFactory
        | list[NormalizationFactory | None]
        | None,
        decompression_activations: list[nn.Module] | None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.width = width
        self.output_dim = output_dim

        self.compress, self.c_heads = self._build_compression_layers(
            input_dim,
            width,
            num_heads,
            compression_norm_factories,
            compression_activations,
        )

        self.decompress, self.d_heads = self._build_decompression_layers(
            width,
            output_dim,
            num_heads,
            decompression_norm_factories,
            decompression_activations,
        )

    def _build_compression_layers(
        self,
        input_dim: int,
        width: int,
        num_heads: int,
        norm_factories: NormalizationFactory | list[NormalizationFactory | None] | None,
        activations: list[nn.Module],
    ) -> tuple[nn.Module, nn.ModuleList]:
        if not isinstance(norm_factories, list):
            compress = nn.Sequential(
                nn.Linear(input_dim * num_heads, width * num_heads),
                norm_factories(width * num_heads) if norm_factories else nn.Identity(),
            )
            heads = nn.ModuleList(activations)
        else:
            compress = nn.Linear(input_dim * num_heads, width * num_heads)
            heads = nn.ModuleList(
                [
                    nn.Sequential(
                        norm_factories[i](width)
                        if norm_factories[i]
                        else nn.Identity(),
                        activations[i],
                    )
                    for i in range(num_heads)
                ]
            )
        return compress, heads

    def _build_decompression_layers(
        self,
        width: int,
        output_dim: int,
        num_heads: int,
        norm_factories: NormalizationFactory | list[NormalizationFactory | None] | None,
        activations: list[nn.Module] | None,
    ) -> tuple[nn.Module, nn.ModuleList]:
        if not isinstance(norm_factories, list):
            decompress = nn.Sequential(
                nn.Linear(width * num_heads, num_heads * output_dim),
                norm_factories(num_heads * output_dim)
                if norm_factories
                else nn.Identity(),
            )
            heads = nn.ModuleList(
                activations
                if activations
                else [nn.Identity() for _ in range(num_heads)]
            )
        else:
            decompress = nn.Linear(width * num_heads, num_heads * output_dim)
            heads = nn.ModuleList(
                [
                    nn.Sequential(
                        norm_factories[i](output_dim)
                        if norm_factories[i]
                        else nn.Identity(),
                        activations[i] if activations else nn.Identity(),
                    )
                    for i in range(num_heads)
                ]
            )
        return decompress, heads

    @property
    def out_dim(self) -> int:
        return self.output_dim * self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        chunks = torch.split(x, self.width, dim=-1)
        x = torch.cat(
            [head(chunk) for head, chunk in zip(self.c_heads, chunks)], dim=-1
        )

        x = self.decompress(x)
        chunks = torch.split(x, self.output_dim, dim=-1)
        x = torch.cat(
            [head(chunk) for head, chunk in zip(self.d_heads, chunks)], dim=-1
        )
        return x


class CrossNetwork(ModuleWithDim):
    def __init__(
        self,
        input_dim: int,
        width: int,
        heads_descriptions: list[CrossHeadDescription],
        global_compressed_norm: NormalizationFactory | None = None,
        global_decompressed_norm: NormalizationFactory | None = None,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._heads_descriptions = sorted(
            heads_descriptions, key=lambda h: h.num_layers, reverse=True
        )
        self._max_layers = self._heads_descriptions[0].num_layers
        self._layers = nn.ModuleList()

        for layer_idx in range(self._max_layers):
            active_heads = [
                h for h in self._heads_descriptions if h.num_layers > layer_idx
            ]
            num_active = len(active_heads)

            if global_compressed_norm:
                c_norms = global_compressed_norm
            else:
                head_norms = [h.compressed_norm for h in active_heads]
                c_norms = head_norms if any(h is not None for h in head_norms) else None

            if global_decompressed_norm:
                d_norms = global_decompressed_norm
            else:
                head_norms = [h.decompressed_norm for h in active_heads]
                d_norms = head_norms if any(h is not None for h in head_norms) else None

            d_acts = [
                h.decompression_activation()
                for h in active_heads
                if h.decompression_activation
            ]

            self._layers.append(
                MultiHeadLayer(
                    input_dim=input_dim,
                    width=width,
                    num_heads=num_active,
                    output_dim=input_dim,
                    compression_norm_factories=c_norms,
                    compression_activations=[
                        h.compression_activation() for h in active_heads
                    ],
                    decompression_norm_factories=d_norms,
                    decompression_activations=d_acts
                    if len(d_acts) == num_active
                    else None,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_heads = len(self._heads_descriptions)
        current_state = x.repeat(1, num_heads)
        final_head_outputs = {}

        for layer_idx, layer in enumerate(self._layers):
            layer_output = layer(current_state)
            chunks = torch.split(layer_output, self._input_dim, dim=-1)

            next_step_chunks = []
            for head_idx in range(layer.num_heads):
                if self._heads_descriptions[head_idx].num_layers == layer_idx + 1:
                    final_head_outputs[head_idx] = chunks[head_idx]
                else:
                    next_step_chunks.append(chunks[head_idx])

            if next_step_chunks:
                current_state = torch.cat(next_step_chunks, dim=-1)
            else:
                break

        assert len(final_head_outputs) == num_heads, (
            f"Expected {num_heads} final outputs, got {len(final_head_outputs)}"
        )

        return torch.cat([final_head_outputs[i] for i in range(num_heads)], dim=-1)

    @property
    def out_dim(self) -> int:
        return self._input_dim * len(self._heads_descriptions)
