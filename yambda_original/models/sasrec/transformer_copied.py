import math

import torch
from flash_attn import flash_attn_varlen_func
from torch import nn
from torch.nn import functional as F


class ConcatenatedEmbedding(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        sparse: bool,
        pretrained_embeddings: torch.Tensor | None,
        has_special_tokens: bool,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            sparse=sparse,
        )
        self.has_special_tokens = has_special_tokens

        if pretrained_embeddings is not None:
            self.pretrained = nn.Buffer(pretrained_embeddings)
        else:
            self.pretrained = None

    @property
    def num_real_items(self) -> int:
        return self.embedding.num_embeddings - (1 if self.has_special_tokens else 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        id_embeddings = self.embedding(input)

        if self.pretrained is None:
            return id_embeddings

        return torch.cat([id_embeddings, self.pretrained[input]], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_dim, bias=True)
        self.w2 = nn.Linear(dim, intermediate_dim, bias=True)
        self.w3 = nn.Linear(intermediate_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

    def init_weights(self, base_std: float, res_std: float):
        """Initialize weights for SwiGLU FFN."""
        for layer in [self.w1, self.w2]:
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=base_std,
                a=-2 * base_std,
                b=2 * base_std,
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=res_std,
            a=-2 * res_std,
            b=2 * res_std,
        )
        if self.w3.bias is not None:
            nn.init.zeros_(self.w3.bias)


class RegularMLP(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.activation(self.linear1(x)))

    def init_weights(self, base_std: float, res_std: float):
        nn.init.trunc_normal_(
            self.linear1.weight,
            mean=0.0,
            std=base_std,
            a=-2 * base_std,
            b=2 * base_std,
        )
        nn.init.zeros_(self.linear1.bias)
        nn.init.trunc_normal_(
            self.linear2.weight,
            mean=0.0,
            std=res_std,
            a=-2 * res_std,
            b=2 * res_std,
        )
        nn.init.zeros_(self.linear2.bias)


# needed to reduce memory usage
class BosTokensIdsBuilder(nn.Module):
    def __init__(self, bos_id: int):
        super().__init__()
        self.bos_id = bos_id

    def forward(
        self,
        item_ids: torch.Tensor,
        cumulative_lengths: torch.Tensor,
    ) -> torch.Tensor:
        device = item_ids.device

        full_ids = torch.empty(
            cumulative_lengths[-1].item(), device=device, dtype=torch.long
        )
        bos_idx = cumulative_lengths[:-1].long()

        full_ids[bos_idx] = self.bos_id

        real_items_mask = torch.ones(
            full_ids.shape[0], device=device, dtype=torch.bool
        )
        real_items_mask[bos_idx] = False
        item_target_indices = torch.where(real_items_mask)[0]
        full_ids[item_target_indices] = item_ids

        return full_ids


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        num_kv_heads: int,
        dropout: float,
        use_swiglu: bool = True,
        use_alibi: bool = True,
        # rope: nn.Module,
    ) -> None:
        super().__init__()
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // nhead
        self.dim = dim
        self.use_alibi = use_alibi

        # self.rope = rope
        self.ln_1 = nn.RMSNorm(dim)

        self.q_proj = nn.Linear(dim, nhead * self.head_dim)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim)

        if use_alibi:
            self.alibi_slopes = nn.Buffer(self._get_alibi_slopes(n_heads=nhead))
        else:
            self.alibi_slopes = None

        self.out_proj = nn.Linear(nhead * self.head_dim, dim)

        self.ln_2 = nn.RMSNorm(dim)

        allign_to = 64
        hidden_dim = (4 * dim + allign_to - 1) // allign_to * allign_to

        if use_swiglu:
            self.ffn = SwiGLU(dim, hidden_dim)
        else:
            self.ffn = RegularMLP(dim, hidden_dim)

        self.dropout = nn.Dropout(dropout, inplace=True)

    @staticmethod
    def _get_alibi_slopes(n_heads: int):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        return torch.tensor(get_slopes_power_of_2(n_heads), dtype=torch.float32)

    def forward(
        self, x: torch.Tensor, cumulative_lens: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        residual = x
        x = self.ln_1(x)

        q = self.q_proj(x).view(-1, self.nhead, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        flash_attn_kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cumulative_lens,
            "cu_seqlens_k": cumulative_lens,
            "max_seqlen_q": max_length,
            "max_seqlen_k": max_length,
            "dropout_p": self.dropout.p if self.training else 0.0,
            "causal": True,
        }

        if self.use_alibi and self.alibi_slopes is not None:
            flash_attn_kwargs["alibi_slopes"] = self.alibi_slopes

        attn_out = flash_attn_varlen_func(**flash_attn_kwargs)

        x = attn_out.reshape(-1, self.nhead * self.head_dim)
        x = residual + self.dropout(self.out_proj(x))

        x = x + self.dropout(self.ffn(self.ln_2(x)))
        return x

    def init_weights(self, initializer_range: float = 0.02):
        for layer in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.trunc_normal_(
                layer.weight,
                mean=0.0,
                std=initializer_range,
                a=-2 * initializer_range,
                b=2 * initializer_range,
            )
            nn.init.zeros_(layer.bias)

        self.ffn.init_weights(initializer_range, initializer_range)


class TransformerUserEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        nhead: int,
        dropout: float,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int = 256,
        pretrained_embeddings: torch.Tensor | None = None,
        sparse: bool = False,
        use_swiglu: bool = False,
        use_bos_tokens: bool = True,
        use_alibi: bool = True,
        use_positional_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.item_embeddings = ConcatenatedEmbedding(
            num_items=num_items + 1 if use_bos_tokens else num_items,
            embedding_dim=embedding_dim,
            sparse=sparse,
            pretrained_embeddings=pretrained_embeddings,
            has_special_tokens=use_bos_tokens,
        )

        self.use_bos_tokens = use_bos_tokens
        self.use_alibi = use_alibi
        self.use_positional_embedding = use_positional_embedding

        if use_bos_tokens:
            self.bos_id = num_items
            self.bos_id_builder = BosTokensIdsBuilder(self.bos_id)
        else:
            self.bos_id = None
            self.bos_id_builder = None

        total_embedding_dim = embedding_dim + (
            pretrained_embeddings.size(1)
            if pretrained_embeddings is not None
            else 0
        )

        self.max_len = max_seq_len + (1 if use_bos_tokens else 0)
        max_pos_len = self.max_len
        if use_positional_embedding:
            self.position_embeddings = nn.Embedding(
                num_embeddings=max_pos_len, embedding_dim=total_embedding_dim
            )
            self.layernorm = nn.LayerNorm(total_embedding_dim, eps=1e-9)
            self.dropout = nn.Dropout(dropout)
        else:
            self.position_embeddings = None
            self.layernorm = None
            self.dropout = None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    dim=total_embedding_dim,
                    nhead=nhead,
                    num_kv_heads=num_kv_heads,
                    dropout=dropout,
                    use_swiglu=use_swiglu,
                    use_alibi=use_alibi,
                    # rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(total_embedding_dim)

    def init_weights(self):
        nn.init.trunc_normal_(self.item_embeddings.embedding.weight, std=0.02)
        if self.position_embeddings is not None:
            nn.init.trunc_normal_(self.position_embeddings.weight, std=0.02)
        for layer in self.layers:
            layer.init_weights()

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled()
            else torch.float32
        )
        item_ids = inputs["item.ids"]
        original_lengths = inputs["item.length"]
        device = item_ids.device

        lengths = original_lengths + (1 if self.use_bos_tokens else 0)
        cumulative_length = torch.zeros(
            lengths.shape[0] + 1, device=device, dtype=torch.int32
        )
        cumulative_length[1:] = torch.cumsum(lengths, dim=0)

        if self.use_bos_tokens:
            full_ids = self.bos_id_builder(item_ids, cumulative_length)
        else:
            full_ids = item_ids
        x = self.item_embeddings(full_ids).to(dtype=dtype)

        if self.use_positional_embedding:
            positions = torch.arange(x.shape[0], device=device)
            seq_indices = torch.repeat_interleave(
                torch.arange(lengths.shape[0], device=device), lengths
            )
            cumulative_ends = cumulative_length[1:]
            positions = cumulative_ends[seq_indices] - 1 - positions

            x = x + self.position_embeddings(positions)
            x = self.dropout(self.layernorm(x))

        for layer in self.layers:
            x = layer(x, cumulative_length, self.max_len)
        x = self.final_norm(x)

        if self.use_bos_tokens:
            real_items_mask = torch.ones(
                x.shape[0], device=device, dtype=torch.bool
            )
            bos_indices = cumulative_length[:-1].long()
            real_items_mask[bos_indices] = False
            return x[real_items_mask]
        else:
            return x
