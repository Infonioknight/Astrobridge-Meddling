"""Minimal, faithful vendored port of AstroCLIP's SpecFormer architecture.

Verified 2026-07-18 against PolymathicAI/AstroCLIP (MIT license), commit
``e129576a16bccd25a2794be21fab34d05c608661``:

- ``astroclip/models/specformer.py`` -- ``SpecFormer`` class: constructor
  hyperparameters, ``forward``, ``forward_without_preprocessing``,
  ``preprocess``, ``_slice``, ``_reset_parameters_datapt``.
- ``astroclip/modules.py`` -- ``LayerNorm``, ``SelfAttention``, ``MLP``,
  ``TransformerBlock``, ``_init_by_depth``.
- ``downstream_tasks/property_estimation/embed_provabgs.py`` -- the
  reference loading/embedding recipe:
  ``checkpoint = torch.load(path); specformer = SpecFormer(**checkpoint["hyper_parameters"]);
  specformer.load_state_dict(checkpoint["state_dict"]); ...
  np.mean(specformer(x)["embedding"].cpu().numpy(), axis=1)``.
- ``configs/specformer.yaml`` -- the hyperparameters used to pretrain the
  checkpoint hosted at https://huggingface.co/polymathic-ai/specformer
  (revision ``160d67f0c07daf33d192568ca60ff38d76c39d66``): ``input_dim=22,
  embed_dim=768, num_layers=6, num_heads=6, max_len=800, dropout=0``. This
  module does not hardcode those values; the caller reads them from
  ``checkpoint["hyper_parameters"]`` at load time.

Deliberate frozen-inference deviations from upstream:

- The original class subclasses ``lightning.LightningModule``. This port
  subclasses plain ``torch.nn.Module`` while preserving all state-dict keys.
  Lightning is needed only to deserialize the checkpoint's AttributeDict.
- Masked-pretraining and trainer methods are omitted because this module is
  used only for frozen evaluation.
"""

from __future__ import annotations

import math
import numbers
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.Module):
    """Layer norm with optional bias, matching AstroCLIP's implementation."""

    def __init__(
        self,
        shape: Union[int, Tuple[int, ...], torch.Size],
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (shape,) if isinstance(shape, numbers.Integral) else tuple(shape)
        self.weight = nn.Parameter(torch.empty(shape))
        self.bias = nn.Parameter(torch.empty(shape)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class MLP(nn.Module):
    """Two-layer MLP matching AstroCLIP's implementation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.encoder = nn.Linear(in_features, hidden_features, bias=bias)
        self.decoder = nn.Linear(hidden_features, in_features, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return x


class SelfAttention(nn.Module):
    """Non-causal multi-head self-attention used by SpecFormer."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim should be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        self.projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c = x.shape
        q, k, v = self.attention(x).split(self.embedding_dim, dim=2)
        nh = self.num_heads
        hs = c // nh
        k = k.view(b, t, nh, hs).transpose(1, 2)
        q = q.view(b, t, nh, hs).transpose(1, 2)
        v = v.view(b, t, nh, hs).transpose(1, 2)
        dropout_p = self.attention_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.residual_dropout(self.projection(y))


class TransformerBlock(nn.Module):
    """Pre-normalization transformer block."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = True,
        mlp_expansion: int = 4,
    ) -> None:
        super().__init__()
        self.layernorm1 = LayerNorm(embedding_dim, bias=bias)
        self.attention = SelfAttention(embedding_dim, num_heads, dropout=dropout, bias=bias)
        self.layernorm2 = LayerNorm(embedding_dim, bias=bias)
        self.mlp = MLP(embedding_dim, mlp_expansion * embedding_dim, dropout=dropout, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x


def _init_by_depth(module: nn.Module, depth: float) -> None:
    """Initialize transformer weights as in AstroCLIP."""
    if isinstance(module, nn.Linear):
        fan_in = module.weight.size(-1)
        std = 1 / math.sqrt(2 * fan_in * depth)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SpecFormer(nn.Module):
    """Frozen-inference port of ``astroclip.models.specformer.SpecFormer``."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        max_len: int,
        mask_num_chunks: int = 6,
        mask_chunk_width: int = 50,
        slice_section_length: int = 20,
        slice_overlap: int = 10,
        dropout: float = 0.1,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.mask_num_chunks = mask_num_chunks
        self.mask_chunk_width = mask_chunk_width
        self.slice_section_length = slice_section_length
        self.slice_overlap = slice_overlap

        self.data_embed = nn.Linear(input_dim, embed_dim)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = LayerNorm(embed_dim, bias=True)
        self.head = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters_datapt()

    def forward(self, x: Tensor) -> dict:
        x = self.preprocess(x)
        return self.forward_without_preprocessing(x)

    def forward_without_preprocessing(self, x: Tensor) -> dict:
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(f"Cannot forward sequence of length {t}, block size is only {self.max_len}")
        pos = torch.arange(0, t, dtype=torch.long, device=x.device)

        data_emb = self.data_embed(x)
        pos_emb = self.position_embed(pos)
        x = self.dropout(data_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.final_layernorm(x)
        reconstructions = self.head(x)
        return {"reconstructions": reconstructions, "embedding": x}

    def preprocess(self, x: Tensor) -> Tensor:
        std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
        x = (x - mean) / std
        x = self._slice(x)
        x = F.pad(x, pad=(2, 0, 1, 0), mode="constant", value=0)
        x[:, 0, 0] = (mean.squeeze() - 2) / 2
        x[:, 0, 1] = (std.squeeze() - 2) / 8
        return x

    def _slice(self, x: Tensor) -> Tensor:
        start_indices = np.arange(
            0,
            x.shape[1] - self.slice_overlap,
            self.slice_section_length - self.slice_overlap,
        )
        sections = [
            x[:, start : start + self.slice_section_length].transpose(1, 2)
            for start in start_indices
        ]
        if sections[-1].shape[1] < self.slice_section_length:
            sections.pop(-1)
        return torch.cat(sections, 1)

    def _reset_parameters_datapt(self) -> None:
        for emb in [self.data_embed, self.position_embed]:
            std = 1 / math.sqrt(self.embed_dim)
            nn.init.trunc_normal_(emb.weight, std=std, a=-3 * std, b=3 * std)
        self.blocks.apply(lambda module: _init_by_depth(module, self.num_layers))
        self.head.apply(lambda module: _init_by_depth(module, 1 / 2))
