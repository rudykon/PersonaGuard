"""CBraMod backbone and parameter-efficient residual adapter.

The backbone implementation follows the official MIT-licensed CBraMod source:
https://github.com/wjq-learning/CBraMod

Copyright (c) 2025 Jiquan Wang. The adaptation keeps the original state-dict
layout so the public ICLR 2025 checkpoint can be loaded strictly. MER-PS
specific pooling, context conditioning, freezing, and residual prediction are
new code in this project.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CrissCrossEncoderLayer(nn.Module):
    """Spatial and temporal self-attention on separate feature halves."""

    def __init__(
        self,
        d_model: int = 200,
        nhead: int = 8,
        dim_feedforward: int = 800,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ):
        super().__init__()
        if d_model % 2 or nhead % 2:
            raise ValueError("CBraMod requires even d_model and nhead")
        self.self_attn_s = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_t = nn.MultiheadAttention(
            d_model // 2,
            nhead // 2,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = True
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def _attention_block(self, x: Tensor) -> Tensor:
        batch, channels, patches, features = x.shape
        spatial = x[..., : features // 2]
        temporal = x[..., features // 2 :]
        spatial = spatial.transpose(1, 2).contiguous().view(
            batch * patches, channels, features // 2
        )
        temporal = temporal.contiguous().view(
            batch * channels, patches, features // 2
        )
        spatial = self.self_attn_s(
            spatial, spatial, spatial, need_weights=False
        )[0]
        temporal = self.self_attn_t(
            temporal, temporal, temporal, need_weights=False
        )[0]
        spatial = spatial.view(
            batch, patches, channels, features // 2
        ).transpose(1, 2)
        temporal = temporal.view(
            batch, channels, patches, features // 2
        )
        return self.dropout1(torch.cat([spatial, temporal], dim=-1))

    def _feedforward_block(self, x: Tensor) -> Tensor:
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(x))))
        )

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        del src_mask
        x = src
        x = x + self._attention_block(self.norm1(x))
        x = x + self._feedforward_block(self.norm2(x))
        return x


class CrissCrossEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = int(num_layers)
        self.norm = None

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int = 200, d_model: int = 200):
        super().__init__()
        self.d_model = int(d_model)
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            )
        )
        self.mask_encoding = nn.Parameter(
            torch.zeros(in_dim), requires_grad=False
        )
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                25, 25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch, channels, patches, patch_size = x.shape
        if patch_size != 200:
            raise ValueError(
                f"CBraMod expects 200 points per patch, got {patch_size}"
            )
        masked = x if mask is None else x.clone()
        if mask is not None:
            masked[mask == 1] = self.mask_encoding
        flattened = masked.contiguous().view(
            batch, 1, channels * patches, patch_size
        )
        patch_embedding = self.proj_in(flattened)
        patch_embedding = patch_embedding.permute(
            0, 2, 1, 3
        ).contiguous().view(batch, channels, patches, self.d_model)

        spectrum = torch.fft.rfft(
            flattened.view(batch * channels * patches, patch_size),
            dim=-1,
            norm="forward",
        )
        spectrum = torch.abs(spectrum).view(
            batch, channels, patches, 101
        )
        patch_embedding = patch_embedding + self.spectral_proj(spectrum)
        positional = self.positional_encoding(
            patch_embedding.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return patch_embedding + positional


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class CBraModBackbone(nn.Module):
    """Official CBraMod-base architecture with compatible parameter names."""

    def __init__(
        self,
        in_dim: int = 200,
        out_dim: int = 200,
        d_model: int = 200,
        dim_feedforward: int = 800,
        seq_len: int = 30,
        n_layer: int = 12,
        nhead: int = 8,
    ):
        super().__init__()
        del seq_len
        self.patch_embedding = PatchEmbedding(in_dim, d_model)
        layer = CrissCrossEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation=F.gelu,
        )
        self.encoder = CrissCrossEncoder(layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.proj_out(
            self.encoder(self.patch_embedding(x, mask))
        )


def load_pretrained_cbramod(
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CBraModBackbone:
    model = CBraModBackbone()
    state = torch.load(
        Path(checkpoint), map_location=map_location, weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model


def pool_cbramod_features(features: Tensor) -> Tensor:
    """Pool channel-patch tokens without discarding extrema."""

    if features.ndim != 4:
        raise ValueError(
            f"Expected [batch, channel, patch, feature], got {features.shape}"
        )
    return torch.cat(
        [
            features.mean(dim=(1, 2)),
            features.amax(dim=(1, 2)),
        ],
        dim=-1,
    )


def cbramod_patch_tokens(features: Tensor) -> Tensor:
    """Average channels while retaining the four one-second patch tokens."""

    if features.ndim != 4:
        raise ValueError(
            f"Expected [batch, channel, patch, feature], got {features.shape}"
        )
    return features.mean(dim=1)


class ResidualAdapter(nn.Module):
    """Zero-initialized bottleneck adapter for frozen embeddings."""

    def __init__(
        self, feature_dim: int, bottleneck: int = 64, dropout: float = 0.2
    ):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck)
        self.up = nn.Linear(bottleneck, feature_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: Tensor) -> Tensor:
        update = self.up(
            self.dropout(F.gelu(self.down(self.norm(x))))
        )
        return x + update


class CBraModResidualHead(nn.Module):
    """Frozen CBraMod plus a small adapter conditioned on context."""

    def __init__(
        self,
        backbone: CBraModBackbone,
        *,
        context_dim: int = 7,
        adapter_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        max_correction: float = 64.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = bool(freeze_backbone)
        self.max_correction = float(max_correction)
        self.adapter = ResidualAdapter(400, adapter_dim, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(400 + context_dim),
            nn.Linear(400 + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode(self, eeg: Tensor) -> Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(eeg)
        else:
            features = self.backbone(eeg)
        return pool_cbramod_features(features)

    def forward_embedding(
        self, embedding: Tensor, context: Tensor
    ) -> Tensor:
        adapted = self.adapter(embedding)
        return self.max_correction * torch.tanh(
            self.head(torch.cat([adapted, context], dim=-1))
        )

    def forward(self, eeg: Tensor, context: Tensor) -> Tensor:
        return self.forward_embedding(self.encode(eeg), context)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
