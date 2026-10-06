"""Small trainable assistance heads; π0.5 is never part of these modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FeatureLayout:
    semantic_dim: int
    action_dim: int
    low_dim: int

    @property
    def input_dim(self) -> int:
        return self.semantic_dim + self.action_dim + self.low_dim


class _ProjectInputs(nn.Module):
    def __init__(self, layout: FeatureLayout) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.Linear(layout.semantic_dim, 128), nn.GELU())
        self.action = nn.Sequential(nn.Linear(layout.action_dim, 128), nn.GELU())
        self.low = nn.Sequential(nn.Linear(layout.low_dim, 64), nn.GELU())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        a = z[..., : self.semantic[0].in_features]
        b0 = a.shape[-1]
        b = z[..., b0 : b0 + self.action[0].in_features]
        c = z[..., b0 + self.action[0].in_features :]
        return torch.cat([self.semantic(a), self.action(b), self.low(c)], dim=-1)


class AssistMLP(nn.Module):
    def __init__(
        self,
        layout: FeatureLayout,
        *,
        hidden_size: int = 256,
        horizons: int = 4,
        complexity_classes: int = 5,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.project = _ProjectInputs(layout)
        self.body = nn.Sequential(
            nn.LayerNorm(320),
            nn.Linear(320, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.help_head = nn.Linear(hidden_size, horizons)
        self.tti_head = nn.Linear(hidden_size, 1)
        self.complexity_head = nn.Linear(hidden_size, complexity_classes)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.body(self.project(z))
        return {
            "help_logits": self.help_head(h),
            "tti": self.tti_head(h).squeeze(-1),
            "complexity_logits": self.complexity_head(h),
        }


class AssistGRU(nn.Module):
    def __init__(
        self,
        layout: FeatureLayout,
        *,
        hidden_size: int = 256,
        horizons: int = 4,
        complexity_classes: int = 5,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.project = _ProjectInputs(layout)
        self.gru = nn.GRU(320, hidden_size, num_layers=1, batch_first=True)
        self.help_head = nn.Linear(hidden_size, horizons)
        self.tti_head = nn.Linear(hidden_size, 1)
        self.complexity_head = nn.Linear(hidden_size, complexity_classes)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.gru(self.project(z))
        h = h[:, -1]
        return {
            "help_logits": self.help_head(h),
            "tti": self.tti_head(h).squeeze(-1),
            "complexity_logits": self.complexity_head(h),
        }


def action_statistics(action_chunk):
    """Return auxiliary action statistics used after hidden features."""

    if isinstance(action_chunk, torch.Tensor):
        mean = action_chunk.mean(dim=-2)
        std = action_chunk.std(dim=-2, unbiased=False)
        norm = action_chunk.reshape(*action_chunk.shape[:-2], -1).norm(dim=-1, keepdim=True)
        first = action_chunk[..., 0, :]
        last = action_chunk[..., -1, :]
        delta = last - first
        return torch.cat([mean, std, norm, first, last, delta], dim=-1)
    import numpy as np

    action_chunk = np.asarray(action_chunk)
    mean = action_chunk.mean(axis=-2)
    std = action_chunk.std(axis=-2)
    norm = np.linalg.norm(action_chunk.reshape(*action_chunk.shape[:-2], -1), axis=-1, keepdims=True)
    first = action_chunk[..., 0, :]
    last = action_chunk[..., -1, :]
    delta = last - first
    return np.concatenate([mean, std, norm, first, last, delta], axis=-1)

