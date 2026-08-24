from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from configs.model_config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, cfg: ModelConfig, dim: int | None = None):
        super().__init__()
        self.dim = dim if dim is not None else cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)
