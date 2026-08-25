from __future__ import annotations

import torch
from torch import nn

from configs.model_config import ModelConfig
from model.rms_norm import RMSNorm


class AttentionResidual(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.norm = RMSNorm(cfg)

    def forward(
        self, blocks: list[torch.Tensor], partial_block: torch.Tensor | None
    ) -> torch.Tensor:
        sources = blocks if partial_block is None else [*blocks, partial_block]
        v = torch.stack(sources)  # [S,B,T,D]
        k = self.norm(v)  # [S,B,T,D]
        logits = torch.einsum("d,sbtd->sbt", self.pseudo_query, k)
        alpha = logits.softmax(dim=0)
        return torch.einsum("sbt,sbtd->btd", alpha, v)
