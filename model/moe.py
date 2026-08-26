from __future__ import annotations

import math

import torch
from torch import nn

from configs.model_config import ModelConfig
from model.rms_norm import RMSNorm


def softcap(x: torch.Tensor, beta: float) -> torch.Tensor:
    return beta * torch.tanh(x / beta)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.num_shared_experts = cfg.num_shared_experts
        self.topk = cfg.num_experts_per_token
        self.moe_latent_dim = cfg.moe_latent_dim
        self.intermediate_size = cfg.moe_intermediate_size
        self.situ_beta_gate = cfg.situ_beta_gate
        self.situ_beta_up = cfg.situ_beta_up
        self.capacity_factor = cfg.moe_capacity_factor
        self.norm = RMSNorm(cfg, dim=cfg.moe_latent_dim)

        self.down_proj = nn.Linear(self.hidden_size, self.moe_latent_dim, bias=False)
        self.up_proj = nn.Linear(self.moe_latent_dim, self.hidden_size, bias=False)

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.w13 = nn.Parameter(
            torch.empty(
                self.num_experts, self.moe_latent_dim, 2 * self.intermediate_size
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.moe_latent_dim)
        )

        self.w13_shared = nn.Parameter(
            torch.empty(
                self.num_shared_experts, self.hidden_size, 2 * self.intermediate_size
            )
        )
        self.w2_shared = nn.Parameter(
            torch.empty(
                self.num_shared_experts, self.intermediate_size, self.hidden_size
            )
        )

        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

        self._qb_margin_chunks: list[torch.Tensor] = []

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.xavier_uniform_(self.w13)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w13_shared)
        nn.init.xavier_uniform_(self.w2_shared)

    def situ_glu(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return (
            softcap(gate, self.situ_beta_gate)
            * torch.sigmoid(gate)
            * softcap(up, self.situ_beta_up)
        )

    def route(self, x_flat: torch.Tensor):
        # [N,E]
        router_logits = self.router(x_flat).to(torch.float32)
        router_scores = torch.sigmoid(router_logits)

        # top-(K+1): K routes + 1 cutoff for QB
        _, idx = torch.topk(router_scores + self.expert_bias, self.topk + 1, dim=-1)
        topk_idx = idx[:, : self.topk]  # [N,K]
        alpha = (
            router_scores.gather(-1, idx[:, self.topk].unsqueeze(-1))
            .squeeze(-1)
            .detach()
        )  # [N]

        chosen = router_scores.gather(-1, topk_idx)  # [N,K]
        weights = chosen / chosen.sum(-1, keepdim=True).clamp_min(1e-9)

        if self.training:
            self._qb_margin_chunks.append(
                (router_scores - alpha.unsqueeze(-1)).detach()
            )

        return topk_idx, weights

    @torch.inference_mode()
    def update_bias(self) -> torch.Tensor:
        if not self._qb_margin_chunks:
            return self.expert_bias
        margins = torch.cat(self._qb_margin_chunks)
        b_hat = -torch.quantile(
            margins.float(), 1 - self.topk / self.num_experts, dim=0
        )
        self.expert_bias.copy_(b_hat - b_hat.mean())
        self._qb_margin_chunks.clear()
        return self.expert_bias

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden = x.shape
        x_flat = x.reshape(-1, hidden)
        N = x_flat.shape[0]
        E = self.num_experts
        K = self.topk

        # [N,D] @ [S,D,2I] -> [N,S,2I]
        gate_up_shared = torch.einsum("nd,sdi->nsi", x_flat, self.w13_shared)
        gate_s, up_s = gate_up_shared.chunk(2, dim=-1)
        shared_h = self.situ_glu(gate_s, up_s)  # [N,S,I]
        shared_out = torch.einsum("nsi,sid->nd", shared_h, self.w2_shared)  # [N,D]

        topk_idx, weights = self.route(x_flat)
        z = self.down_proj(x_flat)  # [N,L]

        expert_ids = topk_idx.reshape(-1)  # [N*K]
        token_ids = (
            torch.arange(N, device=x.device).unsqueeze(1).expand(N, K).reshape(-1)
        )  # [N*K]
        gates = weights.reshape(-1).to(z.dtype)  # [N*K]

        expert_ids, perm = torch.sort(expert_ids)
        token_ids = token_ids[perm]  # [N*K]
        gates = gates[perm]  # [N*K]

        counts = torch.bincount(expert_ids, minlength=E)  # [E]
        if self.training:
            capacity = max(1, math.ceil(N * K / E * self.capacity_factor))
        else:
            # no token dropping at inference so outputs don't depend on N
            capacity = max(1, N * K)
        starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])  # [E]
        slot_ids = torch.arange(N * K, device=x.device) - starts[expert_ids]  # [N*K]
        keep = slot_ids < capacity  # [N*K]

        kept_slot_ids = slot_ids[keep]  # [M]
        kept_token_ids = token_ids[keep]  # [M]
        kept_gates = gates[keep]  # [M]
        dropped = int((~keep).sum())
        dropped_frac = dropped / max(N * K, 1)

        packed = z.new_zeros(E, capacity, self.moe_latent_dim)  # [E,C,L]
        packed[expert_ids[keep], kept_slot_ids] = z[kept_token_ids]

        proj = torch.matmul(packed, self.w13)  # [E,C,L] @ [E,L,2I] -> [E,C,2I]
        gate, up = proj.chunk(2, dim=-1)  # [E,C,I]
        act = self.situ_glu(gate, up)  # [E,C,I]
        expert_out = torch.matmul(act, self.w2)  # [E,C,I] @ [E,I,L] -> [E,C,L]

        valid = expert_out[expert_ids[keep], kept_slot_ids]  # [M,L]
        u = z.new_zeros(N, self.moe_latent_dim)  # [N,L]
        u.index_add_(0, kept_token_ids, valid * kept_gates.unsqueeze(-1))

        routed_out = self.up_proj(self.norm(u))  # [N,D]
        output = (shared_out + routed_out).view(bsz, seq_len, hidden)

        load = counts.float() / (counts.sum() + 1e-9)  # [E]
        entropy = -(load * (load + 1e-9).log()).sum()

        router_metrics = {
            "load_std": load.std().detach().reshape(1),
            "load_max": load.max().detach().reshape(1),
            "load_min": load.min().detach().reshape(1),
            "load_ratio": (load.max() / (load.min() + 1e-9)).detach().reshape(1),
            "utilization": (torch.exp(entropy) / E).detach().reshape(1),
            "entropy": entropy.detach(),
            "dropped_frac": torch.tensor(dropped_frac),
            "dropped_tokens": torch.tensor(float(dropped)),
        }

        return output, router_metrics
