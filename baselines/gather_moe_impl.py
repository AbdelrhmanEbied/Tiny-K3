"""Archived baseline: gather-based dispatch MoE forward.

This was Tiny-K3's first working Stable LatentMoE implementation
(model/moe.py before the packed-dispatch rewrite on 2026-08-24).

It routes identically to the current implementation (sigmoid scores,
top-(K+1) selection with expert_bias, renormalized raw-score weights)
but computes experts by indexing expert weights per token-slot:

    w13_sel = self.w13[topk_idx]   # [N,K,L,2I] copy

which materializes a full copy of every selected expert's weights for
every token-slot each forward pass. Correct but O(N*K*L*2I) memory and
~5-22x slower than packed dispatch on Tesla T4. See
Reports/moe_benchmark_report.md.

Kept for reference and as a correctness oracle in benchmarks/bench_moe.py.
Import via: `sys.path` or package-relative import from `baselines/`.
"""

import torch
import torch.nn.functional as F


def gather_forward(moe, x: torch.Tensor):
    """Drop-in equivalent of the old MoE.forward compute (minus metrics).

    Args:
        moe: model.moe.MoE instance (any dispatch impl, weights are shared).
        x: [B,T,D] input.
    Returns:
        [B,T,D] output.
    """
    bsz, seq_len, hidden = x.shape
    x_flat = x.reshape(-1, hidden)
    N = x_flat.shape[0]
    K = moe.topk

    gate_up_shared = torch.einsum("nd,sdi->nsi", x_flat, moe.w13_shared)
    gate_s, up_s = gate_up_shared.chunk(2, dim=-1)
    shared_h = F.silu(gate_s) * up_s
    shared_out = torch.einsum("nsi,sid->nd", shared_h, moe.w2_shared)

    topk_idx, weights = moe.route(x_flat)
    z = moe.down_proj(x_flat)

    w13_sel = moe.w13[topk_idx]  # [N,K,L,2I]
    w2_sel = moe.w2[topk_idx]  # [N,K,I,L]

    z_per_slot = z.unsqueeze(1).expand(N, K, moe.moe_latent_dim)
    gate_up = torch.einsum("nkl,nkli->nki", z_per_slot, w13_sel)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate) * up
    expert_out = torch.einsum("nki,nkil->nkl", act, w2_sel)
    u = (expert_out * weights.to(expert_out.dtype).unsqueeze(-1)).sum(dim=1)

    routed_out = moe.up_proj(moe.norm(u))
    output = (shared_out + routed_out).view(bsz, seq_len, hidden)
    return output
