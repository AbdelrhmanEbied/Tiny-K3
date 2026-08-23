from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from configs.model_config import ModelConfig
from model.rope import RoPE


class MLA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.qk_nope_dim + cfg.qk_rope_dim

        self.max_seq_len = cfg.max_seq_len
        self.original_max_seq_len = cfg.original_max_seq_len
        self.absorb_weights = cfg.absorb_weights

        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_dim = cfg.qk_nope_dim
        self.qk_rope_dim = cfg.qk_rope_dim

        self.rope = RoPE(cfg)
        self.factor = cfg.factor
        self.mscale = 0.1 * cfg.mscale * math.log(self.factor) + 1

        # fused input projection: [c_kv | c_q | k_rope] in one GEMM
        self.w_in = nn.Linear(
            self.dim, 2 * self.kv_lora_rank + self.qk_rope_dim, bias=False
        )

        # q projections split so the absorbed path can skip computing the nope half
        self.w_uq_nope = nn.Linear(
            self.kv_lora_rank, self.num_heads * self.qk_nope_dim, bias=False
        )
        self.w_qr_rope = nn.Linear(
            self.kv_lora_rank, self.num_heads * self.qk_rope_dim, bias=False
        )

        self.w_uk = nn.Parameter(
            torch.empty(self.num_heads, self.qk_nope_dim, self.kv_lora_rank)
        )
        self.w_uv = nn.Parameter(
            torch.empty(self.num_heads, self.kv_lora_rank, self.head_dim)
        )

        self.w_kr = nn.Linear(self.dim, self.qk_rope_dim, bias=False)
        self.w_o = nn.Linear(self.num_heads * self.head_dim, self.dim, bias=False)

        base_scale = (self.head_dim) ** -0.5
        use_scaled_attention = False
        if self.max_seq_len > self.original_max_seq_len and self.factor > 1.0:
            use_scaled_attention = True

        self.softmax_scale = (
            base_scale * (self.mscale**2) if use_scaled_attention else base_scale
        )

        self.register_buffer("kv_cache", None, persistent=False)
        self.register_buffer("pe_cache", None, persistent=False)

        if self.absorb_weights:
            self.register_buffer(
                "w_q_absorbed",
                torch.empty(self.num_heads, self.kv_lora_rank, self.kv_lora_rank),
                persistent=False,
            )

            self.register_buffer(
                "w_out_absorbed",
                torch.empty(self.dim, self.num_heads, self.kv_lora_rank),
                persistent=False,
            )
            # pre-flattened [H*Dc, D] view for the single-GEMM output projection
            self.register_buffer(
                "w_out_absorbed_flat",
                torch.empty(self.num_heads * self.kv_lora_rank, self.dim),
                persistent=False,
            )

        self._weights_absorbed = False
        self.cache_len = 0

        nn.init.xavier_uniform_(self.w_uk)
        nn.init.xavier_uniform_(self.w_uv)

    def _ensure_cache(self, batch_size: int, device=None, dtype=None):
        if self.training:
            return
        if self.kv_cache is None or self.kv_cache.shape[0] != batch_size:
            device = device or next(self.parameters()).device
            dtype = dtype or next(self.parameters()).dtype
            self.kv_cache = torch.zeros(
                batch_size,
                self.max_seq_len,
                self.kv_lora_rank,
                device=device,
                dtype=dtype,
            )
            self.pe_cache = torch.zeros(
                batch_size,
                self.max_seq_len,
                self.qk_rope_dim,
                device=device,
                dtype=dtype,
            )
            self.cache_len = 0

    def _absorb_weights(self):
        if self._weights_absorbed:
            return

        if not self.absorb_weights:
            raise RuntimeError(
                "Cannot absorb weights: absorb_weights=False in the model config. "
                "Set cfg.absorb_weights=True to enable weight absorption."
            )

        with torch.inference_mode():
            # [H,Dn,Dc]
            w_nope = self.w_uq_nope.weight.view(
                self.num_heads, self.qk_nope_dim, self.kv_lora_rank
            )

            # [H,Dc,Dn] @ [H,Dn,Dc] -> [H,Dc,Dc]: fold the q up-projection into W_uk
            w_q_absorbed = torch.matmul(w_nope.transpose(-1, -2), self.w_uk)

            w_o = (
                self.w_o.weight.contiguous()
            )  # w_o = [H*Dv,D] and w_uv = [H,Dc,Dv] now how do we make them [D,H,Dc]
            # let's view it into [D,H,Dc]
            w_o = w_o.view(self.dim, self.num_heads, self.head_dim)  # [D,H,Dv]
            w_uv = self.w_uv.transpose(-1, -2)  # [H,Dv,Dc]
            w_out_absorbed = torch.einsum("dhv,hvk->dhk", w_o, w_uv)

            self.w_q_absorbed.copy_(w_q_absorbed)
            self.w_out_absorbed.copy_(w_out_absorbed)
            # row index = h * kv_lora_rank + k so out = attn_flat @ flat
            self.w_out_absorbed_flat.copy_(
                w_out_absorbed.permute(1, 2, 0).reshape(
                    self.num_heads * self.kv_lora_rank, self.dim
                )
            )

            assert w_q_absorbed.shape == (
                self.num_heads,
                self.kv_lora_rank,
                self.kv_lora_rank,
            )

            assert w_out_absorbed.shape == (
                self.dim,
                self.num_heads,
                self.kv_lora_rank,
            )

            del self.w_uk
            del self.w_uv

            self._weights_absorbed = True

    def _project_input(self, x: torch.Tensor):
        proj = self.w_in(x)  # [B,T,2*Dc+Dr]
        c_kv = proj[..., : self.kv_lora_rank]
        c_q = proj[..., self.kv_lora_rank : 2 * self.kv_lora_rank]
        k_rope_raw = proj[..., 2 * self.kv_lora_rank :]
        return c_kv, c_q, k_rope_raw

    def _normal_forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ):
        bsz, seq_len, _ = x.shape
        start = self.cache_len
        end = start + seq_len
        if position_ids is None:
            position_ids = (
                torch.arange(start, end, device=x.device).unsqueeze(0).expand(bsz, -1)
            )

        c_kv, c_q, k_rope_raw = self._project_input(x)

        q_nope = self.w_uq_nope(c_q).view(
            bsz, seq_len, self.num_heads, self.qk_nope_dim
        )
        q_rope = self.w_qr_rope(c_q).view(
            bsz, seq_len, self.num_heads, self.qk_rope_dim
        )
        q_rope = self.rope(q_rope, position_ids)
        k_rope = self.rope(k_rope_raw, position_ids)  # [B,T,Dr] shared across heads

        q_nope = q_nope.transpose(1, 2)  # [B,H,T,Dn]
        q_rope = q_rope.transpose(1, 2)  # [B,H,T,Dr]

        if self.training:
            # [B,1,T,Dc] @ [H,Dc,Dn] -> [B,H,T,Dn]
            k_nope = c_kv.unsqueeze(1) @ self.w_uk.transpose(-1, -2)
            k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

            q = torch.cat([q_nope, q_rope], dim=-1)  # [B,H,T,Dn+Dr]
            k = torch.cat([k_nope, k_rope], dim=-1)  # [B,H,T,Dn+Dr]
            v = c_kv.unsqueeze(1) @ self.w_uv  # [B,H,T,Dv]

            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.softmax_scale
            )
        else:
            self._ensure_cache(bsz, x.device, x.dtype)
            self.kv_cache[:bsz, start:end] = c_kv
            self.pe_cache[:bsz, start:end] = k_rope
            self.cache_len = end

            past_ckv = self.kv_cache[:bsz, :end]  # [B,S,Dc]
            past_krope = self.pe_cache[:bsz, :end]  # [B,S,Dr]

            # [B,1,S,Dc] @ [H,Dc,Dn] -> [B,H,S,Dn]
            k_nope = past_ckv.unsqueeze(1) @ self.w_uk.transpose(-1, -2)
            k_rope = past_krope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            v = past_ckv.unsqueeze(1) @ self.w_uv  # [B,H,S,Dv]

            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=start == 0,
                attn_mask=None
                if start == 0
                else self._causal_mask(start, end, seq_len, x.device),
                scale=self.softmax_scale,
            )

        out = out.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.head_dim
        )  # [B,T,H*Dv]
        return self.w_o(out)  # [B,T,D]

    def _absorbed_forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ):
        bsz, seq_len, _ = x.shape
        start = self.cache_len
        end = start + seq_len
        if position_ids is None:
            position_ids = (
                torch.arange(start, end, device=x.device).unsqueeze(0).expand(bsz, -1)
            )

        c_kv, c_q, k_rope_raw = self._project_input(x)

        # only the rope projection is needed; the nope half is folded into w_q_absorbed
        q_rope = self.rope(
            self.w_qr_rope(c_q).view(bsz, seq_len, self.num_heads, self.qk_rope_dim),
            position_ids,
        ).transpose(1, 2)  # [B,H,T,Dr]
        k_rope = self.rope(k_rope_raw, position_ids)  # [B,T,Dr]

        self._ensure_cache(bsz, x.device, x.dtype)
        self.kv_cache[:bsz, start:end] = c_kv
        self.pe_cache[:bsz, start:end] = k_rope
        self.cache_len = end

        past_ckv = self.kv_cache[:bsz, :end]  # [B,S,Dc]
        past_krope = self.pe_cache[:bsz, :end]  # [B,S,Dr]

        # [B,T,H,Dc] -> [B,H,T,Dc]: c_q directly through the absorbed projection
        q_lat = torch.einsum("btd,hdc->bhtc", c_q, self.w_q_absorbed.to(c_q.dtype))

        # keys/values are shared across heads; expand is a zero-copy stride view
        k_lat = past_ckv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        k_rope = past_krope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        q = torch.cat([q_lat, q_rope], dim=-1)  # [B,H,T,Dc+Dr]
        k = torch.cat([k_lat, k_rope], dim=-1)  # [B,H,S,Dc+Dr]

        out = F.scaled_dot_product_attention(
            q,
            k,
            k_lat,  # v in latent space; w_uv folded into w_out_absorbed
            is_causal=start == 0,
            attn_mask=None
            if start == 0
            else self._causal_mask(start, end, seq_len, x.device),
            scale=self.softmax_scale,
        )  # [B,H,T,Dc]

        # single GEMM over all heads: [B,T,H*Dc] @ [H*Dc,D]
        attn_flat = out.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.kv_lora_rank
        )
        return attn_flat @ self.w_out_absorbed_flat.to(attn_flat.dtype)  # [B,T,D]

    def _causal_mask(self, start: int, end: int, seq_len: int, device) -> torch.Tensor:
        """query i (global pos start+i) may attend to keys j <= start+i"""
        return (
            torch.arange(end, device=device)[None, :]
            <= (start + torch.arange(seq_len, device=device))[:, None]
        )

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        if not self.training and self._weights_absorbed:
            return self._absorbed_forward(x, position_ids)
        return self._normal_forward(x, position_ids)
