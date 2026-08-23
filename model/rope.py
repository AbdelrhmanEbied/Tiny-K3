from __future__ import annotations

import math

import torch
from torch import nn

from configs.model_config import ModelConfig


class RoPE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.dim = cfg.qk_rope_dim
        self.rope_type = cfg.rope_type

        self.max_seq_len = cfg.max_seq_len
        self.original_max_seq_len = cfg.original_max_seq_len
        self.factor = cfg.factor

        self.beta_fast = cfg.beta_fast
        self.beta_slow = cfg.beta_slow
        self.theta = cfg.rope_theta

        if self.dim % 2 != 0:
            raise ValueError("qk_rope_dim must be even")

        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )

        if self.rope_type == "yarn":
            inv_freq = self._yarn(inv_freq)

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer(
            "freqs_cis",
            torch.polar(torch.ones_like(freqs), freqs),
            persistent=False,
        )

    def _yarn(self, inv_freq):
        s = max(self.factor, 1)

        if s == 1:
            return inv_freq

        wavelength = 2 * math.pi / inv_freq
        r = self.original_max_seq_len / wavelength
        ramp = ((r - self.beta_slow) / (self.beta_fast - self.beta_slow)).clamp(
            0.0, 1.0
        )
        scaled_inv_freq = inv_freq / s
        inv_freq_yarn = (1 - ramp) * scaled_inv_freq + ramp * inv_freq
        return inv_freq_yarn

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None):
        squeeze_head = False

        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeeze_head = True
        elif x.dim() != 4:
            raise ValueError("RoPE expects x with shape [B, T, D] or [B, T, H, D]")

        dtype = x.dtype
        device = x.device
        bsz, seq_len, n_heads, dim = x.shape

        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds max_seq_len={self.max_seq_len}. Increase max_seq_len or extend the cache."
            )

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        else:
            position_ids = torch.as_tensor(
                position_ids, device=device, dtype=torch.long
            )

        if position_ids.dim() == 0:
            position_ids = position_ids.view(1)

        if position_ids.dim() == 1:
            freqs = (
                self.freqs_cis.index_select(0, position_ids).unsqueeze(0).unsqueeze(2)
            )
        elif position_ids.dim() == 2:
            if position_ids.shape != (bsz, seq_len):
                raise ValueError("position_ids shape must match [B, T]")
            freqs = self.freqs_cis[position_ids].unsqueeze(2)
        else:
            raise ValueError("position_ids must have shape [T] or [B, T]")

        x_complex = torch.view_as_complex(
            x.float().reshape(bsz, seq_len, n_heads, dim // 2, 2)
        )

        y = x_complex * freqs

        y = torch.view_as_real(y).flatten(-2).to(dtype)

        return y.squeeze(2) if squeeze_head else y
