from __future__ import annotations

import codecs
import warnings
from collections.abc import Callable, Iterator
from typing import ClassVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from configs.model_config import ModelConfig
from model.attn_residuals import AttentionResidual
from model.mla import MLA
from model.moe import MoE
from model.rms_norm import RMSNorm
from model.rope import RoPE
from model.sampling import sample_token


def softcap(x: torch.Tensor, beta: float) -> torch.Tensor:
    return beta * torch.tanh(x / beta)


class DenseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.situ_beta_gate = cfg.situ_beta_gate
        self.situ_beta_up = cfg.situ_beta_up
        self.gate_up_proj = nn.Linear(
            cfg.hidden_size, 2 * cfg.moe_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            cfg.moe_intermediate_size, cfg.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        act = softcap(gate, self.situ_beta_gate)
        act = act * torch.sigmoid(gate)
        act = act * softcap(up, self.situ_beta_up)
        return self.down_proj(act)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_number = layer_idx + 1
        self.layers_per_block = cfg.attnres_block_layers

        self.attn_res = AttentionResidual(cfg)
        self.attn_norm = RMSNorm(cfg)
        self.attn = MLA(cfg)

        self.ffn_res = AttentionResidual(cfg)
        self.ffn_norm = RMSNorm(cfg)
        if layer_idx < cfg.first_k_dense_replace:
            self.ffn: DenseFFN | MoE = DenseFFN(cfg)
        else:
            self.ffn = MoE(cfg)

    def is_block_boundary(self) -> bool:
        return self.layer_number % self.layers_per_block == 0

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, dict | None]:
        h = self.attn_res(blocks, partial_block)

        if self.is_block_boundary():
            blocks = (
                [*blocks, partial_block] if partial_block is not None else [*blocks]
            )
            partial_block = None

        attn_out = self.attn(self.attn_norm(h), position_ids=position_ids)
        partial_block = attn_out if partial_block is None else partial_block + attn_out

        h = self.ffn_res(blocks, partial_block)
        result = self.ffn(self.ffn_norm(h))
        if isinstance(result, tuple):
            ffn_out, router_metrics = result
        else:
            ffn_out, router_metrics = result, None
        partial_block = partial_block + ffn_out

        return blocks, partial_block, router_metrics


class TinyK3Model(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig
    base_model_prefix = "tiny_k3"
    _tied_weights_keys: ClassVar[dict] = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(cfg, layer_idx=i) for i in range(cfg.num_layers)
        )
        self.final_res = AttentionResidual(cfg)
        self.final_norm = RMSNorm(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, MoE):
            module.init_weights()
        elif isinstance(module, AttentionResidual):
            nn.init.zeros_(module.pseudo_query)
        elif isinstance(module, RoPE):
            module._build_buffers()
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.inference_mode()
    def aggregated_router_metrics(self) -> dict[str, torch.Tensor]:
        per_layer = self.last_router_metrics
        if not per_layer:
            return {}
        keys = per_layer[0].keys()
        return {
            k: torch.stack([m[k] for m in per_layer]).mean(dim=0).detach() for k in keys
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        kwargs = dict(gradient_checkpointing_kwargs or {})
        if kwargs.pop("use_reentrant", False):
            warnings.warn(
                "TinyK3Model requires use_reentrant=False (block inputs are lists/tensors); "
                "ignoring use_reentrant=True.",
                stacklevel=2,
            )
        self._gradient_checkpointing_kwargs = kwargs
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        if position_ids is None:
            self._reset_caches()

        hidden = self.embed_tokens(input_ids)
        blocks = [hidden]
        partial_block = None
        router_metrics = []

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                blocks, partial_block, metrics = checkpoint(
                    layer, blocks, partial_block, position_ids, use_reentrant=False
                )
            else:
                blocks, partial_block, metrics = layer(
                    blocks, partial_block, position_ids
                )
            if metrics is not None:
                router_metrics.append(metrics)

        self.last_router_metrics = router_metrics

        hidden = self.final_res(blocks, partial_block)
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden).float()

        loss = None
        if labels is None:
            labels = input_ids
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> dict:
        return {"input_ids": input_ids, **kwargs}

    def _reset_caches(self) -> None:
        for layer in self.layers:
            layer.attn.kv_cache = None
            layer.attn.pe_cache = None
            layer.attn.cache_len = 0

    def _iter_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None,
        sampling_kwargs: dict,
    ) -> Iterator[tuple[int, torch.Tensor]]:
        """Incremental decoding with the MLA kv-cache.

        Every AttentionResidual stream value for a position is created during
        that position's own pass through the layers (forward() starts each call
        with blocks=[embeddings], partial=None), so a decode step is just
        single-token prefill: the MLA kv-cache provides all cross-position
        history. Yields (new_token_id, ids_so_far) after each append.
        """
        with torch.inference_mode():
            was_training = self.training
            self.eval()
            try:
                device = input_ids.device
                bsz = input_ids.shape[0]
                self._reset_caches()

                out = input_ids
                finished = torch.zeros(bsz, dtype=torch.bool, device=device)

                for step in range(max_new_tokens + 1):
                    streams = [self.embed_tokens(out[:, -1:] if step else out)]
                    partial_block = None
                    for layer in self.layers:
                        streams, partial_block, _ = layer(streams, partial_block)

                    if step == max_new_tokens:
                        break

                    h = self.final_norm(self.final_res(streams, partial_block))
                    logits = self.lm_head(h[:, -1]).float()

                    next_ids = torch.empty(bsz, dtype=input_ids.dtype, device=device)
                    for b in range(bsz):
                        context = out[b].tolist()
                        next_ids[b] = sample_token(
                            logits[b], context, **sampling_kwargs
                        )

                    if eos_token_id is not None:
                        next_ids = torch.where(
                            finished,
                            torch.full_like(next_ids, eos_token_id),
                            next_ids,
                        )
                        finished |= next_ids == eos_token_id

                    out = torch.cat([out, next_ids.unsqueeze(1)], dim=1)
                    for b in range(bsz):
                        yield int(next_ids[b]), out

                    if eos_token_id is not None and bool(finished.all()):
                        break
            finally:
                if was_training:
                    self.train()

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        decode_fn: Callable[[int], str] | None = None,
    ) -> Iterator[dict[str, object]]:
        """Stream generated tokens as {"token": int, "text": str} dicts.

        Text decoding: uses decode_fn if given; otherwise byte-level
        incremental UTF-8 when vocab_size <= 256 (handles multi-byte tokens
        split across steps); otherwise text is "".
        """
        if decode_fn is None and self.config.vocab_size <= 256:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")

            def decode_fn(t: int) -> str:
                return decoder.decode(bytes([t]))

        elif decode_fn is None:

            def decode_fn(t: int) -> str:
                return ""

        yield from (
            {"token": tok, "text": decode_fn(tok)}
            for tok, _ in self._iter_generate(
                input_ids,
                max_new_tokens,
                eos_token_id,
                {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                },
            )
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Generate up to max_new_tokens per row; returns the full id tensor."""
        out = input_ids
        for _, out in self._iter_generate(
            input_ids,
            max_new_tokens,
            eos_token_id,
            {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "no_repeat_ngram_size": no_repeat_ngram_size,
            },
        ):
            pass
        return out


AutoConfig.register("tiny_k3", ModelConfig)
AutoModelForCausalLM.register(ModelConfig, TinyK3Model)
