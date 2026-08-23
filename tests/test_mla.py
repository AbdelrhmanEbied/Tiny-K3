import pytest
import torch

from configs.model_config import ModelConfig
from model.mla import MLA


def _make_cfg(**kwargs):
    defaults = {
        "hidden_size": 64,
        "num_attention_heads": 2,
        "kv_lora_rank": 16,
        "qk_nope_dim": 8,
        "qk_rope_dim": 8,
        "max_seq_len": 32,
        "original_max_seq_len": 32,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def test_mla_construction():
    cfg = _make_cfg()
    model = MLA(cfg)
    assert model.dim == 64
    assert model.num_heads == 2
    assert model.head_dim == 16
    assert model.kv_lora_rank == 16


def test_mla_parameter_shapes():
    cfg = _make_cfg()
    model = MLA(cfg)
    assert model.w_in.weight.shape == (2 * cfg.kv_lora_rank, cfg.hidden_size)
    assert model.w_uq_qr.weight.shape == (
        cfg.num_attention_heads * (cfg.qk_nope_dim + cfg.qk_rope_dim),
        cfg.kv_lora_rank,
    )
    assert model.w_uk.shape == (
        cfg.num_attention_heads,
        cfg.qk_nope_dim,
        cfg.kv_lora_rank,
    )
    assert model.w_uv.shape == (
        cfg.num_attention_heads,
        cfg.kv_lora_rank,
        cfg.qk_nope_dim + cfg.qk_rope_dim,
    )
    assert model.w_kr.weight.shape == (cfg.qk_rope_dim, cfg.hidden_size)
    assert model.w_o.weight.shape == (
        cfg.hidden_size,
        cfg.num_attention_heads * (cfg.qk_nope_dim + cfg.qk_rope_dim),
    )


def test_mla_has_rope_submodule():
    cfg = _make_cfg()
    model = MLA(cfg)
    assert isinstance(model.rope, torch.nn.Module)


def test_mla_weights_initialized_not_identity():
    cfg = _make_cfg()
    model = MLA(cfg)
    assert not torch.allclose(model.w_uk, torch.zeros_like(model.w_uk))
    assert not torch.allclose(model.w_uv, torch.zeros_like(model.w_uv))


def test_mla_softmax_scale_default():
    cfg = _make_cfg()
    model = MLA(cfg)
    expected = (cfg.qk_nope_dim + cfg.qk_rope_dim) ** -0.5
    assert model.softmax_scale == pytest.approx(expected)


def test_mla_softmax_scale_yarn_scaled():
    cfg = _make_cfg(
        max_seq_len=64, original_max_seq_len=32, rope_type="yarn", factor=4.0
    )
    model = MLA(cfg)
    head_dim = cfg.qk_nope_dim + cfg.qk_rope_dim
    expected_mscale = 0.1 * cfg.mscale * torch.log(torch.tensor(4.0)).item() + 1
    assert model.softmax_scale == pytest.approx(head_dim**-0.5 * expected_mscale**2)


def test_mla_softmax_scale_not_scaled_when_seq_within_original():
    cfg = _make_cfg(
        max_seq_len=32, original_max_seq_len=64, rope_type="yarn", factor=4.0
    )
    model = MLA(cfg)
    assert model.softmax_scale == pytest.approx(
        (cfg.qk_nope_dim + cfg.qk_rope_dim) ** -0.5
    )


def test_mla_train_mode_cache_not_allocated():
    cfg = _make_cfg()
    model = MLA(cfg)
    assert model.kv_cache is None
    assert model.pe_cache is None
    model._ensure_cache(batch_size=2)
    assert model.kv_cache is None
    assert model.pe_cache is None


def test_mla_eval_mode_lazy_allocates_kv_cache():
    cfg = _make_cfg(max_seq_len=32)
    model = MLA(cfg)
    model.eval()
    assert model.kv_cache is None
    model._ensure_cache(batch_size=2)
    assert model.kv_cache.shape == (2, 32, cfg.kv_lora_rank)
    assert model.pe_cache.shape == (2, 32, cfg.qk_rope_dim)


def test_mla_cache_reallocates_on_batch_size_change():
    cfg = _make_cfg(max_seq_len=32)
    model = MLA(cfg)
    model.eval()
    model._ensure_cache(batch_size=1)
    old = model.kv_cache
    model._ensure_cache(batch_size=4)
    assert model.kv_cache.shape[0] == 4
    assert model.kv_cache is not old


def test_mla_absorb_disabled_no_absorbed_buffers():
    cfg = _make_cfg(absorb_weights=False)
    model = MLA(cfg)
    assert not hasattr(model, "w_q_absorbed")
    assert not hasattr(model, "w_out_absorbed")


def test_mla_absorb_enabled_registers_absorbed_buffers():
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    assert model.w_q_absorbed.shape == (
        cfg.num_attention_heads,
        cfg.kv_lora_rank,
        cfg.kv_lora_rank,
    )
    assert model.w_out_absorbed.shape == (
        cfg.hidden_size,
        cfg.num_attention_heads,
        cfg.kv_lora_rank,
    )


def test_mla_absorb_works_with_unequal_nope_rope_dims():
    cfg = _make_cfg(absorb_weights=True, qk_nope_dim=12, qk_rope_dim=8)
    model = MLA(cfg)
    model.eval()
    model._absorb_weights()
    assert model.w_q_absorbed.shape == (
        cfg.num_attention_heads,
        cfg.kv_lora_rank,
        cfg.kv_lora_rank,
    )
    assert torch.isfinite(model.w_q_absorbed).all()


def test_mla_absorb_disabled_raises_on_absorb_call():
    cfg = _make_cfg(absorb_weights=False)
    model = MLA(cfg)
    model.eval()
    with pytest.raises(RuntimeError, match="absorb_weights"):
        model._absorb_weights()
    assert not model._weights_absorbed


def _reset_cache(model):
    model.cache_len = 0
    if model.kv_cache is not None:
        model.kv_cache.zero_()
        model.pe_cache.zero_()


def _reference_mla(model: MLA, x: torch.Tensor):
    """Explicit reference implementation of the MLA math (no SDPA, no cache).

    score(t,j) = (q_nope.W_uk c_kv_j + q_rope.k_rope_j) * softmax_scale
    out_t      = w_o(concat_h sum_j p_tj * W_uv_h^T? c_kv_j)   with p = causal softmax
    """
    B, T, _ = x.shape
    H = model.num_heads
    Dn = model.qk_nope_dim
    Dr = model.qk_rope_dim
    Dc = model.kv_lora_rank

    pos = torch.arange(T).expand(B, -1)

    c_in = model.w_in(x)
    c_kv, c_q = c_in.split(Dc, dim=-1)

    q_proj = model.w_uq_qr(c_q).view(B, T, H, Dn + Dr)
    q_nope, q_rope = q_proj.split([Dn, Dr], dim=-1)
    q_rope = model.rope(q_rope, pos)
    k_rope = model.rope(model.w_kr(x), pos)

    # per-head queries/keys/values
    q = torch.cat(
        [q_nope.transpose(1, 2), q_rope.transpose(1, 2)], dim=-1
    )  # [B,H,T,Dn+Dr]
    k_nope = torch.einsum("btc,hdc->bhtd", c_kv, model.w_uk)  # [B,H,T,Dn]
    k = torch.cat(
        [k_nope, k_rope.unsqueeze(1).expand(-1, H, -1, -1)], dim=-1
    )  # [B,H,T,Dn+Dr]
    v = torch.einsum("btc,hcv->bhtv", c_kv, model.w_uv)  # [B,H,T,Dv]

    scores = (q @ k.transpose(-1, -2)) * model.softmax_scale  # [B,H,T,T]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    attn = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)

    out = attn @ v  # [B,H,T,Dv]
    out = out.transpose(1, 2).reshape(B, T, H * (Dn + Dr))
    return model.w_o(out), attn


def test_mla_matches_reference_math():
    """Model output must equal an explicit hand-computed attention (softmax over true scores)."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    x = torch.randn(2, 7, cfg.hidden_size)

    expected, _ = _reference_mla(model, x)
    actual = model(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_mla_attention_rows_sum_to_one():
    """Softmax rows must be valid probability distributions."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    x = torch.randn(1, 5, cfg.hidden_size)
    _, attn = _reference_mla(model, x)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_mla_causal_no_future_leak():
    """Changing token t must not change outputs at positions < t."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    x1 = torch.randn(1, 6, cfg.hidden_size)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])  # perturb last two tokens

    out1, out2 = model(x1), model(x2)
    assert torch.allclose(out1[:, :4], out2[:, :4], atol=1e-6)
    assert not torch.allclose(out1[:, 4:], out2[:, 4:])


def test_mla_decode_step_matches_reference_math():
    """Cache-based incremental decoding must reproduce the full-sequence reference math."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    model.eval()
    x = torch.randn(1, 5, cfg.hidden_size)

    with torch.inference_mode():
        expected, _ = _reference_mla(model, x)

        outs = []
        for i in range(5):
            outs.append(model(x[:, i : i + 1, :]))
        actual = torch.cat(outs, dim=1)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_mla_attention_weights_upstream_gradient():
    """Gradients must reach the projection weights through the softmax path."""
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    x = torch.randn(1, 4, cfg.hidden_size)
    model(x).square().mean().backward()
    for name in ("w_in", "w_uq_qr", "w_kr", "w_o"):
        assert getattr(model, name).weight.grad is not None
        assert getattr(model, name).weight.grad.abs().sum() > 0
    assert model.w_uk.grad is not None and model.w_uk.grad.abs().sum() > 0
    assert model.w_uv.grad is not None and model.w_uv.grad.abs().sum() > 0


def test_mla_training_forward_shape():
    cfg = _make_cfg()
    model = MLA(cfg)
    x = torch.randn(2, 8, cfg.hidden_size)
    out = model(x)
    assert out.shape == (2, 8, cfg.hidden_size)
    assert torch.isfinite(out).all()


def test_mla_training_forward_gradient_flow():
    cfg = _make_cfg()
    model = MLA(cfg)
    x = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None


def test_mla_eval_prefill_and_decode_shapes():
    cfg = _make_cfg()
    model = MLA(cfg)
    model.eval()
    prefill = model(torch.randn(2, 6, cfg.hidden_size))
    assert prefill.shape == (2, 6, cfg.hidden_size)
    assert model.cache_len == 6
    decode = model(torch.randn(2, 1, cfg.hidden_size))
    assert decode.shape == (2, 1, cfg.hidden_size)
    assert model.cache_len == 7


def test_mla_incremental_matches_full_forward():
    torch.manual_seed(0)
    cfg = _make_cfg()
    model = MLA(cfg)
    model.eval()
    with torch.inference_mode():
        x = torch.randn(1, 6, cfg.hidden_size)
        full = model(x)[:, -1, :]

        _reset_cache(model)
        outs = [model(x[:, i : i + 1, :]) for i in range(6)]
        incr = torch.cat(outs, dim=1)[:, -1, :]
    assert torch.allclose(full, incr, atol=1e-5)


def test_mla_absorbed_vs_unabsorbed_equivalence():
    torch.manual_seed(0)
    cfg = _make_cfg(absorb_weights=True)
    ref = MLA(cfg)
    absorbed = MLA(cfg)
    absorbed.load_state_dict(ref.state_dict())
    absorbed.eval()
    absorbed._absorb_weights()

    ref.eval()
    with torch.inference_mode():
        for _ in range(3):
            x = torch.randn(1, 4, cfg.hidden_size)
            out_ref = ref(x)
            out_abs = absorbed(x)
            assert out_ref.shape == out_abs.shape == (1, 4, cfg.hidden_size)
            assert torch.allclose(out_ref, out_abs, atol=1e-5)


def test_mla_absorbed_decode_step():
    torch.manual_seed(0)
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    model.eval()
    model._absorb_weights()
    with torch.inference_mode():
        model(torch.randn(1, 5, cfg.hidden_size))
        step = model(torch.randn(1, 1, cfg.hidden_size))
    assert step.shape == (1, 1, cfg.hidden_size)
    assert torch.isfinite(step).all()


def test_mla_absorb_sets_flag_and_removes_low_rank_weights():
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    model.eval()
    model._absorb_weights()
    assert model._weights_absorbed
    assert not hasattr(model, "w_uk")
    assert not hasattr(model, "w_uv")


def test_mla_absorb_is_idempotent():
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    model.eval()
    model._absorb_weights()
    w_q_first = model.w_q_absorbed.clone()
    model._absorb_weights()
    assert torch.allclose(model.w_q_absorbed, w_q_first)


def test_mla_absorbed_q_matches_manual_computation():
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    model.eval()

    w_uq_qr = (
        model.w_uq_qr.weight.detach()
        .clone()
        .view(
            cfg.num_attention_heads, cfg.qk_nope_dim + cfg.qk_rope_dim, cfg.kv_lora_rank
        )
    )
    w_nope = w_uq_qr[:, : cfg.qk_nope_dim, :]
    expected_q = torch.matmul(w_nope.transpose(-1, -2), model.w_uk.detach())

    model._absorb_weights()
    assert torch.allclose(model.w_q_absorbed, expected_q, atol=1e-6)


def test_mla_absorbed_out_matches_manual_computation():
    cfg = _make_cfg(absorb_weights=True)
    model = MLA(cfg)
    model.eval()

    head_dim = cfg.qk_nope_dim + cfg.qk_rope_dim
    w_o = (
        model.w_o.weight.detach()
        .clone()
        .view(cfg.hidden_size, cfg.num_attention_heads, head_dim)
    )
    w_uv_t = model.w_uv.detach().transpose(-1, -2)
    expected_out = torch.einsum("dhv,hvk->dhk", w_o, w_uv_t)

    model._absorb_weights()
    assert torch.allclose(model.w_out_absorbed, expected_out, atol=1e-6)
