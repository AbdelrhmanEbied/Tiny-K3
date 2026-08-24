import torch

from configs.model_config import ModelConfig
from model.moe import MoE
from model.rms_norm import RMSNorm


def _make_cfg(**kwargs) -> ModelConfig:
    defaults = {
        "hidden_size": 16,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_shared_experts": 2,
        "moe_intermediate_size": 6,
        "moe_latent_dim": 8,
    }
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _make_moe(**kwargs) -> MoE:
    torch.manual_seed(0)
    cfg = _make_cfg(**kwargs)
    return MoE(cfg)


def test_forward_output_shape():
    moe = _make_moe()
    x = torch.randn(2, 5, 16)
    out, _metrics = moe(x)
    assert out.shape == (2, 5, 16)


def test_routed_experts_live_in_latent_space():
    cfg = _make_cfg()
    moe = MoE(cfg)
    assert moe.w13.shape == (
        cfg.num_experts,
        cfg.moe_latent_dim,
        2 * cfg.moe_intermediate_size,
    )
    assert moe.w2.shape == (
        cfg.num_experts,
        cfg.moe_intermediate_size,
        cfg.moe_latent_dim,
    )


def test_shared_experts_live_in_hidden_space():
    cfg = _make_cfg()
    moe = MoE(cfg)
    assert moe.w13_shared.shape == (
        cfg.num_shared_experts,
        cfg.hidden_size,
        2 * cfg.moe_intermediate_size,
    )
    assert moe.w2_shared.shape == (
        cfg.num_shared_experts,
        cfg.moe_intermediate_size,
        cfg.hidden_size,
    )


def test_moe_latent_differs_from_hidden_still_works():
    moe = _make_moe(moe_latent_dim=5)
    out, _ = moe(torch.randn(2, 3, 16))
    assert out.shape == (2, 3, 16)


def test_expert_bias_is_a_non_grad_buffer():
    moe = _make_moe()
    assert not any(
        "expert_bias" in name for name, p in moe.named_parameters() if p.requires_grad
    )
    assert moe.expert_bias.requires_grad is False
    assert moe.expert_bias.shape == (_make_cfg().num_experts,)


def test_route_shapes_and_valid_selection():
    moe = _make_moe()
    moe.eval()
    x_flat = torch.randn(7, 16)
    topk_idx, weights = moe.route(x_flat)
    assert topk_idx.shape == (7, 2)
    assert weights.shape == (7, 2)
    assert topk_idx.min() >= 0 and topk_idx.max() < 4
    assert all(len(set(row.tolist())) == 2 for row in topk_idx)


def test_weights_are_renormalized_raw_scores():
    moe = _make_moe()
    moe.eval()
    torch.manual_seed(1)
    x_flat = torch.randn(9, 16)
    topk_idx, weights = moe.route(x_flat)
    scores = torch.sigmoid(moe.router(x_flat).float())
    chosen = scores.gather(-1, topk_idx)
    expected = chosen / chosen.sum(-1, keepdim=True).clamp_min(1e-9)
    assert torch.allclose(weights, expected, atol=1e-6)


def test_zero_bias_selection_equals_plain_topk_of_scores():
    moe = _make_moe()
    moe.eval()
    moe.expert_bias.zero_()
    x_flat = torch.randn(6, 16)
    topk_idx, _ = moe.route(x_flat)
    scores = torch.sigmoid(moe.router(x_flat).float())
    _, expected = torch.topk(scores, 2, dim=-1)
    assert torch.equal(topk_idx.sort(dim=-1).values, expected.sort(dim=-1).values)


def test_bias_affects_selection_but_not_weight_formula():
    moe = _make_moe()
    moe.eval()
    torch.manual_seed(2)
    x_flat = torch.randn(12, 16)
    moe.expert_bias.zero_()
    idx_base, w_base = moe.route(x_flat)
    moe.expert_bias.fill_(0.75)
    idx_biased, w_biased = moe.route(x_flat)
    scores = torch.sigmoid(moe.router(x_flat).float())

    def expected_weights(idx):
        chosen = scores.gather(-1, idx)
        return chosen / chosen.sum(-1, keepdim=True)

    assert torch.allclose(w_base, expected_weights(idx_base), atol=1e-6)
    assert torch.allclose(w_biased, expected_weights(idx_biased), atol=1e-6)
    assert torch.allclose(w_biased.sum(-1), torch.ones(12), atol=1e-6)


def test_eval_forward_is_deterministic():
    moe = _make_moe()
    moe.eval()
    torch.manual_seed(3)
    x = torch.randn(2, 4, 16)
    with torch.inference_mode():
        out_a, _ = moe(x)
        out_b, _ = moe(x)
    assert torch.equal(out_a, out_b)


def _reference_forward(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, d = x.shape
    x_flat = x.reshape(-1, d)
    N = x_flat.shape[0]
    scores = torch.sigmoid(moe.router(x_flat).to(torch.float32))
    _, idx = torch.topk(scores + moe.expert_bias, moe.topk + 1, dim=-1)
    topk_idx = idx[:, : moe.topk]
    chosen = scores.gather(-1, topk_idx)
    weights = chosen / chosen.sum(-1, keepdim=True).clamp_min(1e-9)

    def situ_glu(gate, up):
        g = (
            moe.situ_beta_gate
            * torch.tanh(gate / moe.situ_beta_gate)
            * torch.sigmoid(gate)
        )
        u = moe.situ_beta_up * torch.tanh(up / moe.situ_beta_up)
        return g * u

    shared_out = torch.zeros(N, d)
    for s in range(moe.num_shared_experts):
        gate, up = (x_flat @ moe.w13_shared[s]).chunk(2, dim=-1)
        shared_out += situ_glu(gate, up) @ moe.w2_shared[s]
    z = moe.down_proj(x_flat)
    u = torch.zeros(N, moe.moe_latent_dim)
    for n in range(N):
        for k in range(moe.topk):
            e = topk_idx[n, k].item()
            gate, up = (z[n] @ moe.w13[e]).chunk(2, dim=-1)
            u[n] += weights[n, k] * (situ_glu(gate, up) @ moe.w2[e])
    routed_out = moe.up_proj(moe.norm(u))
    return (shared_out + routed_out).view(bsz, seq_len, d)


def test_forward_matches_naive_reference():
    moe = _make_moe(moe_latent_dim=5)
    moe.eval()
    torch.manual_seed(4)
    x = torch.randn(2, 6, 16)
    with torch.inference_mode():
        out, _ = moe(x)
        ref = _reference_forward(moe, x)
    assert torch.allclose(out, ref, atol=1e-5)


def test_update_bias_matches_eq14_on_stored_margins():
    moe = _make_moe()
    moe.train()
    torch.manual_seed(5)
    x_flat = torch.randn(20, 16)
    moe.route(x_flat)
    margins = torch.cat(moe._qb_margin_chunks)
    q_level = 1 - moe.topk / moe.num_experts
    b_hat = -torch.quantile(margins, q_level, dim=0)
    expected = b_hat - b_hat.mean()
    moe.update_bias()
    assert torch.allclose(moe.expert_bias, expected, atol=1e-6)


def test_update_bias_accumulates_across_micro_batches():
    moe = _make_moe()
    moe.train()
    torch.manual_seed(7)
    chunks = [torch.randn(8, moe.num_experts) for _ in range(3)]
    moe._qb_margin_chunks = list(chunks)
    moe.update_bias()
    assert moe._qb_margin_chunks == []
    margins = torch.cat(chunks)
    q_level = 1 - moe.topk / moe.num_experts
    b_hat = -torch.quantile(margins, q_level, dim=0)
    assert torch.allclose(moe.expert_bias, b_hat - b_hat.mean(), atol=1e-6)


def test_qb_drives_skewed_load_toward_target():
    moe = _make_moe(num_experts=8, num_experts_per_token=2)
    moe.train()
    with torch.inference_mode():
        moe.router.weight.normal_(std=0.5)
        moe.router.weight[0].mul_(8.0)
    torch.manual_seed(6)
    x_flat = torch.randn(64, 16)

    def load_ratio():
        counts = torch.bincount(moe.route(x_flat)[0].reshape(-1), minlength=8).float()
        return (counts.max() / counts.min().clamp_min(1)).item(), counts

    first_ratio, _ = load_ratio()
    for _ in range(40):
        moe.route(x_flat)
        moe.update_bias()
    final_ratio, counts = load_ratio()
    target = 64 * 2 / 8
    assert final_ratio < first_ratio
    assert counts.max().item() <= target * 1.5
    assert counts.min().item() >= target * 0.5


def test_update_bias_keeps_mean_centered():
    moe = _make_moe()
    moe.train()
    moe.route(torch.randn(10, 16))
    moe.update_bias()
    assert abs(moe.expert_bias.mean().item()) < 1e-6


def test_gradients_flow_to_all_trainable_params():
    moe = _make_moe()
    moe.train()
    x = torch.randn(2, 3, 16, requires_grad=True)
    out, _ = moe(x)
    out.sum().backward()
    for name, p in moe.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_backward_does_not_touch_expert_bias():
    moe = _make_moe()
    moe.train()
    bias_before = moe.expert_bias.clone()
    out, _ = moe(torch.randn(2, 3, 16))
    out.sum().backward()
    assert torch.equal(moe.expert_bias, bias_before)


def test_rmsnorm_accepts_explicit_dim():
    cfg = _make_cfg()
    norm = RMSNorm(cfg, dim=8)
    x = torch.randn(3, 5, 8)
    out = norm(x)
    assert out.shape == x.shape
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_situ_glu_is_bounded():
    moe = _make_moe()
    gate = torch.tensor([1000.0, -1000.0, 0.0])
    up = torch.tensor([1000.0, -1000.0, 0.0])
    out = moe.situ_glu(gate, up)
    bound = moe.situ_beta_gate * moe.situ_beta_up
    assert out.abs().max().item() <= bound


def test_capacity_drops_overflow_and_reports_it():
    moe = _make_moe(num_experts=4, num_experts_per_token=2)
    moe.train()
    with torch.inference_mode():
        moe.router.weight.zero_()
        moe.router.weight[0].fill_(10.0)
    x = torch.randn(2, 16, 16)
    out, metrics = moe(x)
    assert out.shape == (2, 16, 16) and torch.isfinite(out).all()
    assert metrics["dropped_frac"].item() > 0.0
