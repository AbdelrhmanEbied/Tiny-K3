import torch

from configs.model_config import ModelConfig
from model.attn_residuals import AttentionResidual


def _make_cfg(**kwargs) -> ModelConfig:
    return ModelConfig(hidden_size=16, **kwargs)


def _make_res(**kwargs) -> AttentionResidual:
    torch.manual_seed(0)
    return AttentionResidual(_make_cfg(**kwargs))


def test_output_shape():
    res = _make_res()
    emb = torch.randn(2, 5, 16)
    out = res([emb], None)
    assert out.shape == (2, 5, 16)


def test_zero_query_gives_uniform_average():
    res = _make_res()
    a = torch.randn(2, 3, 16)
    b = torch.randn(2, 3, 16)
    c = torch.randn(2, 3, 16)
    out = res([a, b], c)
    expected = (a + b + c) / 3
    assert torch.allclose(out, expected, atol=1e-6)


def test_weights_are_token_dependent():
    res = _make_res()
    with torch.inference_mode():
        res.pseudo_query.copy_(torch.randn(16))
    v1 = torch.randn(1, 4, 16)
    v2 = torch.randn(1, 4, 16)
    out = res([v1, v2], None)

    k1 = res.norm(v1)
    k2 = res.norm(v2)
    s1 = (res.pseudo_query * k1).sum(-1)
    s2 = (res.pseudo_query * k2).sum(-1)
    alpha1 = torch.softmax(torch.stack([s1, s2]), dim=0)[0]
    expected = alpha1.unsqueeze(-1) * v1 + (1 - alpha1).unsqueeze(-1) * v2
    assert torch.allclose(out, expected, atol=1e-6)
    assert alpha1.std() > 0


def test_rmsnorm_prevents_magnitude_dominance():
    res = _make_res()
    with torch.inference_mode():
        res.pseudo_query.fill_(1.0)
    v = torch.randn(1, 2, 16)

    def alpha_of(source):
        keys = torch.stack([res.norm(s) for s in [source, v]])
        scores = torch.einsum("d,sbtd->sbt", res.pseudo_query, keys)
        return scores.softmax(dim=0)[0]

    alpha_normal = alpha_of(v)
    alpha_huge = alpha_of(v * 100.0)
    alpha_tiny = alpha_of(v * 0.01)
    assert torch.allclose(alpha_normal, alpha_huge, atol=1e-4)
    assert torch.allclose(alpha_normal, alpha_tiny, atol=1e-2)


def test_softmax_sums_to_one_per_token():
    res = _make_res()
    with torch.inference_mode():
        res.pseudo_query.copy_(torch.randn(16))
    sources = [torch.randn(2, 3, 16) for _ in range(5)]
    k = [res.norm(v) for v in sources]
    scores = torch.stack([(res.pseudo_query * kk).sum(-1) for kk in k])
    alphas = scores.softmax(dim=0)
    assert torch.allclose(alphas.sum(dim=0), torch.ones(2, 3), atol=1e-6)


def test_partial_block_changes_result():
    res = _make_res()
    with torch.inference_mode():
        res.pseudo_query.copy_(torch.randn(16))
    blocks = [torch.randn(1, 2, 16)]
    partial_a = torch.randn(1, 2, 16)
    partial_b = torch.randn(1, 2, 16)
    out_a = res(blocks, partial_a)
    out_b = res(blocks, partial_b)
    assert not torch.allclose(out_a, out_b)


def test_gradient_flows_to_pseudo_query_and_inputs():
    res = _make_res()
    a = torch.randn(1, 2, 16, requires_grad=True)
    b = torch.randn(1, 2, 16, requires_grad=True)
    out = res([a], b)
    out.sum().backward()
    assert (
        res.pseudo_query.grad is not None
        and torch.isfinite(res.pseudo_query.grad).all()
    )
    assert a.grad is not None and b.grad is not None


def test_more_sources_than_layers_still_valid():
    res = _make_res()
    sources = [torch.randn(1, 1, 16) for _ in range(12)]
    out = res(sources, None)
    assert out.shape == (1, 1, 16) and torch.isfinite(out).all()
