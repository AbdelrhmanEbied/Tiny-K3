import torch

from configs.model_config import ModelConfig
from model.rms_norm import RMSNorm


def _make_cfg(**kwargs):
    return ModelConfig(hidden_size=12, **kwargs)


def test_rmsnorm_output_shape():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    x = torch.randn(2, 4, 12)
    output = model(x)
    assert output.shape == x.shape


def test_rmsnorm_output_finite():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    x = torch.randn(2, 4, 12)
    output = model(x)
    assert torch.isfinite(output).all()


def test_rmsnorm_unit_rms():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    model.eval()
    x = torch.randn(8, 16, 12)
    output = model(x)
    rms = output.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_weight_ones_init():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    assert model.weight.shape == (12,)
    assert torch.allclose(model.weight.data, torch.ones(12))


def test_rmsnorm_gradient_flow():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    x = torch.randn(2, 4, 12, requires_grad=True)
    output = model(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert model.weight.grad is not None


def test_rmsnorm_custom_eps():
    cfg = _make_cfg(rms_norm_eps=1e-3)
    model = RMSNorm(cfg)
    assert model.eps == 1e-3
    x = torch.randn(2, 4, 12)
    output = model(x)
    assert torch.isfinite(output).all()


def test_rmsnorm_batch_independence():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    x = torch.randn(4, 8, 12)
    output = model(x)
    rms = output.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


def test_rmsnorm_single_token():
    cfg = _make_cfg()
    model = RMSNorm(cfg)
    x = torch.randn(1, 1, 12)
    output = model(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
