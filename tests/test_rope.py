import pytest
import torch

from configs.ModelConfig import ModelConfig
from model.RoPE import RoPE


def _make_cfg(**kwargs):
    defaults = dict(qk_rope_dim=16, max_seq_len=64, original_max_seq_len=32)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def test_rope_output_shape_3d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 16)
    y = model(x)
    assert y.shape == x.shape


def test_rope_output_shape_4d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 8, 16)
    y = model(x)
    assert y.shape == x.shape


def test_rope_output_finite():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 16)
    y = model(x)
    assert torch.isfinite(y).all()


def test_rope_preserves_norm():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(2, 8, 16)
    y = model(x)
    x_norm = x.float().norm(dim=-1)
    y_norm = y.float().norm(dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-5)


def test_rope_preserves_norm_4d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(2, 8, 4, 16)
    y = model(x)
    x_norm = x.float().norm(dim=-1)
    y_norm = y.float().norm(dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-5)


def test_rope_gradient_flow():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_rope_gradient_flow_4d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 2, 16, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None


def test_rope_odd_dim_raises():
    cfg = _make_cfg(qk_rope_dim=15)
    with pytest.raises(ValueError, match="must be even"):
        RoPE(cfg)


def test_rope_seq_len_exceeds_max_raises():
    cfg = _make_cfg(max_seq_len=8)
    model = RoPE(cfg)
    x = torch.randn(1, 16, 16)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(x)


def test_rope_invalid_input_dim_raises():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="shape \\[B, T, D\\] or \\[B, T, H, D\\]"):
        model(x)


def test_rope_odd_feature_dim_raises():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 7)
    with pytest.raises(ValueError, match="even"):
        model(x)


def test_rope_invalid_position_ids_shape_raises():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(2, 4, 16)
    pos_ids = torch.arange(8).reshape(2, 4)
    x2 = torch.randn(3, 4, 16)
    with pytest.raises(ValueError, match="position_ids shape must match"):
        model(x2, position_ids=pos_ids)


def test_rope_invalid_position_ids_dim_raises():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(1, 4, 16)
    pos_ids = torch.arange(4).reshape(1, 1, 4)
    with pytest.raises(ValueError, match="position_ids must have shape"):
        model(x, position_ids=pos_ids)


def test_rope_single_token():
    cfg = _make_cfg()
    model = RoPE(cfg)
    x = torch.randn(1, 1, 16)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_custom_position_ids_1d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(1, 4, 16)
    pos_ids = torch.tensor([10, 20, 30, 40])
    y = model(x, position_ids=pos_ids)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_custom_position_ids_2d():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(2, 4, 16)
    pos_ids = torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]])
    y = model(x, position_ids=pos_ids)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_position_shift_changes_output():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(1, 4, 16)
    y0 = model(x, position_ids=torch.tensor([0, 1, 2, 3]))
    y1 = model(x, position_ids=torch.tensor([1, 2, 3, 4]))
    assert not torch.allclose(y0, y1)


def test_rope_batch_independence():
    cfg = _make_cfg()
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(4, 8, 16)
    y = model(x)
    for i in range(4):
        yi = model(x[i : i + 1])
        assert torch.allclose(y[i : i + 1], yi, atol=1e-6)


def test_rope_inv_freq_shape():
    cfg = _make_cfg(qk_rope_dim=16)
    model = RoPE(cfg)
    assert model.inv_freq.shape == (8,)


def test_rope_inv_freq_values():
    cfg = _make_cfg(qk_rope_dim=16, rope_theta=10000.0)
    model = RoPE(cfg)
    expected = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    assert torch.allclose(model.inv_freq, expected, atol=1e-7)


def test_rope_freqs_cis_shape():
    cfg = _make_cfg(qk_rope_dim=16, max_seq_len=32)
    model = RoPE(cfg)
    assert model.freqs_cis.shape == (32, 8)


def test_rope_freqs_cis_unit_magnitude():
    cfg = _make_cfg()
    model = RoPE(cfg)
    magnitudes = model.freqs_cis.abs()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)


def test_rope_yarn_with_factor_1_is_identity_on_inv_freq():
    cfg = _make_cfg(rope_type="yarn", factor=1.0)
    model = RoPE(cfg)
    expected = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.qk_rope_dim, 2, dtype=torch.float32) / cfg.qk_rope_dim))
    assert torch.allclose(model.inv_freq, expected, atol=1e-7)


def test_rope_yarn_with_factor_gt_1():
    cfg = _make_cfg(rope_type="yarn", factor=2.0)
    model = RoPE(cfg)
    x = torch.randn(2, 4, 16)
    y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_yarn_preserves_norm():
    cfg = _make_cfg(rope_type="yarn", factor=2.0)
    model = RoPE(cfg)
    model.eval()
    x = torch.randn(2, 8, 16)
    y = model(x)
    x_norm = x.float().norm(dim=-1)
    y_norm = y.float().norm(dim=-1)
    assert torch.allclose(x_norm, y_norm, atol=1e-5)


def test_rope_yarn_different_factors():
    cfg1 = _make_cfg(rope_type="yarn", factor=2.0)
    cfg2 = _make_cfg(rope_type="yarn", factor=4.0)
    m1 = RoPE(cfg1)
    m2 = RoPE(cfg2)
    assert not torch.allclose(m1.inv_freq, m2.inv_freq)


def test_rope_default_vs_yarn_different_inv_freq():
    cfg_d = _make_cfg(rope_type="default")
    cfg_y = _make_cfg(rope_type="yarn", factor=2.0)
    m_d = RoPE(cfg_d)
    m_y = RoPE(cfg_y)
    assert not torch.allclose(m_d.inv_freq, m_y.inv_freq)
