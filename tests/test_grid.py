"""Tests for quantization grid geometry and the gamma-squared law."""

import pytest
import torch

from slq.quant.grid import (
    QuantConfig,
    centering_inefficiency,
    effective_bits,
    fake_quantize,
    quantize,
    step_size,
    theoretical_noise_variance,
)


def test_effective_bits_reproduces_table_7():
    """Table 7: realized bits per parameter including scale/zero-point."""
    for b in (4, 5, 6, 7, 8):
        assert effective_bits(QuantConfig(b, 128, False, "int")) == pytest.approx(b + 0.15625)
        assert effective_bits(QuantConfig(b, 128, True, "int")) == pytest.approx(b + 0.125)
    assert effective_bits(QuantConfig(4, 16, False, "fp")) == pytest.approx(5.0)
    assert effective_bits(QuantConfig(4, 16, True, "fp")) == pytest.approx(4.5)
    assert effective_bits(QuantConfig(8, 16, False, "fp")) == pytest.approx(8.0)
    assert effective_bits(QuantConfig(8, 16, True, "fp")) == pytest.approx(8.0)


def test_gamma_is_one_for_symmetric_weights():
    """Each group must be symmetric about zero for gamma to be 1."""
    row = torch.linspace(-1, 1, 128)
    w = torch.stack([row, row])
    gamma = centering_inefficiency(w, 128)
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-5)


def test_gamma_matches_paper_example():
    """Weights in [-0.8, 1.2] give gamma = 2*1.2/2.0 = 1.2 (Definition 3.1)."""
    w = torch.linspace(-0.8, 1.2, 128).reshape(1, 128)
    assert float(centering_inefficiency(w, 128)[0]) == pytest.approx(1.2, abs=1e-4)


def test_step_size_ratio_is_gamma():
    """Lemma B.1: Delta_sym = gamma * Delta_asym, independent of n."""
    torch.manual_seed(0)
    w = torch.rand(4, 128) * 2.0 - 0.8
    gamma = centering_inefficiency(w, 128)
    for bits in (3, 4, 8):
        ratio = step_size(w, QuantConfig(bits, 128, True)) / step_size(
            w, QuantConfig(bits, 128, False)
        )
        assert torch.allclose(ratio, gamma, rtol=1e-4)


def test_noise_variance_scales_with_gamma_squared():
    """Lemma 3.2: sigma^2_sym = gamma^2 * sigma^2_asym."""
    torch.manual_seed(0)
    w = torch.rand(4, 128) * 2.0 - 0.8
    gamma = centering_inefficiency(w, 128)
    ratio = theoretical_noise_variance(w, QuantConfig(6, 128, True)) / (
        theoretical_noise_variance(w, QuantConfig(6, 128, False))
    )
    assert torch.allclose(ratio, gamma.pow(2), rtol=1e-4)


def test_empirical_error_ratio_tracks_gamma_squared():
    """Eq. 2: the measured MSE ratio should approach gamma^2 at high rate."""
    torch.manual_seed(0)
    w = torch.rand(64, 128) * 2.0 - 0.8  # offset from zero, so gamma > 1
    gamma_sq = float(centering_inefficiency(w, 128).mean()) ** 2
    cfg_a = QuantConfig(8, 128, symmetric=False)
    cfg_s = QuantConfig(8, 128, symmetric=True)
    mse_a = (w - fake_quantize(w, cfg_a)).pow(2).mean()
    mse_s = (w - fake_quantize(w, cfg_s)).pow(2).mean()
    assert float(mse_s / mse_a) == pytest.approx(gamma_sq, rel=0.12)


def test_quantize_roundtrip_is_bounded_by_step_size():
    torch.manual_seed(0)
    w = torch.randn(8, 128) * 0.1
    cfg = QuantConfig(5, 128)
    err = (w - fake_quantize(w, cfg)).abs().max()
    assert float(err) <= float(step_size(w, cfg).max()) / 2 + 1e-6


def test_codes_stay_in_range():
    torch.manual_seed(0)
    w = torch.randn(4, 128)
    for bits in (2, 4, 8):
        codes, _, _ = quantize(w, QuantConfig(bits, 128))
        assert codes.min() >= 0 and codes.max() <= (1 << bits) - 1


def test_asymmetric_beats_symmetric_on_offset_weights():
    torch.manual_seed(0)
    w = torch.rand(32, 128) + 0.5  # strictly positive: worst case for symmetric
    cfg_a = QuantConfig(4, 128, symmetric=False)
    cfg_s = QuantConfig(4, 128, symmetric=True)
    assert (w - fake_quantize(w, cfg_a)).pow(2).mean() < (
        w - fake_quantize(w, cfg_s)
    ).pow(2).mean()


def test_group_size_clamps_to_narrow_rows():
    w = torch.randn(8, 64)
    assert fake_quantize(w, QuantConfig(4, 128)).shape == w.shape


def test_indivisible_group_size_raises():
    w = torch.randn(4, 100)
    with pytest.raises(ValueError, match="not divisible"):
        fake_quantize(w, QuantConfig(4, 32))


def test_invalid_configs_rejected():
    with pytest.raises(ValueError):
        QuantConfig(bits=0)
    with pytest.raises(ValueError):
        QuantConfig(bits=4, fmt="bogus")
