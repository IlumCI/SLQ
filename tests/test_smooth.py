"""Tests for SmoothQuant difficulty migration and its SLQ integration."""

import pytest
import torch
from torch import nn

from slq.model.reference import ReferenceConfig, ReferenceTransformer
from slq.quant.act import ActQuantConfig, quantize_activation
from slq.smooth import (
    collect_act_scales,
    compute_smoothing_scale,
    discover_sites,
    smooth_model,
)


@pytest.fixture
def model_and_input():
    torch.manual_seed(0)
    m = ReferenceTransformer(ReferenceConfig(num_layers=2))
    return m, torch.randint(0, 512, (2, 16))


def test_discovers_qkv_and_gate_up_fusion_sites(model_and_input):
    """Traced discovery should find what upstream SmoothQuant hardcodes."""
    model, x = model_and_input
    sites = discover_sites(model, x)
    by_norm = {s.norm: set(s.linears) for s in sites}

    assert "layers.0.input_layernorm" in by_norm
    assert by_norm["layers.0.input_layernorm"] == {
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
    }
    assert by_norm["layers.0.post_attention_layernorm"] == {
        "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj",
    }


def test_smoothing_preserves_model_output(model_and_input):
    """X diag(s)^-1 . diag(s) W is mathematically an identity."""
    model, x = model_and_input
    before = model(x).clone()
    scales = collect_act_scales(model, [x])
    sites = discover_sites(model, x)
    smooth_model(model, scales, sites, alpha=0.5)
    after = model(x)
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-3)


def test_smoothing_is_output_preserving_across_alphas(model_and_input):
    model, x = model_and_input
    before = model(x).clone()
    scales = collect_act_scales(model, [x])
    sites = discover_sites(model, x)
    smooth_model(model, scales, sites, alpha=0.85)
    assert torch.allclose(before, model(x), atol=1e-4, rtol=1e-3)


def test_alpha_zero_is_a_no_op(model_and_input):
    model, x = model_and_input
    w = model.layers[0].self_attn.q_proj.weight.data.clone()
    scales = collect_act_scales(model, [x])
    smooth_model(model, scales, discover_sites(model, x), alpha=0.0)
    assert torch.allclose(w, model.layers[0].self_attn.q_proj.weight.data)


def test_smoothing_scale_matches_reference_formula():
    """s_j = max|X_j|^alpha / max|W_j|^(1-alpha)."""
    torch.manual_seed(0)
    act = torch.rand(32) + 0.5
    w = [torch.randn(16, 32)]
    for alpha in (0.25, 0.5, 0.85):
        got = compute_smoothing_scale(act, w, alpha=alpha)
        want = act.pow(alpha) / w[0].abs().amax(dim=0).pow(1 - alpha)
        assert torch.allclose(got, want, rtol=1e-5)


def test_smoothing_scale_rejects_bad_alpha():
    with pytest.raises(ValueError, match="alpha"):
        compute_smoothing_scale(torch.ones(4), [torch.ones(2, 4)], alpha=1.5)


def test_smoothing_scale_rejects_channel_mismatch():
    with pytest.raises(ValueError, match="channels"):
        compute_smoothing_scale(torch.ones(8), [torch.ones(2, 4)])


def test_report_records_gamma_change(model_and_input):
    model, x = model_and_input
    scales = collect_act_scales(model, [x])
    sites = discover_sites(model, x)
    report = smooth_model(model, scales, sites, alpha=0.5)
    assert set(report.gamma_before) == set(report.gamma_after)
    assert all(g > 0 for g in report.gamma_before.values())
    assert report.summary()["sites"] == len(sites)


def test_act_scales_are_per_input_channel(model_and_input):
    model, x = model_and_input
    scales = collect_act_scales(model, [x])
    assert scales["layers.0.self_attn.q_proj"].shape == (model.config.hidden_size,)
    assert (scales["layers.0.self_attn.q_proj"] >= 0).all()


def test_asymmetric_activation_quant_beats_symmetric_on_offset_data():
    """The gamma-squared argument applied to the activation path."""
    torch.manual_seed(0)
    x = nn.functional.silu(torch.randn(128, 256) * 2)  # bounded below, long right tail
    err = {}
    for sym in (True, False):
        q = quantize_activation(x, ActQuantConfig(8, "per_token", sym))
        err[sym] = float((x - q).pow(2).mean())
    assert err[False] < err[True]


def test_activation_quant_disabled_is_identity():
    x = torch.randn(4, 8)
    assert torch.equal(quantize_activation(x, ActQuantConfig(bits=None)), x)
    assert torch.equal(quantize_activation(x, ActQuantConfig(bits=16)), x)


def test_activation_quant_granularities_run():
    torch.manual_seed(0)
    x = torch.randn(4, 6, 32)
    for g in ("per_token", "per_tensor", "per_channel"):
        out = quantize_activation(x, ActQuantConfig(8, g, False))
        assert out.shape == x.shape
        assert float((x - out).abs().max()) < 1.0
