"""Tests for the EAR / KL fidelity metrics (Section 3.2)."""

import pytest
import torch

from slq.metrics.fidelity import FidelityMeter, decision_flip_rates, fidelity


def test_identical_distributions_give_perfect_ear():
    """EAR = 1 - d_TV, so EAR(p, p) must be exactly 1."""
    torch.manual_seed(0)
    logits = torch.randn(64, 200)
    r = fidelity(logits, logits)
    assert r.ear == pytest.approx(1.0, abs=1e-6)
    assert r.kl == pytest.approx(0.0, abs=1e-9)
    assert r.flip_rate == 0.0


def test_ear_equals_one_minus_total_variation():
    """The defining identity of EAR, on the restricted support."""
    torch.manual_seed(0)
    k = 10
    p = torch.softmax(torch.randn(32, 50), dim=-1)
    q = torch.softmax(torch.randn(32, 50), dim=-1)
    r = fidelity(p, q, topk=k, is_logits=False)

    p_top, idx = torch.topk(p, k, dim=-1)
    q_top = torch.gather(q, 1, idx)
    p_n = p_top / p_top.sum(-1, keepdim=True)
    q_n = q_top / q_top.sum(-1, keepdim=True)
    d_tv = 0.5 * (p_n - q_n).abs().sum(-1)
    assert r.ear == pytest.approx(float((1 - d_tv).mean()), abs=1e-5)


def test_ear_is_bounded_in_unit_interval():
    torch.manual_seed(0)
    for scale in (0.1, 1.0, 10.0):
        r = fidelity(torch.randn(32, 100), torch.randn(32, 100) * scale)
        assert 0.0 <= r.ear <= 1.0


def test_kl_is_non_negative_when_normalized():
    torch.manual_seed(0)
    for _ in range(20):
        r = fidelity(torch.randn(16, 80), torch.randn(16, 80))
        assert r.kl >= -1e-9


def test_ear_decreases_as_perturbation_grows():
    torch.manual_seed(0)
    ref = torch.randn(128, 200) * 2
    prev = 1.0
    for noise in (0.01, 0.1, 0.5, 1.0):
        ear = fidelity(ref, ref + torch.randn_like(ref) * noise).ear
        assert ear <= prev + 1e-6
        prev = ear


def test_unnormalized_ear_is_capped_by_topk_mass():
    """Without conditioning, EAR cannot exceed the reference top-K mass."""
    torch.manual_seed(0)
    ref = torch.randn(64, 500)
    r = fidelity(ref, ref, normalize=False)
    assert r.ear == pytest.approx(r.topk_mass, abs=1e-6)
    assert r.ear < 1.0


def test_topk_mass_reported_under_both_modes():
    torch.manual_seed(0)
    ref, cand = torch.randn(32, 300), torch.randn(32, 300)
    assert fidelity(ref, cand).topk_mass == pytest.approx(
        fidelity(ref, cand, normalize=False).topk_mass
    )


def test_meter_matches_single_shot():
    torch.manual_seed(0)
    ref = [torch.randn(16, 120) for _ in range(4)]
    cand = [r + torch.randn_like(r) * 0.2 for r in ref]
    meter = FidelityMeter()
    for r, c in zip(ref, cand, strict=True):
        meter.update(r, c)
    batched = meter.compute()
    one_shot = fidelity(torch.cat(ref), torch.cat(cand))
    assert batched.ear == pytest.approx(one_shot.ear, abs=1e-6)
    assert batched.kl == pytest.approx(one_shot.kl, abs=1e-6)
    assert batched.positions == one_shot.positions


def test_flip_rate_counts_argmax_changes():
    p = torch.tensor([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    q = torch.tensor([[1.0, 5.0, 0.0], [0.0, 5.0, 1.0]])
    assert fidelity(p, q, topk=3).flip_rate == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        fidelity(torch.randn(4, 10), torch.randn(4, 11))


def test_decision_flip_rates_partition_all_positions():
    torch.manual_seed(0)
    ref = torch.randn(200, 50)
    out = decision_flip_rates(ref, ref + torch.randn_like(ref) * 0.5, n_bins=4)
    assert sum(out["count"]) == 200
    assert len(out["flip_rate"]) == 4
