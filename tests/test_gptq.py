"""Tests for the GPTQ per-layer quantizer."""

import pytest
import torch

from slq.model.bank import WeightBank
from slq.quant.gptq import GPTQQuantizer, gptq_quantize
from slq.quant.grid import QuantConfig, fake_quantize


@pytest.fixture
def layer():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.05 + 0.02  # offset, so gamma > 1
    x = torch.randn(512, 256)
    return w, x


def test_gptq_beats_round_to_nearest_on_output_error(layer):
    """GPTQ minimizes ||WX - W_hat X||, so it should beat RTN in that norm."""
    w, x = layer
    for bits in (2, 3, 4, 8):
        cfg = QuantConfig(bits, 128)
        err_gptq = ((w - gptq_quantize(w, x, cfg)) @ x.T).pow(2).mean()
        err_rtn = ((w - fake_quantize(w, cfg)) @ x.T).pow(2).mean()
        assert err_gptq < err_rtn, f"GPTQ did not improve on RTN at {bits} bits"


def test_output_error_decreases_with_bitwidth(layer):
    w, x = layer
    prev = float("inf")
    for bits in (2, 3, 4, 6, 8):
        err = float(((w - gptq_quantize(w, x, QuantConfig(bits, 128))) @ x.T).pow(2).mean())
        assert err < prev
        prev = err


def test_act_order_runs_and_stays_accurate(layer):
    w, x = layer
    cfg = QuantConfig(3, 128)
    plain = ((w - gptq_quantize(w, x, cfg)) @ x.T).pow(2).mean()
    ordered = ((w - gptq_quantize(w, x, cfg, act_order=True)) @ x.T).pow(2).mean()
    assert float(ordered) < float(plain) * 2.0


def test_prepare_is_cached_and_reused(layer):
    w, x = layer
    q = GPTQQuantizer(w)
    q.add_batch(x)
    first = q.prepare()
    assert q.prepare() is first  # same object, not recomputed
    assert q.prepare(act_order=True) is not first  # different key


def test_shared_cholesky_matches_independent_runs(layer):
    """Caching the factorization must not change the result."""
    w, x = layer
    shared = GPTQQuantizer(w)
    shared.add_batch(x)
    for bits in (4, 8):
        cfg = QuantConfig(bits, 128)
        fresh = GPTQQuantizer(w)
        fresh.add_batch(x)
        assert torch.allclose(shared.quantize(cfg), fresh.quantize(cfg), atol=1e-6)


def test_thread_count_is_restored(layer):
    w, x = layer
    before = torch.get_num_threads()
    gptq_quantize(w, x, QuantConfig(4, 128))
    assert torch.get_num_threads() == before


def test_add_batch_composes_across_batch_sizes():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    x = torch.randn(100, 32)
    one = GPTQQuantizer(w)
    one.add_batch(x)
    split = GPTQQuantizer(w)
    split.add_batch(x[:40])
    split.add_batch(x[40:])
    assert torch.allclose(one.hessian, split.hessian, rtol=1e-4, atol=1e-6)


def test_dead_columns_are_handled():
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    x = torch.randn(128, 64)
    x[:, 10] = 0.0  # a column the calibration never exercises
    out = gptq_quantize(w, x, QuantConfig(4, 64))
    assert torch.isfinite(out).all()


def test_wrong_input_width_raises():
    q = GPTQQuantizer(torch.randn(4, 16))
    with pytest.raises(ValueError, match="features"):
        q.add_batch(torch.randn(8, 20))


def test_non_2d_weight_raises():
    with pytest.raises(ValueError, match="2-D"):
        GPTQQuantizer(torch.randn(2, 3, 4))


# ------------------------------------------------------------------- bank --


def test_bank_requires_hessian_for_gptq():
    bank = WeightBank(bitwidths=(4,), method="gptq")
    with pytest.raises(ValueError, match="requires a Hessian"):
        bank.add_layer("l", torch.randn(8, 64))


def test_bank_rtn_needs_no_hessian():
    bank = WeightBank(bitwidths=(4, 8), method="rtn")
    bank.add_layer("l", torch.randn(8, 64))
    assert bank.has("l", 4) and bank.has("l", 8)


def test_bank_materializes_expected_shape_and_dtype():
    bank = WeightBank(bitwidths=(4,), method="rtn")
    w = torch.randn(8, 64)
    bank.add_layer("l", w)
    out = bank.get("l", 4).materialize("cpu", torch.float32)
    assert out.shape == w.shape and out.dtype == torch.float32


def test_bank_missing_entry_error_lists_available():
    bank = WeightBank(bitwidths=(4,), method="rtn")
    bank.add_layer("l", torch.randn(8, 64))
    with pytest.raises(KeyError, match="available bitwidths"):
        bank.get("l", 7)


def test_bank_roundtrips_through_disk(tmp_path):
    bank = WeightBank(bitwidths=(4, 8), method="rtn")
    w = torch.randn(8, 64)
    bank.add_layer("l", w)
    path = tmp_path / "bank.pt"
    bank.save(str(path))
    back = WeightBank.load(str(path))
    assert torch.allclose(
        bank.get("l", 4).materialize("cpu", torch.float32),
        back.get("l", 4).materialize("cpu", torch.float32),
    )
