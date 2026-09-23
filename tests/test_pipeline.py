"""Integration tests: grouping, the model wrapper, and the end-to-end run."""

import pytest
import torch

from slq.model.bank import WeightBank
from slq.model.grouping import GroupingPolicy, average_bitwidth, build_groups
from slq.model.reference import ReferenceConfig, ReferenceTransformer
from slq.model.wrapper import QuantizableModel
from slq.sensitivity.linear import linear_sensitivity
from slq.sensitivity.shapley import shapley_sensitivity

BITS = (2, 4, 8)


@pytest.fixture
def qmodel():
    torch.manual_seed(0)
    model = ReferenceTransformer(ReferenceConfig(num_layers=2, vocab_size=128))
    calib = [torch.randint(0, 128, (2, 16)) for _ in range(3)]
    qm = QuantizableModel(model, calib, bank=WeightBank(bitwidths=BITS, method="rtn"))
    qm.build_bank()
    return qm


# ------------------------------------------------------------------ grouping --


def test_layer_granularity_gives_one_group_per_layer():
    layers = [(f"layers.{i}.self_attn.q_proj", 100) for i in range(3)]
    assert len(build_groups(layers, GroupingPolicy(granularity="layer"))) == 3


def test_block_granularity_merges_within_a_block():
    layers = [
        ("layers.0.self_attn.q_proj", 100),
        ("layers.0.mlp.gate_proj", 200),
        ("layers.1.self_attn.q_proj", 100),
    ]
    groups = build_groups(layers, GroupingPolicy(granularity="block"))
    assert len(groups) == 2
    assert next(g for g in groups if g.name == "block.0").numel == 300


def test_module_type_granularity_merges_across_blocks():
    layers = [(f"layers.{i}.self_attn.q_proj", 100) for i in range(4)]
    groups = build_groups(layers, GroupingPolicy(granularity="module_type"))
    assert len(groups) == 1 and groups[0].numel == 400


def test_qkv_fusion_forces_a_shared_group():
    layers = [
        ("layers.0.self_attn.q_proj", 10),
        ("layers.0.self_attn.k_proj", 10),
        ("layers.0.self_attn.v_proj", 10),
        ("layers.0.self_attn.o_proj", 10),
    ]
    groups = build_groups(layers, GroupingPolicy(granularity="layer", fuse_qkv=True))
    assert len(groups) == 2
    assert max(len(g) for g in groups) == 3


def test_excluded_layers_are_dropped():
    layers = [("lm_head", 100), ("embed_tokens", 100), ("layers.0.mlp.up_proj", 50)]
    groups = build_groups(layers)
    assert len(groups) == 1 and groups[0].numel == 50


def test_average_bitwidth_is_parameter_weighted():
    groups = build_groups([("layers.0.a", 300), ("layers.1.b", 100)])
    got = average_bitwidth(groups, {g.name: b for g, b in zip(groups, (4, 8), strict=True)})
    assert got == pytest.approx((4 * 300 + 8 * 100) / 400)


def test_invalid_granularity_rejected():
    with pytest.raises(ValueError, match="granularity"):
        GroupingPolicy(granularity="nonsense")


# ------------------------------------------------------------------- wrapper --


def test_apply_then_restore_returns_original_weights(qmodel):
    original = qmodel.model.layers[0].mlp.up_proj.weight.data.clone()
    qmodel.apply(qmodel.uniform(2))
    assert not torch.allclose(original, qmodel.model.layers[0].mlp.up_proj.weight.data)
    qmodel.restore()
    assert torch.allclose(original, qmodel.model.layers[0].mlp.up_proj.weight.data)


def test_context_manager_restores(qmodel):
    original = qmodel.model.layers[0].mlp.up_proj.weight.data.clone()
    with qmodel as qm:
        qm.apply(qm.uniform(2))
    assert torch.allclose(original, qmodel.model.layers[0].mlp.up_proj.weight.data)


def test_reference_logits_are_full_precision_and_cached(qmodel):
    ref = qmodel.reference_logits()
    qmodel.apply(qmodel.uniform(2))
    assert qmodel.reference_logits() is ref  # cached, not recomputed while quantized


def test_fidelity_degrades_monotonically_with_bitwidth(qmodel):
    eight = qmodel.evaluate(qmodel.uniform(8))
    two = qmodel.evaluate(qmodel.uniform(2))
    assert eight.ear >= two.ear
    assert eight.kl <= two.kl


def test_full_precision_assignment_is_perfect(qmodel):
    r = qmodel.evaluate({g.name: None for g in qmodel.groups})
    assert r.ear == pytest.approx(1.0, abs=1e-5)
    assert r.kl == pytest.approx(0.0, abs=1e-7)


def test_average_bits_includes_overhead(qmodel):
    nominal = qmodel.average_bits(qmodel.uniform(4), effective=False)
    realized = qmodel.average_bits(qmodel.uniform(4), effective=True)
    assert nominal == pytest.approx(4.0)
    assert realized > nominal


def test_unknown_group_raises(qmodel):
    with pytest.raises(KeyError):
        qmodel.apply({"not-a-group": 4})


def test_requires_calibration():
    model = ReferenceTransformer(ReferenceConfig(num_layers=1))
    with pytest.raises(ValueError, match="calibration"):
        QuantizableModel(model, [])


# --------------------------------------------------------------- sensitivity --


def test_shapley_database_is_well_formed(qmodel):
    db = shapley_sensitivity(qmodel, bitwidths=BITS, permutations=2)
    assert set(db.groups) == set(qmodel.group_names)
    for g in db.groups:
        assert db.kl[g][8] == 0.0 and db.ear_drop[g][8] == 0.0
        for lo, hi in zip(BITS, BITS[1:], strict=False):
            assert db.ear_drop[g][lo] >= db.ear_drop[g][hi] - 1e-12


def test_shapley_prediction_tracks_measurement(qmodel):
    """Predicted EAR should be in the right neighbourhood of the measured one."""
    db = shapley_sensitivity(qmodel, bitwidths=BITS, permutations=4)
    for bits in (4, 2):
        a = qmodel.uniform(bits)
        assert db.predict_ear(a) == pytest.approx(qmodel.evaluate(a).ear, abs=0.05)


def test_shapley_baseline_is_recorded(qmodel):
    db = shapley_sensitivity(qmodel, bitwidths=BITS, permutations=1)
    assert db.baseline_ear == pytest.approx(qmodel.evaluate(qmodel.uniform(8)).ear, abs=1e-6)


def test_linear_sensitivity_runs_and_is_monotone(qmodel):
    db = linear_sensitivity(qmodel, bitwidths=BITS)
    assert db.method == "linear"
    for g in db.groups:
        assert db.ear_drop[g][2] >= db.ear_drop[g][8] - 1e-12


def test_sensitivity_leaves_model_at_full_precision(qmodel):
    original = qmodel.model.layers[0].mlp.up_proj.weight.data.clone()
    shapley_sensitivity(qmodel, bitwidths=BITS, permutations=1)
    assert torch.allclose(original, qmodel.model.layers[0].mlp.up_proj.weight.data)


# ------------------------------------------------------------------ pipeline --


def test_end_to_end_dl_run_produces_a_valid_assignment():
    from slq.pipeline import SLQConfig, run_slq

    torch.manual_seed(0)
    model = ReferenceTransformer(ReferenceConfig(num_layers=2, vocab_size=128))
    calib = [torch.randint(0, 128, (2, 16)) for _ in range(3)]
    res = run_slq(
        model,
        calib,
        SLQConfig(bitwidths=BITS, permutations=2, target="dl", target_ear=0.9,
                  quantizer="rtn"),
    )
    assert set(res.assignment) == set(res.model.group_names)
    assert all(b in BITS for b in res.assignment.values())
    assert 2.0 <= res.average_bits <= 8.5
    assert res.measured.positions > 0


def test_tl_run_without_benchmark_is_rejected():
    from slq.pipeline import SLQConfig, run_slq

    model = ReferenceTransformer(ReferenceConfig(num_layers=1, vocab_size=64))
    calib = [torch.randint(0, 64, (1, 8))]
    with pytest.raises(ValueError, match="benchmark"):
        run_slq(model, calib, SLQConfig(bitwidths=BITS, target="tl", quantizer="rtn"))


def test_smoothing_inside_the_pipeline_runs():
    from slq.pipeline import SLQConfig, run_slq

    torch.manual_seed(0)
    model = ReferenceTransformer(ReferenceConfig(num_layers=2, vocab_size=128))
    calib = [torch.randint(0, 128, (2, 16)) for _ in range(2)]
    res = run_slq(
        model,
        calib,
        SLQConfig(bitwidths=BITS, permutations=1, target="dl", target_ear=0.9,
                  quantizer="rtn", smooth=True),
    )
    assert res.smooth_report is not None
    assert res.smooth_report.summary()["sites"] > 0
