"""Tests for the ILP allocator and the fidelity-targeted searches."""

import pytest

from slq.alloc.evolution import EvolutionConfig, evolutionary_search
from slq.alloc.ilp import solve_allocation
from slq.alloc.search import search_distribution_lossless, search_task_lossless
from slq.sensitivity.database import SensitivityDatabase

BITS = [2, 3, 4, 5, 6, 7, 8]


def make_db(n_groups=6, sizes=None, scale=0.002):
    """A database whose cost falls geometrically with bitwidth."""
    groups = [f"g{i}" for i in range(n_groups)]
    numel = sizes or {g: 1000 * (i + 1) for i, g in enumerate(groups)}
    db = SensitivityDatabase.empty(groups, BITS, numel, method="shapley")
    for i, g in enumerate(groups):
        for b in BITS:
            cost = scale * (i + 1) * 2.0 ** (-(b - 2) / 2)
            db.kl[g][b] = cost
            db.ear_drop[g][b] = cost
    for g in groups:
        db.kl[g][8] = 0.0
        db.ear_drop[g][8] = 0.0
    return db


def eff(b):
    """Table 7 INT asymmetric overhead at group size 128."""
    return b + 20 / 128


# --------------------------------------------------------------------- ILP --


def test_allocation_respects_budget():
    db = make_db()
    for budget in (3.0, 4.5, 6.0):
        r = solve_allocation(
            db.groups, BITS, db.cost_table("ear"), db.numel, budget, effective_bits_fn=eff
        )
        assert r.average_bits <= budget + 1e-6
        assert r.feasible


def test_every_group_gets_exactly_one_bitwidth():
    db = make_db()
    r = solve_allocation(db.groups, BITS, db.cost_table("ear"), db.numel, 5.0)
    assert set(r.assignment) == set(db.groups)
    assert all(b in BITS for b in r.assignment.values())


def test_higher_budget_never_costs_more():
    db = make_db()
    prev = float("inf")
    for budget in (3.0, 4.0, 5.0, 6.0, 7.0):
        r = solve_allocation(db.groups, BITS, db.cost_table("ear"), db.numel, budget)
        assert r.predicted_cost <= prev + 1e-12
        prev = r.predicted_cost


def test_budget_constraint_is_parameter_weighted():
    """A large insensitive group must not be charged the same as a small one."""
    groups = ["big", "small"]
    numel = {"big": 100_000, "small": 1_000}
    db = SensitivityDatabase.empty(groups, BITS, numel, method="shapley")
    for b in BITS:
        db.ear_drop["big"][b] = 0.0001 * (8 - b)  # insensitive
        db.ear_drop["small"][b] = 0.05 * (8 - b)  # very sensitive
    r = solve_allocation(groups, BITS, db.cost_table("ear"), numel, 4.0)
    # The tiny sensitive group should get top precision; it is nearly free.
    assert r.assignment["small"] == 8
    assert r.assignment["big"] <= 4


def test_infeasible_budget_reports_floor():
    db = make_db()
    r = solve_allocation(
        db.groups, BITS, db.cost_table("ear"), db.numel, 1.0, effective_bits_fn=eff
    )
    assert not r.feasible
    assert set(r.assignment.values()) == {2}


def test_solvers_agree_closely():
    db = make_db()
    a = solve_allocation(db.groups, BITS, db.cost_table("ear"), db.numel, 5.0, solver="milp")
    b = solve_allocation(
        db.groups, BITS, db.cost_table("ear"), db.numel, 5.0, solver="lagrangian"
    )
    assert a.predicted_cost <= b.predicted_cost * 1.05 + 1e-12


# ------------------------------------------------------------------ search --


def test_dl_search_meets_target_and_minimizes_bits():
    db = make_db()
    loose = search_distribution_lossless(db, target_ear=0.97, effective_bits_fn=eff)
    tight = search_distribution_lossless(db, target_ear=0.995, effective_bits_fn=eff)
    assert loose.satisfied and tight.satisfied
    assert loose.average_bits <= tight.average_bits


def test_dl_search_reports_unreachable_target():
    db = make_db(scale=0.5)  # so degrading is catastrophic
    for g in db.groups:
        for b in BITS:
            db.ear_drop[g][b] = 0.5 if b < 8 else 0.4
    r = search_distribution_lossless(db, target_ear=0.99)
    assert not r.satisfied
    assert "unreachable" in r.notes.get("reason", "")


def test_dl_search_uses_measured_ear_when_available():
    db = make_db()
    r = search_distribution_lossless(
        db, target_ear=0.98, effective_bits_fn=eff, measure=lambda a: (0.5, 0.2)
    )
    assert r.measured_ear == 0.5
    assert not r.satisfied  # measurement overrides the optimistic prediction


def test_tl_search_recovers_known_threshold():
    """With a synthetic linear recovery model, the search must honour it."""
    db = make_db()
    alpha = 20.0  # recovery = 1 - alpha * KL

    def measure_kl(assignment):
        return db.predict_kl(assignment)

    def benchmark(assignment):
        if assignment is None:
            return 1.0
        return 1.0 - alpha * db.predict_kl(assignment)

    r = search_task_lossless(
        db,
        measure_kl=measure_kl,
        benchmark=benchmark,
        target_recovery=0.99,
        calibration_bits=4.0,
        effective_bits_fn=eff,
    )
    assert r.satisfied
    assert r.notes["alpha"] == pytest.approx(alpha, rel=1e-6)
    assert r.notes["kl_threshold"] == pytest.approx(0.01 / alpha, rel=1e-6)
    assert 1.0 - alpha * r.measured_kl >= 0.99 - 1e-6


def test_tl_search_rejects_degenerate_anchor():
    db = make_db()
    with pytest.raises(ValueError, match="measurable degradation"):
        search_task_lossless(
            db, measure_kl=lambda a: 0.0, benchmark=lambda a: 1.0, calibration_bits=4.0
        )


def test_tl_search_handles_anchor_beating_baseline():
    db = make_db()
    r = search_task_lossless(
        db,
        measure_kl=lambda a: db.predict_kl(a),
        benchmark=lambda a: 1.0 if a is None else 1.05,
        calibration_bits=4.0,
    )
    assert r.satisfied
    assert "not identifiable" in r.notes["reason"]


# ------------------------------------------------------------- evolutionary --


def test_evolutionary_search_reduces_bits_under_constraint():
    db = make_db()
    tau = 0.01

    def evaluate(assignment, _max_batches):
        return db.predict_kl(assignment)

    r = evolutionary_search(
        db.groups,
        BITS,
        db.numel,
        evaluate,
        EvolutionConfig(offspring=8, threshold=tau, max_generations=25, stall_limit=3),
        effective_bits_fn=eff,
    )
    assert r.kl <= tau
    assert r.average_bits < eff(8)


def test_evolutionary_search_respects_a_strict_constraint():
    db = make_db()

    def evaluate(assignment, _max_batches):
        return db.predict_kl(assignment)

    r = evolutionary_search(
        db.groups,
        BITS,
        db.numel,
        evaluate,
        EvolutionConfig(offspring=6, threshold=1e-9, max_generations=10),
        effective_bits_fn=eff,
    )
    assert r.assignment == dict.fromkeys(db.groups, 8)


# ---------------------------------------------------------------- database --


def test_monotonicity_is_enforced():
    db = make_db()
    db.ear_drop["g0"][6] = 99.0  # noise: worse at 6 bits than at 4
    db.enforce_monotonic()
    for b_lo, b_hi in zip(BITS, BITS[1:], strict=False):
        assert db.ear_drop["g0"][b_lo] >= db.ear_drop["g0"][b_hi] - 1e-12


def test_monotonicity_does_not_flatten_the_database():
    """Regression: a running minimum would propagate the zero cost at b_max.

    Costs are pinned to zero at the reference bitwidth, so enforcing
    monotonicity in the wrong direction silently zeroes every entry -- the
    database still looks well-formed, but every configuration then predicts
    identical fidelity and the search collapses to the minimum bitwidth.
    """
    db = make_db()
    db.ear_drop["g0"][3] = 0.0001  # noise: 3 bits looks cheaper than 4
    before_2bit = db.ear_drop["g0"][2]
    db.enforce_monotonic()
    assert db.ear_drop["g0"][2] == pytest.approx(before_2bit)
    assert db.ear_drop["g0"][3] >= db.ear_drop["g0"][4]
    assert any(db.ear_drop[g][2] > 0 for g in db.groups)


def test_search_discriminates_across_targets_after_monotonicity():
    """A flattened database would return the same bitwidth for every target."""
    db = make_db()
    db.enforce_monotonic()
    bits = [
        search_distribution_lossless(db, target_ear=t, effective_bits_fn=eff).average_bits
        for t in (0.95, 0.99, 0.999)
    ]
    assert bits[0] < bits[-1], f"search did not discriminate: {bits}"


def test_prediction_anchors_on_baseline():
    db = make_db()
    db.baseline_ear = 0.75
    assert db.predict_ear(dict.fromkeys(db.groups, 8)) == pytest.approx(0.75)


def test_database_roundtrips_through_json(tmp_path):
    db = make_db()
    db.baseline_ear, db.baseline_kl = 0.97, 0.001
    path = tmp_path / "db.json"
    db.save(str(path))
    back = SensitivityDatabase.load(str(path))
    assert back.groups == db.groups
    assert back.baseline_ear == pytest.approx(0.97)
    a = dict.fromkeys(db.groups, 4)
    assert back.predict_ear(a) == pytest.approx(db.predict_ear(a))


# ---------------------------------------------------------- memory budget --


def test_memory_budget_fits():
    from slq.alloc import search_memory_budget

    db = make_db(n_groups=8, sizes={f"g{i}": 1_000_000 for i in range(8)})
    total = sum(db.numel.values())
    budget = total * 5.0 / 8  # 5 bits per parameter
    r = search_memory_budget(db, budget, effective_bits_fn=eff)
    assert r.satisfied
    assert r.notes["realized_bytes"] <= budget + 1e-6


def test_memory_budget_reserves_overhead():
    from slq.alloc import search_memory_budget

    db = make_db(n_groups=8, sizes={f"g{i}": 1_000_000 for i in range(8)})
    total = sum(db.numel.values())
    budget = total * 6.0 / 8
    full = search_memory_budget(db, budget, effective_bits_fn=eff, overhead_fraction=0.0)
    reserved = search_memory_budget(db, budget, effective_bits_fn=eff, overhead_fraction=0.25)
    assert reserved.average_bits < full.average_bits
    assert reserved.notes["realized_bytes"] <= budget * 0.75 + 1e-6


def test_memory_budget_too_small_explains_the_minimum():
    from slq.alloc import search_memory_budget

    db = make_db(n_groups=4, sizes={f"g{i}": 1_000_000 for i in range(4)})
    with pytest.raises(ValueError, match="below the smallest available bitwidth"):
        search_memory_budget(db, 100.0, effective_bits_fn=eff)


def test_memory_budget_spends_more_bits_when_given_more():
    from slq.alloc import search_memory_budget

    db = make_db(n_groups=8, sizes={f"g{i}": 1_000_000 for i in range(8)})
    total = sum(db.numel.values())
    prev = 0.0
    for bpp in (3.0, 4.0, 5.0, 6.0):
        r = search_memory_budget(db, total * bpp / 8, effective_bits_fn=eff)
        assert r.average_bits > prev
        prev = r.average_bits
