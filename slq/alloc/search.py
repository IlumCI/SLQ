"""Fidelity-targeted bitwidth search (Section 3.3, Algorithm 2).

SLQ inverts the usual framing. Instead of fixing a bitwidth and reporting the
damage, it fixes a fidelity target and finds the minimum average bitwidth that
meets it. Two targets are supported:

* **Distribution-lossless (DL)** -- constrain EAR directly, e.g. ``EAR >= 0.99``.
  The sensitivity database predicts EAR for any candidate configuration, so the
  binary search over the bitwidth budget needs no model evaluations at all.

* **Task-lossless (TL)** -- constrain downstream benchmark recovery. This
  exploits the near-linear relationship between KL divergence and recovery
  (Eq. 4, ``recovery ~ 1 - alpha * D_KL``). Because the intercept is known
  (``D_KL = 0`` implies ``recovery = 1``), a *single* calibration point fixes
  the slope, and the search then runs entirely on arithmetic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from slq.alloc.ilp import AllocationResult, solve_allocation
from slq.sensitivity.database import SensitivityDatabase

__all__ = [
    "SearchResult",
    "search_distribution_lossless",
    "search_memory_budget",
    "search_task_lossless",
]

#: Measures actual KL on a concrete assignment. One forward pass.
MeasureKLFn = Callable[[dict[str, int]], float]

#: Returns a benchmark score for an assignment; ``None`` means the BF16 baseline.
BenchmarkFn = Callable[[dict[str, int] | None], float]


@dataclass
class SearchResult:
    """The configuration a search settled on, with its audit trail."""

    assignment: dict[str, int]
    average_bits: float
    target: str
    predicted_ear: float = float("nan")
    predicted_kl: float = float("nan")
    measured_ear: float = float("nan")
    measured_kl: float = float("nan")
    satisfied: bool = False
    trace: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "assignment": dict(self.assignment),
            "average_bits": self.average_bits,
            "target": self.target,
            "predicted_ear": self.predicted_ear,
            "predicted_kl": self.predicted_kl,
            "measured_ear": self.measured_ear,
            "measured_kl": self.measured_kl,
            "satisfied": self.satisfied,
            "trace": self.trace,
            "notes": self.notes,
        }


def _solve(
    db: SensitivityDatabase,
    budget: float,
    metric: str,
    effective_bits_fn: Callable[[int], float] | None,
    solver: str,
) -> AllocationResult:
    return solve_allocation(
        groups=db.groups,
        bitwidths=db.bitwidths,
        costs=db.cost_table(metric),
        numel=db.numel,
        budget_bits=budget,
        effective_bits_fn=effective_bits_fn,
        solver=solver,
    )


def search_distribution_lossless(
    db: SensitivityDatabase,
    target_ear: float = 0.99,
    effective_bits_fn: Callable[[int], float] | None = None,
    tolerance: float = 0.01,
    max_iters: int = 40,
    solver: str = "auto",
    measure: Callable[[dict[str, int]], tuple[float, float]] | None = None,
) -> SearchResult:
    """Find the minimum average bitwidth whose predicted EAR meets a target.

    Args:
        db: Sensitivity database from either estimator.
        target_ear: The DL constraint. The paper uses 0.985-0.99 by model.
        effective_bits_fn: Nominal-to-realized bitwidth map (Table 7).
        tolerance: Binary-search resolution in bits.
        max_iters: Cap on bisection steps.
        solver: Passed to the allocator.
        measure: Optional ``assignment -> (ear, kl)`` used once at the end to
            verify the prediction against the real model.

    Returns:
        A :class:`SearchResult`. ``satisfied`` reflects the measured EAR when
        ``measure`` is given, and the predicted EAR otherwise.
    """
    f = effective_bits_fn or (lambda b: float(b))
    lo, hi = float(min(db.bitwidths)), float(max(db.bitwidths))
    lo_eff, hi_eff = f(int(lo)), f(int(hi))

    best: AllocationResult | None = None
    trace: list[dict] = []

    # The highest bitwidth is the best this database can do; if even that misses
    # the target, report it rather than silently returning something worse.
    top = _solve(db, hi_eff, "ear", effective_bits_fn, solver)
    if db.predict_ear(top.assignment) < target_ear:
        result = SearchResult(
            assignment=top.assignment,
            average_bits=top.average_bits,
            target=f"EAR>={target_ear}",
            predicted_ear=db.predict_ear(top.assignment),
            predicted_kl=db.predict_kl(top.assignment),
            satisfied=False,
            trace=[{"budget": hi_eff, "ear": db.predict_ear(top.assignment)}],
            notes={"reason": "target unreachable at the maximum bitwidth"},
        )
        if measure is not None:
            result.measured_ear, result.measured_kl = measure(top.assignment)
        return result

    best = top
    lo_b, hi_b = lo_eff, hi_eff
    for _ in range(max_iters):
        if hi_b - lo_b <= tolerance:
            break
        mid = 0.5 * (lo_b + hi_b)
        res = _solve(db, mid, "ear", effective_bits_fn, solver)
        ear = db.predict_ear(res.assignment)
        trace.append({"budget": mid, "average_bits": res.average_bits, "predicted_ear": ear})
        if ear >= target_ear:
            hi_b, best = mid, res
        else:
            lo_b = mid

    result = SearchResult(
        assignment=best.assignment,
        average_bits=best.average_bits,
        target=f"EAR>={target_ear}",
        predicted_ear=db.predict_ear(best.assignment),
        predicted_kl=db.predict_kl(best.assignment),
        satisfied=db.predict_ear(best.assignment) >= target_ear,
        trace=trace,
    )
    if measure is not None:
        result.measured_ear, result.measured_kl = measure(best.assignment)
        result.satisfied = result.measured_ear >= target_ear
    return result


def search_task_lossless(
    db: SensitivityDatabase,
    measure_kl: MeasureKLFn,
    benchmark: BenchmarkFn,
    target_recovery: float = 0.99,
    calibration_bits: float = 4.0,
    effective_bits_fn: Callable[[int], float] | None = None,
    tolerance: float = 0.01,
    max_iters: int = 40,
    solver: str = "auto",
    guardrail_factor: float = 2.0,
    fallback_calibration_bits: float = 6.0,
) -> SearchResult:
    """Task-lossless search via single-point calibration (Algorithm 2).

    Steps 1-2 cost one forward pass and two benchmark runs; step 3 costs
    nothing beyond ILP solves.

    Args:
        db: Sensitivity database.
        measure_kl: Measures actual KL for an assignment (one forward pass).
        benchmark: Benchmark score for an assignment; called with ``None`` for
            the full-precision baseline.
        target_recovery: Desired fraction of baseline accuracy, e.g. 0.99.
        calibration_bits: The anchor budget. The paper uses uniform 4-bit and
            notes 4- and 6-bit anchors agree, while 3-bit is not recommended.
        effective_bits_fn: Nominal-to-realized bitwidth map (Table 7).
        tolerance: Binary-search resolution in bits.
        max_iters: Cap on bisection steps.
        solver: Passed to the allocator.
        guardrail_factor: Reject the result if the actual-to-predicted KL ratio
            drifts from ``rho`` by more than this factor, and re-anchor once.
        fallback_calibration_bits: Anchor used on re-anchoring.

    Returns:
        A :class:`SearchResult` whose ``notes`` record ``alpha``, ``rho``, the
        derived KL threshold and whether the guardrail fired.
    """
    # -- Step 1: calibrate ------------------------------------------------- #
    cal = _solve(db, calibration_bits, "kl", effective_bits_fn, solver)
    kl_actual = measure_kl(cal.assignment)
    kl_pred = db.predict_kl(cal.assignment)

    if kl_actual <= 0:
        raise ValueError(
            f"calibration anchor at {calibration_bits} bits measured KL={kl_actual:.3e}; "
            "the anchor must show measurable degradation. Lower calibration_bits, "
            "or use kl_mode='renormalized' if the truncated KL went negative."
        )
    if kl_pred <= 0:
        raise ValueError(
            f"sensitivity database predicts KL={kl_pred:.3e} at the anchor; "
            "the database carries no usable signal (try more permutations)"
        )
    rho = kl_actual / kl_pred

    base_score = benchmark(None)
    cal_score = benchmark(cal.assignment)
    if base_score == 0:
        raise ValueError("baseline benchmark score is zero; cannot form a recovery ratio")
    recovery_cal = cal_score / base_score

    # -- Step 2: fit the one-point linear model (Eq. 4) -------------------- #
    alpha = (1.0 - recovery_cal) / kl_actual
    if alpha <= 0:
        # The anchor matched or beat the baseline, so no slope is identifiable.
        # Treat the anchor as already task-lossless and return it.
        return SearchResult(
            assignment=cal.assignment,
            average_bits=cal.average_bits,
            target=f"recovery>={target_recovery}",
            predicted_kl=kl_pred,
            measured_kl=kl_actual,
            satisfied=True,
            notes={
                "alpha": alpha,
                "rho": rho,
                "recovery_at_anchor": recovery_cal,
                "reason": "anchor met or exceeded baseline; slope not identifiable",
            },
        )
    kl_threshold = (1.0 - target_recovery) / alpha

    # -- Step 3: binary search, no forward passes -------------------------- #
    f = effective_bits_fn or (lambda b: float(b))
    lo_b, hi_b = f(min(db.bitwidths)), f(max(db.bitwidths))
    best = _solve(db, hi_b, "kl", effective_bits_fn, solver)
    trace: list[dict] = []

    for _ in range(max_iters):
        if hi_b - lo_b <= tolerance:
            break
        mid = 0.5 * (lo_b + hi_b)
        res = _solve(db, mid, "kl", effective_bits_fn, solver)
        kl_hat = rho * db.predict_kl(res.assignment)
        trace.append(
            {"budget": mid, "average_bits": res.average_bits, "calibrated_kl": kl_hat}
        )
        if kl_hat <= kl_threshold:
            hi_b, best = mid, res
        else:
            lo_b = mid

    # -- Guardrail against an out-of-regime anchor ------------------------- #
    final_pred = db.predict_kl(best.assignment)
    final_actual = measure_kl(best.assignment)
    guardrail_fired = False
    if final_pred > 0:
        ratio = final_actual / final_pred
        drift = max(ratio / rho, rho / ratio) if ratio > 0 else float("inf")
        if drift > guardrail_factor and fallback_calibration_bits != calibration_bits:
            guardrail_fired = True
            return _reanchor(
                db,
                measure_kl,
                benchmark,
                target_recovery,
                fallback_calibration_bits,
                effective_bits_fn,
                tolerance,
                max_iters,
                solver,
                original_anchor=calibration_bits,
                drift=drift,
            )

    return SearchResult(
        assignment=best.assignment,
        average_bits=best.average_bits,
        target=f"recovery>={target_recovery}",
        predicted_kl=final_pred,
        measured_kl=final_actual,
        satisfied=final_actual <= kl_threshold,
        trace=trace,
        notes={
            "alpha": alpha,
            "rho": rho,
            "kl_threshold": kl_threshold,
            "recovery_at_anchor": recovery_cal,
            "calibration_bits": calibration_bits,
            "guardrail_fired": guardrail_fired,
        },
    )


def _reanchor(
    db: SensitivityDatabase,
    measure_kl: MeasureKLFn,
    benchmark: BenchmarkFn,
    target_recovery: float,
    anchor: float,
    effective_bits_fn,
    tolerance: float,
    max_iters: int,
    solver: str,
    original_anchor: float,
    drift: float,
) -> SearchResult:
    """Re-run the search from a different anchor after the guardrail fired."""
    result = search_task_lossless(
        db,
        measure_kl,
        benchmark,
        target_recovery=target_recovery,
        calibration_bits=anchor,
        effective_bits_fn=effective_bits_fn,
        tolerance=tolerance,
        max_iters=max_iters,
        solver=solver,
        guardrail_factor=float("inf"),  # do not recurse
    )
    result.notes.update(
        {
            "guardrail_fired": True,
            "original_calibration_bits": original_anchor,
            "observed_drift": drift,
        }
    )
    return result


def search_memory_budget(
    db: SensitivityDatabase,
    budget_bytes: float,
    effective_bits_fn: Callable[[int], float] | None = None,
    solver: str = "auto",
    overhead_fraction: float = 0.0,
) -> SearchResult:
    """Maximize predicted fidelity subject to a hard memory budget.

    The bitwidth-targeted searches above answer "what is the cheapest model
    meeting this quality bar". Deployment usually asks the opposite question:
    the RAM is fixed, so spend it as well as possible. This solves the knapsack
    once, at the largest average bitwidth the budget allows.

    Args:
        db: Sensitivity database.
        budget_bytes: Total bytes available for the quantized weights.
        effective_bits_fn: Nominal-to-realized bitwidth map (Table 7), so the
            budget accounts for scale and zero-point overhead.
        solver: Passed to the allocator.
        overhead_fraction: Reserve this fraction of the budget for runtime
            overhead -- KV cache, activations, the framework itself. For
            example 0.15 keeps 15% free.

    Returns:
        A :class:`SearchResult` whose ``average_bits`` is the realized bits per
        parameter and whose ``notes`` record the byte accounting.

    Raises:
        ValueError: If the budget cannot fit even the smallest bitwidth.
    """
    if budget_bytes <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    usable = budget_bytes * (1.0 - overhead_fraction)
    total_params = sum(db.numel[g] for g in db.groups)
    if total_params == 0:
        raise ValueError("sensitivity database has no parameters")

    # Bits per parameter the budget affords, then solve at that budget.
    bpp = usable * 8.0 / total_params
    f = effective_bits_fn or (lambda b: float(b))
    floor_bpp = f(min(db.bitwidths))
    if bpp < floor_bpp:
        need = total_params * floor_bpp / 8.0
        raise ValueError(
            f"budget of {budget_bytes / 1e9:.2f} GB affords {bpp:.2f} bits/param, "
            f"below the smallest available bitwidth ({floor_bpp:.2f}). "
            f"The smallest configuration needs {need / 1e9:.2f} GB "
            f"({need / (1 - overhead_fraction) / 1e9:.2f} GB with overhead)."
        )

    res = _solve(db, min(bpp, f(max(db.bitwidths))), "ear", effective_bits_fn, solver)
    realized = res.average_bits * total_params / 8.0
    return SearchResult(
        assignment=res.assignment,
        average_bits=res.average_bits,
        target=f"fit<={budget_bytes / 1e9:.2f}GB",
        predicted_ear=db.predict_ear(res.assignment),
        predicted_kl=db.predict_kl(res.assignment),
        satisfied=realized <= usable * (1 + 1e-9),
        notes={
            "budget_bytes": budget_bytes,
            "usable_bytes": usable,
            "realized_bytes": realized,
            "realized_gigabytes": realized / 1e9,
            "headroom_bytes": usable - realized,
            "overhead_fraction": overhead_fraction,
            "bits_per_param_afforded": bpp,
        },
    )
