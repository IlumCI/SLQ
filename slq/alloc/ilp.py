"""Bitwidth allocation as a multiple-choice knapsack problem (Section 3.3).

Given the sensitivity database ``c[m][b]``, assign each group exactly one
bitwidth so as to minimize total predicted degradation under an
average-bitwidth budget:

    minimize    sum_{m,b} x_{m,b} c_{m,b}
    subject to  sum_b x_{m,b} = 1                        for every group m
                sum_{m,b} x_{m,b} * bits(b) * |G_m|  <=  b_bar * sum_m |G_m|
                x_{m,b} in {0,1}

Note the constraint is weighted by each group's parameter count, matching the
paper's definition of the average bitwidth
``b_bar = sum_l b_l |W_l| / sum_l |W_l|``; an unweighted budget would let the
solver spend bits freely on the largest tensors.

Two solvers are provided. The default uses HiGHS via ``scipy.optimize.milp``
for a proven-optimal answer; a Lagrangian-relaxation fallback keeps the package
working without SciPy and is used automatically if the MILP solve fails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["AllocationResult", "solve_allocation"]


@dataclass
class AllocationResult:
    """The outcome of one allocation solve."""

    assignment: dict[str, int]
    predicted_cost: float
    average_bits: float
    feasible: bool
    solver: str

    def as_dict(self) -> dict:
        return {
            "assignment": dict(self.assignment),
            "predicted_cost": self.predicted_cost,
            "average_bits": self.average_bits,
            "feasible": self.feasible,
            "solver": self.solver,
        }


def _prepare(
    groups: Sequence[str],
    bitwidths: Sequence[int],
    costs: dict[str, dict[int, float]],
    numel: dict[str, int],
    effective_bits_fn: Callable[[int], float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the cost, weight and size matrices used by both solvers."""
    f = effective_bits_fn or (lambda b: float(b))
    n_g, n_b = len(groups), len(bitwidths)
    c = np.empty((n_g, n_b), dtype=np.float64)
    w = np.empty((n_g, n_b), dtype=np.float64)
    sizes = np.array([float(numel[g]) for g in groups], dtype=np.float64)
    for i, g in enumerate(groups):
        for j, b in enumerate(bitwidths):
            c[i, j] = costs[g][b]
            w[i, j] = f(b) * sizes[i]
    return c, w, sizes


def _solve_milp(
    c: np.ndarray, w: np.ndarray, budget: float
) -> tuple[np.ndarray | None, str]:
    """Exact solve with HiGHS through SciPy."""
    try:
        from scipy.optimize import LinearConstraint, milp
        from scipy.sparse import csr_matrix
    except ImportError:
        return None, "scipy-missing"

    n_g, n_b = c.shape
    n_var = n_g * n_b

    # One-hot constraint per group.
    rows, cols = [], []
    for i in range(n_g):
        for j in range(n_b):
            rows.append(i)
            cols.append(i * n_b + j)
    a_eq = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_g, n_var))
    onehot = LinearConstraint(a_eq, lb=np.ones(n_g), ub=np.ones(n_g))

    # Average-bitwidth budget.
    a_ub = csr_matrix(w.reshape(1, -1))
    budget_c = LinearConstraint(a_ub, lb=-np.inf, ub=budget)

    res = milp(
        c=c.reshape(-1),
        constraints=[onehot, budget_c],
        integrality=np.ones(n_var),
        bounds=(lambda: __import__("scipy.optimize", fromlist=["Bounds"]).Bounds(0, 1))(),
    )
    if not res.success or res.x is None:
        return None, "milp-infeasible"
    return np.asarray(res.x).reshape(n_g, n_b), "milp"


def _solve_lagrangian(
    c: np.ndarray, w: np.ndarray, budget: float, iterations: int = 64
) -> tuple[np.ndarray, str]:
    """Lagrangian-relaxation fallback.

    Bisects the multiplier ``lam`` on the budget constraint; for a fixed ``lam``
    each group independently picks ``argmin_b (c + lam*w)``. Then greedily
    upgrades groups while budget remains, which repairs the integrality gap the
    relaxation can leave.
    """
    n_g, n_b = c.shape

    def pick(lam: float) -> np.ndarray:
        return np.argmin(c + lam * w, axis=1)

    lo, hi = 0.0, 1.0
    scale = max(float(np.abs(c).max()), 1e-12) / max(float(w.max()), 1e-12)
    hi = scale * 1e6
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        used = w[np.arange(n_g), pick(mid)].sum()
        if used > budget:
            lo = mid
        else:
            hi = mid
    choice = pick(hi)

    # Greedy upgrade: spend leftover budget where it buys the most fidelity.
    used = w[np.arange(n_g), choice].sum()
    improved = True
    while improved:
        improved = False
        best_ratio, best = 0.0, None
        for i in range(n_g):
            for j in range(n_b):
                dw = w[i, j] - w[i, choice[i]]
                dc = c[i, choice[i]] - c[i, j]
                if dw <= 0 or dc <= 0 or used + dw > budget:
                    continue
                ratio = dc / dw
                if ratio > best_ratio:
                    best_ratio, best = ratio, (i, j)
        if best is not None:
            i, j = best
            used += w[i, j] - w[i, choice[i]]
            choice[i] = j
            improved = True

    x = np.zeros_like(c)
    x[np.arange(n_g), choice] = 1.0
    return x, "lagrangian"


def solve_allocation(
    groups: Sequence[str],
    bitwidths: Sequence[int],
    costs: dict[str, dict[int, float]],
    numel: dict[str, int],
    budget_bits: float,
    effective_bits_fn: Callable[[int], float] | None = None,
    solver: str = "auto",
) -> AllocationResult:
    """Solve the multiple-choice knapsack for one bitwidth budget.

    Args:
        groups: Group names.
        bitwidths: Candidate bitwidths.
        costs: ``costs[group][bits]`` predicted degradation.
        numel: Parameter count per group.
        budget_bits: Target average bits per parameter ``b_bar``.
        effective_bits_fn: Maps a nominal bitwidth to its realized cost
            including scale / zero-point overhead (Table 7). Without it the
            budget is interpreted in nominal bits.
        solver: ``"auto"`` (MILP, falling back to Lagrangian), ``"milp"`` or
            ``"lagrangian"``.

    Returns:
        An :class:`AllocationResult`. ``feasible`` is ``False`` when the budget
        is below what the smallest bitwidth can achieve, in which case the
        minimum-bitwidth assignment is returned.
    """
    groups = list(groups)
    bitwidths = sorted(bitwidths)
    if not groups:
        raise ValueError("no groups to allocate")

    c, w, sizes = _prepare(groups, bitwidths, costs, numel, effective_bits_fn)
    total = float(sizes.sum())
    budget = budget_bits * total

    # Infeasible below the cheapest configuration: return it and say so.
    floor = w[:, 0].sum()
    if budget < floor:
        assignment = {g: bitwidths[0] for g in groups}
        return AllocationResult(
            assignment=assignment,
            predicted_cost=float(c[:, 0].sum()),
            average_bits=floor / total,
            feasible=False,
            solver="floor",
        )

    x = None
    name = ""
    if solver in ("auto", "milp"):
        x, name = _solve_milp(c, w, budget)
        if x is None and solver == "milp":
            raise RuntimeError(f"MILP solve failed ({name})")
    if x is None:
        x, name = _solve_lagrangian(c, w, budget)

    choice = np.argmax(x, axis=1)
    assignment = {g: bitwidths[int(j)] for g, j in zip(groups, choice, strict=True)}
    used = float(w[np.arange(len(groups)), choice].sum())
    return AllocationResult(
        assignment=assignment,
        predicted_cost=float(c[np.arange(len(groups)), choice].sum()),
        average_bits=used / total,
        feasible=used <= budget * (1 + 1e-9),
        solver=name,
    )
