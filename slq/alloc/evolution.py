"""Constraint-based evolutionary search (Algorithm 3, Appendix A.4).

An alternative to the ILP solver, adapting EvoPress (Sieberling et al., 2025)
to SLQ's inverted formulation: where the original minimizes loss subject to a
bitwidth budget, this minimizes *bitwidth* subject to a quality constraint.

Fitness for an offspring ``b'`` with parent ``b`` and reduction
``db = b_bar(b) - b_bar(b')`` (Eq. 9):

    f(b') = D_KL(b') / db  *  { 1                        if D_KL(b') <= tau
                              { penalty * (D_KL(b') - tau)  otherwise

Lower is better; offspring that do not reduce bitwidth are rejected outright.

Because the search only accepts bitwidth-reducing mutations, a bad early
allocation cannot be undone. Adaptive curation handles this: when progress
stalls, it performs bitwidth-neutral swaps between equal-sized groups, holding
the average constant while moving capacity from insensitive to sensitive
groups.

Appendix A.4 is explicit that this needs orders of magnitude more forward
passes than the ILP and does not scale to large models; it is included as the
reference formulation, and all headline results in the paper use the ILP.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = ["EvolutionConfig", "EvolutionResult", "evolutionary_search"]

#: Evaluates an assignment at a given fidelity budget, returning KL.
EvaluateFn = Callable[[dict[str, int], int | None], float]


@dataclass
class EvolutionConfig:
    """Hyperparameters for the evolutionary search.

    Args:
        offspring: Number of offspring ``lambda`` generated per generation.
        threshold: The quality constraint ``tau`` on KL divergence.
        penalty: The coefficient ``gamma >> 1`` applied to constraint violations.
            Unrelated to the centering inefficiency of Definition 3.1.
        stall_limit: Generations without improvement before curation fires.
        max_generations: Hard cap on generations.
        selection_stages: Calibration-batch counts per selection stage, in
            increasing order, so cheap evaluations filter most candidates.
        mutations: Number of level switches per mutation.
        seed: RNG seed.
    """

    offspring: int = 16
    threshold: float = 0.01
    penalty: float = 1e3
    stall_limit: int = 5
    max_generations: int = 100
    selection_stages: tuple[int, ...] = (1, 4)
    mutations: int = 2
    seed: int = 0


@dataclass
class EvolutionResult:
    assignment: dict[str, int]
    average_bits: float
    kl: float
    generations: int
    history: list[dict] = field(default_factory=list)


def _average_bits(
    assignment: dict[str, int],
    numel: dict[str, int],
    effective_bits_fn: Callable[[int], float] | None,
) -> float:
    f = effective_bits_fn or (lambda b: float(b))
    total = sum(numel.values())
    if total == 0:
        return float("nan")
    return sum(f(b) * numel[g] for g, b in assignment.items()) / total


def _mutate(
    assignment: dict[str, int],
    bitwidths: Sequence[int],
    rng: random.Random,
    n_mut: int,
) -> dict[str, int]:
    """Level-switch mutation: drop a few groups one step down the grid."""
    child = dict(assignment)
    groups = [g for g, b in child.items() if b > min(bitwidths)]
    if not groups:
        return child
    for g in rng.sample(groups, min(n_mut, len(groups))):
        lower = [b for b in bitwidths if b < child[g]]
        child[g] = rng.choice(lower)
    return child


def _curate(
    assignment: dict[str, int],
    numel: dict[str, int],
    bitwidths: Sequence[int],
    rng: random.Random,
) -> dict[str, int]:
    """Bitwidth-neutral reallocation between equal-sized groups.

    Swapping the bitwidths of two groups of identical size leaves the average
    bitwidth exactly unchanged, so this explores sideways when the
    monotonically-decreasing search has painted itself into a corner.
    """
    by_size: dict[int, list[str]] = {}
    for g, n in numel.items():
        by_size.setdefault(n, []).append(g)
    candidates = [gs for gs in by_size.values() if len(gs) >= 2]
    if not candidates:
        return dict(assignment)

    child = dict(assignment)
    pool = rng.choice(candidates)
    a, b = rng.sample(pool, 2)
    child[a], child[b] = child[b], child[a]
    return child


def evolutionary_search(
    groups: Sequence[str],
    bitwidths: Sequence[int],
    numel: dict[str, int],
    evaluate: EvaluateFn,
    config: EvolutionConfig | None = None,
    effective_bits_fn: Callable[[int], float] | None = None,
) -> EvolutionResult:
    """Minimize average bitwidth subject to ``D_KL <= tau``.

    Args:
        groups: Group names.
        bitwidths: Candidate bitwidths.
        numel: Parameter count per group.
        evaluate: ``(assignment, max_batches) -> KL``. ``max_batches`` selects
            how much calibration data the stage uses.
        config: Search hyperparameters.
        effective_bits_fn: Nominal-to-realized bitwidth map (Table 7).

    Returns:
        The best feasible configuration found.
    """
    cfg = config or EvolutionConfig()
    rng = random.Random(cfg.seed)
    bits = sorted(bitwidths)

    parent = {g: bits[-1] for g in groups}
    parent_bits = _average_bits(parent, numel, effective_bits_fn)
    parent_kl = evaluate(parent, None)
    stall = 0
    history: list[dict] = []

    for gen in range(cfg.max_generations):
        candidates = [_mutate(parent, bits, rng, cfg.mutations) for _ in range(cfg.offspring)]
        # Reject offspring that fail to reduce the average bitwidth.
        candidates = [
            c
            for c in candidates
            if _average_bits(c, numel, effective_bits_fn) < parent_bits - 1e-12
        ]

        if candidates:
            # Multi-stage selection: each stage uses more calibration data.
            for stage_batches in cfg.selection_stages[:-1]:
                if len(candidates) <= 1:
                    break
                scored = [
                    (_fitness(c, parent_bits, evaluate(c, stage_batches), cfg, numel,
                              effective_bits_fn), c)
                    for c in candidates
                ]
                scored.sort(key=lambda t: t[0])
                candidates = [c for _, c in scored[: max(1, len(scored) // 2)]]

            final_batches = cfg.selection_stages[-1] if cfg.selection_stages else None
            scored = [
                (_fitness(c, parent_bits, evaluate(c, final_batches), cfg, numel,
                          effective_bits_fn), c)
                for c in candidates
            ]
            scored.sort(key=lambda t: t[0])
            best_fit, best_child = scored[0]
            child_kl = evaluate(best_child, None)
            child_bits = _average_bits(best_child, numel, effective_bits_fn)
        else:
            best_child, child_kl, child_bits = None, float("inf"), float("inf")

        accepted = (
            best_child is not None
            and child_kl <= cfg.threshold
            and child_bits < parent_bits - 1e-12
        )
        if accepted:
            parent, parent_kl, parent_bits = best_child, child_kl, child_bits
            stall = 0
        else:
            stall += 1

        history.append(
            {"generation": gen, "average_bits": parent_bits, "kl": parent_kl, "stall": stall}
        )

        if stall >= cfg.stall_limit:
            curated = _curate(parent, numel, bits, rng)
            curated_kl = evaluate(curated, None)
            # Accept a sideways move only if it does not break the constraint.
            if curated_kl <= cfg.threshold and curated_kl < parent_kl:
                parent, parent_kl = curated, curated_kl
            stall = 0

    return EvolutionResult(
        assignment=parent,
        average_bits=parent_bits,
        kl=parent_kl,
        generations=len(history),
        history=history,
    )


def _fitness(
    child: dict[str, int],
    parent_bits: float,
    kl: float,
    cfg: EvolutionConfig,
    numel: dict[str, int],
    effective_bits_fn: Callable[[int], float] | None,
) -> float:
    """Eq. 9. Lower is better; infeasible offspring are pushed far up."""
    delta_b = parent_bits - _average_bits(child, numel, effective_bits_fn)
    if delta_b <= 0:
        return float("inf")
    base = kl / delta_b
    if kl <= cfg.threshold:
        return base
    return base * cfg.penalty * (kl - cfg.threshold)
