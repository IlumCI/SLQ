"""Multi-bitwidth Shapley sensitivity estimation (Algorithm 1, Section 3.3).

Existing Shapley approaches to mixed-precision quantization play a single
binary game, switching each layer between one high and one low precision. That
restricts allocation to two levels, which the paper shows is too coarse for the
near-lossless regime. Algorithm 1 runs a *separate* binary game for each target
bitwidth ``b* in B \\ {b_max}``:

    for each b*:
        for p = 1..P:
            sample a permutation pi of the groups
            set all groups to b_max, evaluate
            for j = 1..M:
                switch G_{pi_j} to b*, evaluate, record the marginal change
        phi_m^(b*) = mean over permutations of the marginal change

Because a group's marginal is measured against a partially-degraded model, the
values capture cross-layer interaction, not just isolated reconstruction error.
Cost is ``O(P * M * |B|)`` forward passes; the games are independent and can be
run in parallel.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import torch

from slq.model.wrapper import QuantizableModel
from slq.sensitivity.database import SensitivityDatabase

__all__ = ["shapley_sensitivity"]

ProgressFn = Callable[[str, int, int], None]


@torch.no_grad()
def shapley_sensitivity(
    qmodel: QuantizableModel,
    bitwidths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    permutations: int = 8,
    seed: int = 0,
    max_batches: int | None = None,
    progress: ProgressFn | None = None,
) -> SensitivityDatabase:
    """Estimate per-group sensitivity across the full bitwidth range.

    Args:
        qmodel: A model whose weight bank already covers ``bitwidths``.
        bitwidths: Candidate bitwidths ``B``. The largest is the reference
            ``b_max`` against which marginals are measured.
        permutations: Number of random permutations ``P`` per game. Higher
            values reduce variance at linear cost.
        seed: Seed for permutation sampling.
        max_batches: Evaluate on a prefix of the calibration set, trading
            estimator variance for speed.
        progress: Optional callback ``(stage, done, total)``.

    Returns:
        A :class:`SensitivityDatabase` holding ``phi_m^(b)`` for KL and EAR,
        made monotone in bitwidth before it is returned.
    """
    bits = sorted(bitwidths)
    if len(bits) < 2:
        raise ValueError("at least two bitwidths are required")
    b_max = bits[-1]
    targets = [b for b in bits if b != b_max]
    groups = qmodel.group_names
    m = len(groups)

    db = SensitivityDatabase.empty(groups, bits, qmodel.group_numel(), method="shapley")
    # The reference bitwidth is the baseline, so it carries zero cost by
    # construction; every other value is a degradation relative to it.
    for g in groups:
        db.kl[g][b_max] = 0.0
        db.ear_drop[g][b_max] = 0.0

    # Measure the reference configuration once; predictions are anchored on it.
    qmodel.apply(dict.fromkeys(groups, b_max))
    baseline = qmodel.evaluate(max_batches=max_batches)
    db.baseline_ear = baseline.ear
    db.baseline_kl = baseline.kl

    rng = random.Random(seed)
    total_steps = len(targets) * permutations * m
    done = 0

    # Marginal contributions accumulated per (group, target bitwidth).
    acc_kl = {b: dict.fromkeys(groups, 0.0) for b in targets}
    acc_ear = {b: dict.fromkeys(groups, 0.0) for b in targets}

    for b_star in targets:
        for _ in range(permutations):
            assignment = dict.fromkeys(groups, b_max)
            qmodel.apply(assignment)
            prev = qmodel.evaluate(max_batches=max_batches)

            order = groups[:]
            rng.shuffle(order)
            for g in order:
                assignment[g] = b_star
                qmodel.apply(assignment)
                cur = qmodel.evaluate(max_batches=max_batches)

                # Marginal change from switching this group, in the context of
                # whichever groups the permutation already switched.
                acc_kl[b_star][g] += cur.kl - prev.kl
                acc_ear[b_star][g] += prev.ear - cur.ear
                prev = cur

                done += 1
                if progress is not None:
                    progress(f"shapley b*={b_star}", done, total_steps)

    for b_star in targets:
        for g in groups:
            db.kl[g][b_star] = acc_kl[b_star][g] / permutations
            db.ear_drop[g][b_star] = acc_ear[b_star][g] / permutations

    db.meta = {
        "permutations": permutations,
        "seed": seed,
        "b_max": b_max,
        "max_batches": max_batches,
        "calibration_batches": len(qmodel.calibration),
        "baseline_ear": db.baseline_ear,
        "baseline_kl": db.baseline_kl,
    }
    qmodel.restore()
    return db.enforce_monotonic()
