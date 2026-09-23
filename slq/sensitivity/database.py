"""The sensitivity database: predicted degradation per (group, bitwidth).

Both estimators in this package produce the same artifact -- a cost table
``c[group][bits]`` for KL and EAR -- and the allocator consumes only that. The
database is computed once and every subsequent ILP solve is arithmetic on it,
which is what makes the binary search over bitwidth budgets essentially free
(Section 3.3).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = ["SensitivityDatabase"]


@dataclass
class SensitivityDatabase:
    """Predicted metric degradation for every (group, bitwidth) pair.

    Attributes:
        groups: Group names, in allocation order.
        bitwidths: Candidate bitwidths.
        numel: Parameter count per group, used for the weighted average bitwidth.
        kl: ``kl[group][bits]`` predicted KL divergence.
        ear_drop: ``ear_drop[group][bits]`` predicted EAR *degradation*, so that
            predicted EAR is ``1 - sum_m ear_drop[m][b_m]`` (Eq. 8).
        baseline_ear: Measured EAR of the reference configuration (every group
            at ``b_max``). Equations 7-8 assume this is 1, which holds for a
            sharply peaked LLM but not in general; anchoring on the measured
            value makes the prediction correct either way and reduces to the
            paper's formula when the baseline is 1.
        baseline_kl: Measured KL of the reference configuration, normally ~0.
        method: ``"shapley"`` or ``"linear"``.
        meta: Free-form provenance (permutation count, calibration size, ...).
    """

    groups: list[str]
    bitwidths: list[int]
    numel: dict[str, int]
    kl: dict[str, dict[int, float]] = field(default_factory=dict)
    ear_drop: dict[str, dict[int, float]] = field(default_factory=dict)
    baseline_ear: float = 1.0
    baseline_kl: float = 0.0
    method: str = "shapley"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Prediction (Appendix A.2)
    # ------------------------------------------------------------------ #

    def predict_kl(self, assignment: dict[str, int]) -> float:
        """Predicted KL divergence (Eq. 7), anchored on the measured baseline.

        Eq. 7 reads ``D_KL_hat = sum_m phi_m^(b_m)``. The Shapley values are
        increments relative to the all-``b_max`` configuration, so the baseline
        KL of that configuration is added back; it is ~0 in practice, making
        this identical to the paper's formula.
        """
        return self.baseline_kl + sum(self.kl[g][assignment[g]] for g in self.groups)

    def predict_ear(self, assignment: dict[str, int]) -> float:
        """Predicted EAR (Eq. 8), anchored on the measured baseline.

        Eq. 8 reads ``EAR_hat = 1 - sum_m phi_m^(b_m)``, which takes the
        reference configuration's EAR to be exactly 1. That is a good
        approximation for a sharply peaked LLM, where EAR at ``b_max`` exceeds
        0.999, but it inflates every prediction by ``1 - EAR(b_max)`` otherwise
        -- enough to make the DL binary search undershoot the bitwidth it needs.
        Using the measured baseline is exact in both regimes.
        """
        return self.baseline_ear - sum(self.ear_drop[g][assignment[g]] for g in self.groups)

    def average_bits(
        self, assignment: dict[str, int], effective_bits_fn=None
    ) -> float:
        """Parameter-weighted average bitwidth of an assignment.

        Args:
            assignment: Group-to-bitwidth mapping.
            effective_bits_fn: Optional ``bits -> realized bits`` map applying
                the scale / zero-point overhead of Table 7.
        """
        total = sum(self.numel[g] for g in self.groups)
        if total == 0:
            return float("nan")
        f = effective_bits_fn or (lambda b: float(b))
        return sum(f(assignment[g]) * self.numel[g] for g in self.groups) / total

    def cost_table(self, metric: str = "ear") -> dict[str, dict[int, float]]:
        """The cost table the ILP minimizes."""
        if metric == "ear":
            return self.ear_drop
        if metric == "kl":
            return self.kl
        raise ValueError(f"metric must be 'ear' or 'kl', got {metric!r}")

    # ------------------------------------------------------------------ #
    # Consistency helpers
    # ------------------------------------------------------------------ #

    def enforce_monotonic(self) -> SensitivityDatabase:
        """Make costs non-increasing in bitwidth.

        Sampling noise can leave a group marginally cheaper at ``b`` than at
        ``b+1``, which lets the ILP buy fidelity by *removing* bits and produces
        nonsensical allocations. Costs are therefore made monotone by taking a
        running minimum from the highest bitwidth downward.
        """
        for table in (self.kl, self.ear_drop):
            for g in self.groups:
                best = float("inf")
                for b in sorted(self.bitwidths, reverse=True):
                    best = min(best, table[g][b])
                    table[g][b] = best
        return self

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "groups": self.groups,
            "bitwidths": self.bitwidths,
            "numel": self.numel,
            "kl": {g: {str(b): v for b, v in d.items()} for g, d in self.kl.items()},
            "ear_drop": {
                g: {str(b): v for b, v in d.items()} for g, d in self.ear_drop.items()
            },
            "baseline_ear": self.baseline_ear,
            "baseline_kl": self.baseline_kl,
            "method": self.method,
            "meta": self.meta,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> SensitivityDatabase:
        with open(path, encoding="utf-8") as f:
            p = json.load(f)
        return cls(
            groups=p["groups"],
            bitwidths=p["bitwidths"],
            numel=p["numel"],
            kl={g: {int(b): v for b, v in d.items()} for g, d in p["kl"].items()},
            ear_drop={g: {int(b): v for b, v in d.items()} for g, d in p["ear_drop"].items()},
            baseline_ear=p.get("baseline_ear", 1.0),
            baseline_kl=p.get("baseline_kl", 0.0),
            method=p.get("method", "shapley"),
            meta=p.get("meta", {}),
        )

    @classmethod
    def empty(
        cls, groups: Sequence[str], bitwidths: Sequence[int], numel: dict[str, int], method: str
    ) -> SensitivityDatabase:
        return cls(
            groups=list(groups),
            bitwidths=list(bitwidths),
            numel=dict(numel),
            kl={g: {b: 0.0 for b in bitwidths} for g in groups},
            ear_drop={g: {b: 0.0 for b in bitwidths} for g in groups},
            method=method,
        )
