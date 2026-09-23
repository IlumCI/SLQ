"""Per-site smoothing-strength search under SLQ's fidelity metrics.

SmoothQuant picks one global ``alpha`` (0.5 by default, 0.85 for the hardest
models) and validates it by downstream accuracy. That is coarse in two ways:
the best migration strength differs between attention and MLP sites, and
accuracy is a noisy, expensive signal.

SLQ supplies what is missing. EAR is a cheap, bounded, calibration-set metric
that predicts distributional fidelity directly, so ``alpha`` can be chosen per
site by measuring rather than guessing -- and it can be chosen *jointly* with
the quantization grid it will be quantized on, which matters because migration
moves difficulty onto exactly the weights the grid has to represent.

Two procedures are provided:

* :func:`search_global_alpha` -- sweep one shared ``alpha``, the honest
  like-for-like upgrade over SmoothQuant's fixed choice.
* :func:`search_per_site_alpha` -- coordinate descent over sites, each step
  taking the ``alpha`` that maximizes EAR with the rest held fixed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from slq.smooth.smooth import SmoothSite, smooth_model

__all__ = ["AlphaSearchResult", "search_global_alpha", "search_per_site_alpha"]

DEFAULT_GRID = (0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95)

#: Rebuilds and scores a model under a given per-site alpha map, returning EAR.
ScoreFn = Callable[[dict[str, float]], float]


@dataclass
class AlphaSearchResult:
    """The chosen smoothing strengths and the search's audit trail."""

    alphas: dict[str, float]
    score: float
    trace: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"alphas": dict(self.alphas), "score": self.score, "trace": self.trace}


def make_scorer(
    build: Callable[[dict[str, float]], "object"],
    evaluate: Callable[["object"], float],
) -> ScoreFn:
    """Compose a model builder and an evaluator into a scoring function.

    Args:
        build: Maps an alpha map to a fresh smoothed-and-quantized model.
        evaluate: Scores that model; higher is better (EAR).
    """

    def score(alphas: dict[str, float]) -> float:
        return evaluate(build(alphas))

    return score


def search_global_alpha(
    score: ScoreFn,
    sites: Sequence[SmoothSite],
    grid: Sequence[float] = DEFAULT_GRID,
) -> AlphaSearchResult:
    """Sweep a single shared migration strength across all sites.

    Args:
        score: Maps an alpha map to EAR; higher is better.
        sites: The smoothing sites to configure identically.
        grid: Candidate ``alpha`` values.

    Returns:
        The best shared ``alpha``, expanded to a per-site map.
    """
    trace: list[dict] = []
    best_alpha, best_score = None, -float("inf")
    for a in grid:
        alphas = {s.norm: float(a) for s in sites}
        val = score(alphas)
        trace.append({"alpha": float(a), "score": val})
        if val > best_score:
            best_alpha, best_score = float(a), val
    return AlphaSearchResult(
        alphas={s.norm: best_alpha for s in sites}, score=best_score, trace=trace
    )


def search_per_site_alpha(
    score: ScoreFn,
    sites: Sequence[SmoothSite],
    grid: Sequence[float] = DEFAULT_GRID,
    init: float | dict[str, float] = 0.5,
    rounds: int = 1,
) -> AlphaSearchResult:
    """Coordinate descent on per-site migration strength.

    Each site is swept in turn with the others held at their current values,
    keeping whichever ``alpha`` maximizes EAR. One round costs
    ``len(sites) * len(grid)`` evaluations.

    Args:
        score: Maps an alpha map to EAR; higher is better.
        sites: Smoothing sites, optimized in the given order.
        grid: Candidate ``alpha`` values per site.
        init: Starting point, shared or per site.
        rounds: Coordinate-descent sweeps. A second round rarely moves much
            because the sites interact only weakly through the residual stream.

    Returns:
        The per-site alpha map and the EAR it achieved.
    """
    alphas: dict[str, float] = (
        {s.norm: float(init) for s in sites}
        if isinstance(init, (int, float))
        else {s.norm: float(init.get(s.norm, 0.5)) for s in sites}
    )
    best = score(alphas)
    trace: list[dict] = [{"round": -1, "site": None, "alpha": None, "score": best}]

    for r in range(rounds):
        for site in sites:
            current = alphas[site.norm]
            local_best, local_score = current, best
            for a in grid:
                if a == current:
                    continue
                trial = dict(alphas)
                trial[site.norm] = float(a)
                val = score(trial)
                trace.append(
                    {"round": r, "site": site.norm, "alpha": float(a), "score": val}
                )
                if val > local_score:
                    local_best, local_score = float(a), val
            alphas[site.norm], best = local_best, local_score

    return AlphaSearchResult(alphas=alphas, score=best, trace=trace)


@torch.no_grad()
def smoothed_copy(
    model: nn.Module,
    act_scales: dict[str, torch.Tensor],
    sites: Sequence[SmoothSite],
    alphas: dict[str, float],
    group_size: int = 128,
) -> nn.Module:
    """Deep-copy a model and apply smoothing to the copy.

    Smoothing rewrites weights in place, so a search over ``alpha`` needs a
    clean model per trial. For anything larger than a test model, prefer
    restoring saved weights over copying.
    """
    clone = copy.deepcopy(model)
    smooth_model(clone, act_scales, sites, alpha=alphas, group_size=group_size)
    return clone
