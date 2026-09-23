"""Output-fidelity metrics for losslessness (Section 3.2).

Both metrics are restricted to the top-``K`` tokens under the *reference*
distribution ``p`` and are accumulated over calibration token positions:

    D_KL = (1/N) sum_i sum_{k in T_i} p_i(k) log(p_i(k) / q_i(k))
    EAR  = (1/N) sum_i sum_{k in T_i} min(p_i(k), q_i(k))          (Eq. 3)

On normalization
----------------
"Restricted to the top-K tokens under ``p_i``" is implemented here as
*conditioning* on that support: ``p`` and ``q`` are renormalized over ``T_i``
before the metrics are taken. Three things force that reading.

1. The paper states EAR equals ``1 - d_TV(p, q)`` and that it is "the maximum
   probability that two random variables ``X ~ p`` and ``Y ~ q`` can be coupled
   to agree". Both identities require ``p`` and ``q`` to be probability
   distributions; on a truncated, unnormalized vector they simply do not hold.
2. Without renormalizing, ``EAR <= sum_{k in T_i} p_i(k)``, so EAR is capped by
   the top-K mass. Measured on Qwen3-0.6B over WikiText-2 that cap is ~0.75,
   which would make the paper's ``EAR >= 0.99`` target unreachable by any
   quantizer, including a perfect one.
3. Unnormalized, ``EAR(p, p) < 1`` and the truncated KL can go negative, so
   neither metric is well-behaved as a search objective.

Set ``normalize=False`` to get the literal truncated forms; ``topk_mass`` is
always reported so the size of the truncation stays visible.

EAR equals ``1 - d_TV(p, q)``: the maximum probability that ``X ~ p`` and
``Y ~ q`` can be coupled to agree. Both are computed from the same forward
pass at no additional cost, which is what makes the Shapley sweep affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["FidelityResult", "FidelityMeter", "fidelity", "decision_flip_rates"]

DEFAULT_TOPK = 10


@dataclass(frozen=True)
class FidelityResult:
    """Fidelity of a quantized model's next-token distribution."""

    ear: float
    kl: float
    flip_rate: float
    margin_at_disagreement: float
    positions: int
    topk_mass: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "ear": self.ear,
            "kl": self.kl,
            "flip_rate": self.flip_rate,
            "margin_at_disagreement": self.margin_at_disagreement,
            "topk_mass": self.topk_mass,
            "positions": float(self.positions),
        }

    @property
    def ear_normalized(self) -> float:
        """EAR rescaled by the top-K mass it is bounded by.

        Only meaningful when the metrics were taken with ``normalize=False``,
        where ``EAR <= topk_mass`` by construction. With the default
        ``normalize=True`` the conditioning is already applied and this returns
        a value above 1; use :attr:`ear` instead.
        """
        if not self.topk_mass or self.topk_mass != self.topk_mass:
            return float("nan")
        return self.ear / self.topk_mass


def _to_probs(x: torch.Tensor, is_logits: bool) -> torch.Tensor:
    x = x.float().reshape(-1, x.shape[-1])
    return torch.softmax(x, dim=-1) if is_logits else x


@torch.no_grad()
def fidelity(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    topk: int = DEFAULT_TOPK,
    is_logits: bool = True,
    eps: float = 1e-10,
    normalize: bool = True,
) -> FidelityResult:
    """Compute EAR, top-K KL, decision-flip rate and disagreement margin.

    Args:
        reference: Baseline (BF16) logits or probabilities, ``[..., vocab]``.
        candidate: Quantized-model logits or probabilities, same shape.
        topk: Truncation ``K`` applied to the reference distribution.
        is_logits: Whether the inputs are logits (softmax is applied) or
            already-normalized probabilities.
        eps: Floor guarding the logarithm and the division.
        normalize: Condition ``p`` and ``q`` on the top-K support before
            measuring, so EAR equals ``1 - d_TV`` and KL is a proper
            divergence. See the module docstring. ``False`` gives the literal
            truncated sums.

    Returns:
        A :class:`FidelityResult` aggregated over all token positions.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"shape mismatch: reference {tuple(reference.shape)} vs candidate {tuple(candidate.shape)}"
        )
    p_full = _to_probs(reference, is_logits)
    q_full = _to_probs(candidate, is_logits)
    n_pos, vocab = p_full.shape
    if n_pos == 0:
        return FidelityResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    k = min(topk, vocab)

    # Restrict both distributions to the top-K support of p (Section 3.2).
    p_top, idx = torch.topk(p_full, k, dim=-1)
    q_top = torch.gather(q_full, 1, idx)
    mass = p_top.sum(dim=-1)

    if normalize:
        p_m = p_top / mass.unsqueeze(-1).clamp_min(eps)
        q_m = q_top / q_top.sum(dim=-1, keepdim=True).clamp_min(eps)
    else:
        p_m, q_m = p_top, q_top

    ear = torch.minimum(p_m, q_m).sum(dim=-1)
    kl = (p_m * (p_m.clamp_min(eps).log() - q_m.clamp_min(eps).log())).sum(dim=-1)

    # Decision flips: positions where the argmax token changes.
    ref_arg = idx[:, 0]
    cand_arg = q_full.argmax(dim=-1)
    flips = ref_arg != cand_arg

    # Margin at disagreement: the reference top-1/top-2 gap on flipped positions.
    if k >= 2:
        margin_all = p_top[:, 0] - p_top[:, 1]
    else:
        margin_all = torch.zeros(n_pos, device=p_full.device)
    n_flips = int(flips.sum())
    margin = float(margin_all[flips].mean()) if n_flips > 0 else 0.0

    return FidelityResult(
        ear=float(ear.mean()),
        kl=float(kl.mean()),
        flip_rate=n_flips / n_pos,
        margin_at_disagreement=margin,
        positions=n_pos,
        topk_mass=float(mass.mean()),
    )


@dataclass
class FidelityMeter:
    """Streaming accumulator so metrics can be gathered batch by batch."""

    topk: int = DEFAULT_TOPK
    is_logits: bool = True
    eps: float = 1e-10
    normalize: bool = True
    _ear: float = field(default=0.0, init=False)
    _kl: float = field(default=0.0, init=False)
    _flips: int = field(default=0, init=False)
    _margin: float = field(default=0.0, init=False)
    _mass: float = field(default=0.0, init=False)
    _n: int = field(default=0, init=False)

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        r = fidelity(reference, candidate, self.topk, self.is_logits, self.eps, self.normalize)
        if r.positions == 0:
            return
        self._ear += r.ear * r.positions
        self._kl += r.kl * r.positions
        self._mass += r.topk_mass * r.positions
        n_flips = round(r.flip_rate * r.positions)
        self._flips += n_flips
        self._margin += r.margin_at_disagreement * n_flips
        self._n += r.positions

    def compute(self) -> FidelityResult:
        if self._n == 0:
            return FidelityResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)
        return FidelityResult(
            ear=self._ear / self._n,
            kl=self._kl / self._n,
            flip_rate=self._flips / self._n,
            margin_at_disagreement=(self._margin / self._flips) if self._flips else 0.0,
            positions=self._n,
            topk_mass=self._mass / self._n,
        )

    def reset(self) -> None:
        self._ear = self._kl = self._margin = self._mass = 0.0
        self._flips = self._n = 0


@torch.no_grad()
def decision_flip_rates(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    n_bins: int = 5,
    is_logits: bool = True,
) -> dict[str, list[float]]:
    """Decision-flip rate stratified by reference-distribution entropy.

    Reproduces the analysis behind Figures 2-3: symmetric quantization flips
    more tokens in every entropy bin, with the sym/asym ratio widest in the
    low-entropy bins where the original model was most confident.

    Returns:
        A dict with ``bin_edges`` (entropy quantile edges), ``flip_rate`` per
        bin and ``count`` per bin.
    """
    p = _to_probs(reference, is_logits)
    q = _to_probs(candidate, is_logits)
    entropy = -(p * p.clamp_min(1e-10).log()).sum(dim=-1)
    flips = (p.argmax(dim=-1) != q.argmax(dim=-1)).float()

    qs = torch.linspace(0, 1, n_bins + 1, device=entropy.device)
    edges = torch.quantile(entropy, qs)
    rates, counts = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (entropy >= lo) & (entropy <= hi if i == n_bins - 1 else entropy < hi)
        n = int(mask.sum())
        counts.append(n)
        rates.append(float(flips[mask].mean()) if n else 0.0)
    return {
        "bin_edges": [float(e) for e in edges],
        "flip_rate": rates,
        "count": [float(c) for c in counts],
    }
