"""SmoothQuant difficulty migration, generalized and instrumented for SLQ.

SmoothQuant (Xiao et al., 2023) migrates quantization difficulty from
activations to weights with a per-input-channel scale

    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha),                     (SQ Eq. 4)

applied as ``X -> X diag(s)^-1`` and ``W -> diag(s) W``, which leaves the layer
output unchanged. The inverse scaling is folded into the preceding
normalization layer, so it costs nothing at inference time.

Two departures from the upstream implementation:

* **Site discovery is traced, not hardcoded.** Upstream enumerates
  ``OPTDecoderLayer``, ``LlamaDecoderLayer``, ``BloomBlock`` and friends by
  type. Here a single traced forward pass records which linear layers consume a
  given normalization output, so any architecture -- including ones that did
  not exist when this was written -- is handled without new code.
* **The effect on ``gamma`` is reported.** Smoothing rescales weight columns and
  therefore changes each quantization group's centering inefficiency
  ``gamma = 2M/R`` (Definition 3.1). Since symmetric quantization pays
  ``gamma^2`` in noise variance (Lemma 3.2), smoothing and grid geometry
  interact; :func:`smooth_model` returns the realized change so the interaction
  is measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from slq.quant.grid import centering_inefficiency

__all__ = ["SmoothSite", "SmoothReport", "discover_sites", "compute_smoothing_scale", "smooth_model"]

#: Modules treated as fold-in points for the inverse scale. Any module exposing
#: a per-feature ``weight`` of the right size qualifies, which covers
#: ``LayerNorm``, ``RMSNorm`` and their many reimplementations.
_NORM_HINT = ("norm", "ln", "layernorm", "rmsnorm")


@dataclass
class SmoothSite:
    """A normalization layer and the linear layers fed directly from it."""

    norm: str
    linears: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SmoothSite(norm={self.norm!r}, linears={self.linears!r})"


@dataclass
class SmoothReport:
    """What smoothing did, per site."""

    sites: list[SmoothSite]
    alphas: dict[str, float]
    gamma_before: dict[str, float]
    gamma_after: dict[str, float]

    def gamma_delta(self) -> dict[str, float]:
        """Per-site change in mean centering inefficiency."""
        return {k: self.gamma_after[k] - self.gamma_before[k] for k in self.gamma_before}

    def summary(self) -> dict[str, float]:
        if not self.gamma_before:
            return {"sites": 0.0}
        before = sum(self.gamma_before.values()) / len(self.gamma_before)
        after = sum(self.gamma_after.values()) / len(self.gamma_after)
        return {
            "sites": float(len(self.sites)),
            "mean_gamma_before": before,
            "mean_gamma_after": after,
            "mean_alpha": sum(self.alphas.values()) / max(len(self.alphas), 1),
        }


def _is_norm(name: str, module: nn.Module) -> bool:
    if isinstance(module, nn.LayerNorm):
        return True
    if not hasattr(module, "weight") or module.weight is None:
        return False
    if getattr(module.weight, "ndim", 0) != 1:
        return False
    cls = type(module).__name__.lower()
    leaf = name.rsplit(".", 1)[-1].lower()
    return any(h in cls for h in _NORM_HINT) or any(h in leaf for h in _NORM_HINT)


@torch.no_grad()
def discover_sites(
    model: nn.Module, example_input: torch.Tensor, forward: callable | None = None
) -> list[SmoothSite]:
    """Find (normalization -> linear) fusion sites by tracing one forward pass.

    A linear layer is attached to a norm when the tensor it receives is the very
    tensor that norm produced, which is exactly the condition under which the
    inverse scale can be folded into the norm's weight.

    Args:
        model: The model to trace.
        example_input: One representative batch.
        forward: How to invoke the model; defaults to ``model(example_input)``.

    Returns:
        Sites with at least one attached linear layer, in module order.
    """
    norm_outputs: dict[int, str] = {}
    consumers: dict[str, list[str]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def norm_hook(name: str):
        def fn(_m: nn.Module, _i: tuple, out: torch.Tensor) -> None:
            if isinstance(out, torch.Tensor):
                norm_outputs[id(out)] = name

        return fn

    def linear_hook(name: str):
        def fn(_m: nn.Module, inputs: tuple, _o: torch.Tensor) -> None:
            x = inputs[0]
            if isinstance(x, tuple):
                x = x[0]
            src = norm_outputs.get(id(x))
            if src is not None:
                consumers.setdefault(src, []).append(name)

        return fn

    try:
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(linear_hook(name)))
            elif _is_norm(name, m):
                handles.append(m.register_forward_hook(norm_hook(name)))
        was_training = model.training
        model.eval()
        (forward or (lambda b: model(b)))(example_input)
        model.train(was_training)
    finally:
        for h in handles:
            h.remove()

    return [SmoothSite(norm=n, linears=ls) for n, ls in consumers.items() if ls]


@torch.no_grad()
def compute_smoothing_scale(
    act_scale: torch.Tensor,
    weights: Sequence[torch.Tensor],
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """SmoothQuant's per-input-channel scale ``s`` (SQ Eq. 4).

    Args:
        act_scale: Per-input-channel activation absmax, shape ``[in_features]``.
        weights: Weight matrices of every linear at the site, each ``[out, in]``.
            Their per-channel maxima are combined so all layers share one scale.
        alpha: Migration strength. ``0`` leaves activations untouched, ``1``
            moves all difficulty onto the weights. SmoothQuant defaults to 0.5
            (0.85 for the hardest models).
        eps: Floor on scales, matching upstream's ``clamp(min=1e-5)``.

    Returns:
        The scale vector ``s`` of shape ``[in_features]``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    w_scale = torch.stack([w.abs().amax(dim=0) for w in weights]).amax(dim=0)
    w_scale = w_scale.to(torch.float32).clamp_min(eps)
    a_scale = act_scale.to(torch.float32).clamp_min(eps)
    if a_scale.numel() != w_scale.numel():
        raise ValueError(
            f"activation scale has {a_scale.numel()} channels but weights have {w_scale.numel()}"
        )
    if alpha == 0.0:
        return torch.ones_like(w_scale)
    return (a_scale.pow(alpha) / w_scale.pow(1.0 - alpha)).clamp_min(eps)


@torch.no_grad()
def smooth_model(
    model: nn.Module,
    act_scales: dict[str, torch.Tensor],
    sites: Sequence[SmoothSite],
    alpha: float | dict[str, float] = 0.5,
    group_size: int = 128,
) -> SmoothReport:
    """Apply difficulty migration in place and report the effect on ``gamma``.

    Args:
        model: Model to modify in place.
        act_scales: Per-layer activation scales from
            :func:`slq.smooth.scales.collect_act_scales`.
        sites: Fusion sites from :func:`discover_sites`.
        alpha: A single migration strength, or one per site keyed by norm name.
            Sites absent from a dict fall back to 0.5.
        group_size: Group size used when measuring ``gamma``; reporting only.

    Returns:
        A :class:`SmoothReport` with the per-site mean ``gamma`` before and after.
    """
    modules = dict(model.named_modules())
    alphas: dict[str, float] = {}
    gamma_before: dict[str, float] = {}
    gamma_after: dict[str, float] = {}

    for site in sites:
        norm = modules.get(site.norm)
        linears = [modules[n] for n in site.linears if n in modules]
        if norm is None or not linears:
            continue

        # All linears at a site share the activation scale of the first one for
        # which we have statistics, mirroring SmoothQuant's fused-QKV treatment.
        act = next((act_scales[n] for n in site.linears if n in act_scales), None)
        if act is None:
            continue

        a = alpha.get(site.norm, 0.5) if isinstance(alpha, dict) else float(alpha)
        alphas[site.norm] = a
        if a == 0.0:
            continue

        weights = [m.weight.data for m in linears]
        dev, dtype = weights[0].device, weights[0].dtype
        gamma_before[site.norm] = float(
            torch.cat([centering_inefficiency(w.float(), group_size) for w in weights]).mean()
        )

        s = compute_smoothing_scale(act.to(dev), weights, alpha=a).to(dtype)

        norm.weight.data.div_(s)
        if getattr(norm, "bias", None) is not None:
            norm.bias.data.div_(s)
        for m in linears:
            m.weight.data.mul_(s.view(1, -1))

        gamma_after[site.norm] = float(
            torch.cat(
                [centering_inefficiency(m.weight.data.float(), group_size) for m in linears]
            ).mean()
        )

    return SmoothReport(
        sites=list(sites), alphas=alphas, gamma_before=gamma_before, gamma_after=gamma_after
    )
