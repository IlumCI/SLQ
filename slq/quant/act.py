"""Activation quantizers for weight-and-activation (W+A) configurations.

SmoothQuant (Xiao et al., 2023) quantizes activations with a symmetric absmax
grid. SLQ's gamma-squared law (Lemma 3.2) says a zero-anchored grid inflates
noise variance by ``gamma^2 = (2M/R)^2`` whenever the distribution is offset
from zero -- which post-GELU/SiLU activations emphatically are, since they are
bounded below but not above. Asymmetric per-token grids are therefore offered
alongside the symmetric ones and are the default here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ActQuantConfig", "quantize_activation"]

_GRANULARITIES = ("per_token", "per_tensor", "per_channel")


@dataclass(frozen=True)
class ActQuantConfig:
    """Activation quantizer configuration.

    Args:
        bits: Bitwidth. ``None`` or ``16`` disables activation quantization.
        granularity: ``"per_token"`` (a grid per row of the flattened input,
            SmoothQuant's default), ``"per_tensor"`` (one grid for everything,
            the cheapest to serve), or ``"per_channel"`` (a grid per feature).
        symmetric: Zero-anchored grid. ``False`` (asymmetric) avoids the
            ``gamma^2`` variance penalty of Lemma 3.2.
    """

    bits: int | None = 8
    granularity: str = "per_token"
    symmetric: bool = False

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise ValueError(
                f"granularity must be one of {_GRANULARITIES}, got {self.granularity!r}"
            )
        if self.bits is not None and not 2 <= self.bits <= 16:
            raise ValueError(f"bits must be in [2, 16] or None, got {self.bits}")

    @property
    def enabled(self) -> bool:
        return self.bits is not None and self.bits < 16


def _reduce_dims(x: torch.Tensor, granularity: str) -> tuple[int, ...] | None:
    if granularity == "per_token":
        return (-1,)
    if granularity == "per_channel":
        return tuple(range(x.ndim - 1))
    return None  # per_tensor: reduce everything


@torch.no_grad()
def quantize_activation(
    x: torch.Tensor, cfg: ActQuantConfig, eps: float = 1e-5
) -> torch.Tensor:
    """Fake-quantize an activation tensor.

    Args:
        x: Activation tensor, quantized along the granularity implied by ``cfg``.
        cfg: Activation quantizer configuration.
        eps: Floor on the scale, matching SmoothQuant's ``clamp(min=1e-5)``.

    Returns:
        The quantize-dequantize round trip of ``x``, same shape and dtype.
    """
    if not cfg.enabled:
        return x

    orig_dtype = x.dtype
    xf = x.float()
    dims = _reduce_dims(xf, cfg.granularity)
    n = 1 << cfg.bits

    if cfg.symmetric:
        # SmoothQuant's grid: signed absmax over [-q_max, q_max].
        q_max = float(2 ** (cfg.bits - 1) - 1)
        if dims is None:
            scale = xf.abs().max()
        else:
            scale = xf.abs().amax(dim=dims, keepdim=True)
        scale = scale.clamp_min(eps) / q_max
        out = (xf / scale).round_().clamp_(-q_max - 1, q_max) * scale
        return out.to(orig_dtype)

    # Asymmetric: the grid spans exactly [L, U] via a zero-point.
    if dims is None:
        lo, hi = xf.min(), xf.max()
    else:
        lo = xf.amin(dim=dims, keepdim=True)
        hi = xf.amax(dim=dims, keepdim=True)
    lo = torch.minimum(lo, torch.zeros_like(lo))
    hi = torch.maximum(hi, torch.zeros_like(hi))
    scale = ((hi - lo) / (n - 1)).clamp_min(eps)
    zero = torch.round(-lo / scale).clamp_(0, n - 1)
    out = ((xf / scale + zero).round_().clamp_(0, n - 1) - zero) * scale
    return out.to(orig_dtype)
