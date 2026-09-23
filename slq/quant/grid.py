"""Scalar affine quantization grids (Section 3.1 of the paper).

Implements the symmetric / asymmetric group-wise affine quantizers whose
geometry underpins the gamma-squared variance law (Lemma 3.2):

    gamma          = 2M / R                         (Definition 3.1)
    Delta_asym     = R / (n - 1)                    (Eq. 10)
    Delta_sym      = 2M / (n - 1)                   (Eq. 11)
    Delta_sym      = gamma * Delta_asym             (Lemma B.1)
    sigma^2        = Delta^2 / 12                   (Lemma B.2, Bennett high-rate)
    sigma^2_sym    = gamma^2 * sigma^2_asym         (Lemma 3.2 / Eq. 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "QuantConfig",
    "centering_inefficiency",
    "compute_qparams",
    "dequantize",
    "effective_bits",
    "fake_quantize",
    "quantize",
    "step_size",
    "theoretical_noise_variance",
]


@dataclass(frozen=True)
class QuantConfig:
    """Configuration of a scalar affine quantizer.

    Args:
        bits: Nominal bitwidth ``b``; the grid has ``n = 2**b`` levels.
        group_size: Number of consecutive input-channel weights sharing one
            (scale, zero-point) pair. ``-1`` means per-output-channel.
        symmetric: If ``True`` the grid is anchored at zero and spans
            ``[-M, M]``; otherwise it spans exactly ``[L, U]`` via a zero-point.
        fmt: ``"int"`` for integer grids, ``"fp"`` for NVFP-style grids. Only
            affects the storage-overhead accounting in :func:`effective_bits`.
    """

    bits: int
    group_size: int = 128
    symmetric: bool = False
    fmt: str = "int"

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 16:
            raise ValueError(f"bits must be in [1, 16], got {self.bits}")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"group_size must be -1 or positive, got {self.group_size}")
        if self.fmt not in ("int", "fp"):
            raise ValueError(f"fmt must be 'int' or 'fp', got {self.fmt!r}")

    @property
    def levels(self) -> int:
        """Number of representable grid points ``n = 2**b``."""
        return 1 << self.bits


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #


def _reshape_groups(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    """Reshape ``[out, in]`` weights into ``[out * n_groups, group_size]``."""
    if w.ndim != 2:
        raise ValueError(f"expected a 2-D weight matrix, got shape {tuple(w.shape)}")
    out_features, in_features = w.shape
    gs = in_features if group_size == -1 else group_size
    if in_features % gs != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={gs}; "
            "pad the layer or choose a compatible group size"
        )
    return w.reshape(out_features * (in_features // gs), gs), gs


def centering_inefficiency(
    w: torch.Tensor, group_size: int = 128, eps: float = 1e-12
) -> torch.Tensor:
    """Per-group centering inefficiency ``gamma = 2M / R`` (Definition 3.1).

    ``gamma == 1`` for groups symmetric about zero and ``gamma > 1`` otherwise,
    which is the regime in which symmetric quantization wastes grid capacity.

    Returns:
        A 1-D tensor holding one gamma per quantization group.
    """
    g, _ = _reshape_groups(w, group_size)
    lo = g.min(dim=1).values
    hi = g.max(dim=1).values
    rng = (hi - lo).clamp_min(eps)
    max_abs = torch.maximum(lo.abs(), hi.abs())
    return 2.0 * max_abs / rng


def step_size(w: torch.Tensor, cfg: QuantConfig, eps: float = 1e-12) -> torch.Tensor:
    """Per-group quantizer step size ``Delta`` (Eq. 10 / Eq. 11)."""
    g, _ = _reshape_groups(w, cfg.group_size)
    lo = g.min(dim=1).values
    hi = g.max(dim=1).values
    n = cfg.levels
    if cfg.symmetric:
        max_abs = torch.maximum(lo.abs(), hi.abs())
        return (2.0 * max_abs / (n - 1)).clamp_min(eps)
    return ((hi - lo) / (n - 1)).clamp_min(eps)


def theoretical_noise_variance(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """High-rate noise variance ``sigma^2 = Delta^2 / 12`` (Lemma B.2)."""
    return step_size(w, cfg).pow(2) / 12.0


# --------------------------------------------------------------------------- #
# Quantizer
# --------------------------------------------------------------------------- #


def compute_qparams(
    w: torch.Tensor, cfg: QuantConfig, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive per-group ``(scale, zero_point)`` from a weight matrix.

    The asymmetric grid spans exactly ``[L, U]``; the symmetric grid is anchored
    at zero and spans ``[-M, M]`` with a fixed mid-grid zero-point.

    Returns:
        ``(scale, zero_point)``, each of shape ``[out * n_groups, 1]``. The
        zero-point is an integer-valued float tensor.
    """
    g, _ = _reshape_groups(w, cfg.group_size)
    n = cfg.levels
    lo = g.min(dim=1, keepdim=True).values
    hi = g.max(dim=1, keepdim=True).values

    if cfg.symmetric:
        max_abs = torch.maximum(lo.abs(), hi.abs()).clamp_min(eps)
        scale = (2.0 * max_abs / (n - 1)).clamp_min(eps)
        zero = torch.full_like(scale, float((n - 1) // 2))
        return scale, zero

    # Always include 0 in the represented interval so that exact zeros survive.
    lo = torch.minimum(lo, torch.zeros_like(lo))
    hi = torch.maximum(hi, torch.zeros_like(hi))
    scale = ((hi - lo) / (n - 1)).clamp_min(eps)
    zero = torch.round(-lo / scale).clamp_(0, n - 1)
    return scale, zero


def quantize(w: torch.Tensor, cfg: QuantConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to integer codes.

    Returns:
        ``(codes, scale, zero_point)`` where ``codes`` has the shape of ``w``
        and holds integers in ``[0, 2**bits - 1]``.
    """
    scale, zero = compute_qparams(w, cfg)
    g, gs = _reshape_groups(w, cfg.group_size)
    codes = torch.round(g / scale + zero).clamp_(0, cfg.levels - 1)
    return codes.reshape(w.shape), scale, zero


def dequantize(
    codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, cfg: QuantConfig
) -> torch.Tensor:
    """Inverse of :func:`quantize`."""
    g, _ = _reshape_groups(codes, cfg.group_size)
    return ((g - zero) * scale).reshape(codes.shape)


def fake_quantize(w: torch.Tensor, cfg: QuantConfig) -> torch.Tensor:
    """Quantize-dequantize round trip, returning ``W_hat`` in the dtype of ``w``."""
    codes, scale, zero = quantize(w.float(), cfg)
    return dequantize(codes, scale, zero, cfg).to(w.dtype)


# --------------------------------------------------------------------------- #
# Storage accounting (Table 7)
# --------------------------------------------------------------------------- #

# Overhead per quantization group, in bits, reproducing Table 7 exactly:
#   INT g=128 asym -> 16-bit scale + 4-bit zero-point  = 20 bits -> +0.15625
#   INT g=128 sym  -> 16-bit scale                     = 16 bits -> +0.125
#   FP  g=16  asym -> 8-bit scale  + 8-bit zero-point  = 16 bits -> +1.0
#   FP  g=16  sym  -> 8-bit scale                      =  8 bits -> +0.5
_SCALE_BITS = {"int": 16, "fp": 8}
_ZERO_BITS = {"int": 4, "fp": 8}


def effective_bits(cfg: QuantConfig, in_features: int | None = None) -> float:
    """Realized bits per parameter including scale / zero-point overhead.

    Reproduces Table 7 of the paper. ``fp`` grids at 8 bits are treated as a
    natively 8-bit format carrying no per-group overhead.

    Args:
        cfg: The quantizer configuration.
        in_features: Needed only when ``cfg.group_size == -1`` (per-channel),
            where the group spans the full input dimension.
    """
    if cfg.fmt == "fp" and cfg.bits == 8:
        return 8.0
    if cfg.group_size == -1:
        if in_features is None:
            raise ValueError("in_features is required for per-channel group_size=-1")
        gs = in_features
    else:
        gs = cfg.group_size
    overhead = _SCALE_BITS[cfg.fmt] + (0 if cfg.symmetric else _ZERO_BITS[cfg.fmt])
    return cfg.bits + overhead / gs
