"""GPTQ: the per-layer quantizer used throughout SLQ.

Implements Frantar et al. (2023) with group-wise asymmetric grids. SLQ is
orthogonal to this choice -- the allocation machinery only needs *some*
layer-wise quantizer -- but GPTQ is the paper's default because it is the de
facto standard for vLLM deployment.

The algorithm minimizes the layer-wise proxy objective

    argmin_{W_hat} || W X - W_hat X ||_F^2

by quantizing one input column at a time and compensating the induced error in
the not-yet-quantized columns using the inverse Hessian ``H^-1`` of
``H = 2 X X^T``.
"""

from __future__ import annotations

import math

import torch

from slq.quant.grid import QuantConfig, compute_qparams

__all__ = ["GPTQQuantizer", "gptq_quantize"]


class GPTQQuantizer:
    """Accumulates the layer Hessian, then quantizes the weight matrix.

    Typical use::

        gptq = GPTQQuantizer(linear.weight)
        for batch in calibration_batches:
            gptq.add_batch(batch)            # batch: [..., in_features]
        w_hat = gptq.quantize(QuantConfig(bits=4))
    """

    def __init__(self, weight: torch.Tensor) -> None:
        if weight.ndim != 2:
            raise ValueError(f"expected a 2-D weight matrix, got shape {tuple(weight.shape)}")
        self.weight = weight
        self.in_features = weight.shape[1]
        self.hessian = torch.zeros(
            (self.in_features, self.in_features), dtype=torch.float32, device=weight.device
        )
        self.n_samples = 0

    @torch.no_grad()
    def add_batch(self, inputs: torch.Tensor) -> None:
        """Accumulate ``H = 2 X X^T`` from a batch of layer inputs."""
        x = inputs.detach().reshape(-1, inputs.shape[-1]).to(torch.float32)
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"input has {x.shape[1]} features, layer expects {self.in_features}"
            )
        n_new = x.shape[0]
        if n_new == 0:
            return
        # Running mean of 2 X X^T so that batches of different sizes compose.
        self.hessian *= self.n_samples / (self.n_samples + n_new)
        self.n_samples += n_new
        x = x * math.sqrt(2.0 / self.n_samples)
        self.hessian += x.T @ x

    @torch.no_grad()
    def quantize(
        self,
        cfg: QuantConfig,
        block_size: int = 128,
        percdamp: float = 0.01,
        act_order: bool = False,
    ) -> torch.Tensor:
        """Run GPTQ and return the dequantized weights ``W_hat``.

        Args:
            cfg: Target quantizer configuration.
            block_size: Number of columns processed per lazy-update block.
            percdamp: Hessian damping as a fraction of ``mean(diag(H))``.
            act_order: Quantize columns in order of decreasing Hessian
                diagonal (activation order), which helps at low bitwidths.

        Returns:
            ``W_hat`` with the same shape and dtype as the original weight.
        """
        w = self.weight.detach().to(torch.float32).clone()
        out_features, n_cols = w.shape
        h = self.hessian.clone()

        # Columns whose inputs are always zero carry no signal; freeze them.
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0
        w[:, dead] = 0.0

        perm = inv_perm = None
        if act_order:
            perm = torch.argsort(torch.diag(h), descending=True)
            w = w[:, perm]
            h = h[perm][:, perm]
            inv_perm = torch.argsort(perm)

        damp = percdamp * torch.mean(torch.diag(h))
        h[range(n_cols), range(n_cols)] += damp

        # H^-1 via Cholesky; upper triangular factor of the inverse.
        h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
        h_inv = torch.linalg.cholesky(h_inv, upper=True)

        q = torch.zeros_like(w)
        gs = n_cols if cfg.group_size == -1 else cfg.group_size
        scale = zero = None

        for start in range(0, n_cols, block_size):
            end = min(start + block_size, n_cols)
            w_blk = w[:, start:end].clone()
            q_blk = torch.zeros_like(w_blk)
            err_blk = torch.zeros_like(w_blk)
            hinv_blk = h_inv[start:end, start:end]

            for j in range(end - start):
                col = start + j
                w_col = w_blk[:, j]
                d = hinv_blk[j, j]

                # Re-derive grid parameters at each group boundary, from the
                # error-compensated weights rather than the originals.
                if col % gs == 0:
                    g_end = min(col + gs, n_cols)
                    scale, zero = compute_qparams(w[:, col:g_end], _group_all(cfg, g_end - col))

                q_col = _quantize_column(w_col, scale, zero, cfg.levels)
                q_blk[:, j] = q_col

                # Propagate the rounding error to the remaining columns.
                err = (w_col - q_col) / d
                w_blk[:, j:] -= err.unsqueeze(1) @ hinv_blk[j, j:].unsqueeze(0)
                err_blk[:, j] = err

            q[:, start:end] = q_blk
            # Lazy batch update of all columns after this block.
            if end < n_cols:
                w[:, end:] -= err_blk @ h_inv[start:end, end:]

        if inv_perm is not None:
            q = q[:, inv_perm]
        return q.to(self.weight.dtype)

    def free(self) -> None:
        """Release the Hessian, which dominates memory for wide layers."""
        self.hessian = torch.zeros((0, 0))


def _group_all(cfg: QuantConfig, width: int) -> QuantConfig:
    """A per-group config that treats ``width`` columns as a single group."""
    return QuantConfig(bits=cfg.bits, group_size=width, symmetric=cfg.symmetric, fmt=cfg.fmt)


def _quantize_column(
    w_col: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, levels: int
) -> torch.Tensor:
    """Quantize-dequantize a single column against per-output-row grid params."""
    s = scale.reshape(-1)
    z = zero.reshape(-1)
    codes = torch.round(w_col / s + z).clamp_(0, levels - 1)
    return (codes - z) * s


@torch.no_grad()
def gptq_quantize(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    cfg: QuantConfig,
    block_size: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
) -> torch.Tensor:
    """One-shot convenience wrapper around :class:`GPTQQuantizer`."""
    gptq = GPTQQuantizer(weight)
    gptq.add_batch(inputs)
    return gptq.quantize(cfg, block_size=block_size, percdamp=percdamp, act_order=act_order)
