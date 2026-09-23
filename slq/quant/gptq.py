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

import contextlib
import math

import torch

from slq.quant.grid import QuantConfig, compute_qparams

__all__ = ["GPTQQuantizer", "gptq_quantize"]


@contextlib.contextmanager
def _serial_ops():
    """Run PyTorch single-threaded for the duration.

    GPTQ's error compensation is inherently sequential: one column at a time,
    each doing a rank-1 update of a few hundred kilobytes. At that size the cost
    of dispatching work to a thread pool dominates the arithmetic, and on a
    container with few cores the threads contend badly enough to slow the loop
    by two orders of magnitude -- measured on this machine, one 256x512 layer
    takes 0.04s single-threaded against 4.3s on four threads. The surrounding
    matrix work (the Cholesky in :meth:`GPTQQuantizer.prepare` and the blocked
    lazy update) is large enough to benefit from threads, so only the column
    loop is pinned.
    """
    prev = torch.get_num_threads()
    if prev > 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if prev > 1:
            torch.set_num_threads(prev)


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
        self._prepared: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None
        self._prepared_key: tuple[float, bool] | None = None

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
    def prepare(
        self, percdamp: float = 0.01, act_order: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Factorize the Hessian once, for reuse across bitwidths.

        The inverse-Hessian factor depends only on the calibration statistics,
        not on the target bitwidth, so quantizing one layer at every candidate
        bitwidth -- which is exactly what the weight bank does -- can share a
        single Cholesky. On a 3072-wide layer that factorization dominates, so
        caching it cuts bank construction roughly in proportion to ``|B|``.

        Args:
            percdamp: Hessian damping as a fraction of ``mean(diag(H))``.
            act_order: Quantize columns in order of decreasing Hessian diagonal.

        Returns:
            ``(h_inv, perm, inv_perm)``; the permutations are ``None`` unless
            ``act_order`` is set.
        """
        key = (percdamp, act_order)
        if self._prepared is not None and self._prepared_key == key:
            return self._prepared

        n_cols = self.in_features
        h = self.hessian.clone()
        dead = torch.diag(h) == 0
        h[dead, dead] = 1.0

        perm = inv_perm = None
        if act_order:
            perm = torch.argsort(torch.diag(h), descending=True)
            h = h[perm][:, perm]
            inv_perm = torch.argsort(perm)

        damp = percdamp * torch.mean(torch.diag(h))
        h[range(n_cols), range(n_cols)] += damp

        h_inv = torch.cholesky_inverse(torch.linalg.cholesky(h))
        h_inv = torch.linalg.cholesky(h_inv, upper=True)

        self._prepared = (h_inv, perm, inv_perm)
        self._prepared_key = key
        return self._prepared

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
        _out_features, n_cols = w.shape

        # Columns whose inputs are always zero carry no signal; freeze them.
        dead = torch.diag(self.hessian) == 0
        w[:, dead] = 0.0

        h_inv, perm, inv_perm = self.prepare(percdamp=percdamp, act_order=act_order)
        if perm is not None:
            w = w[:, perm]

        q = torch.zeros_like(w)
        gs = n_cols if cfg.group_size == -1 else cfg.group_size
        scale = zero = None

        for start in range(0, n_cols, block_size):
            end = min(start + block_size, n_cols)
            w_blk = w[:, start:end].clone()
            q_blk = torch.zeros_like(w_blk)
            err_blk = torch.zeros_like(w_blk)
            hinv_blk = h_inv[start:end, start:end]

            with _serial_ops():
                for j in range(end - start):
                    col = start + j
                    w_col = w_blk[:, j]
                    d = hinv_blk[j, j]

                    # Re-derive grid parameters at each group boundary, from the
                    # error-compensated weights rather than the originals.
                    if col % gs == 0:
                        g_end = min(col + gs, n_cols)
                        scale, zero = compute_qparams(
                            w[:, col:g_end], _group_all(cfg, g_end - col)
                        )

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
        """Release the Hessian and its factorization."""
        self.hessian = torch.zeros((0, 0))
        self._prepared = None
        self._prepared_key = None


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
