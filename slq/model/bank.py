"""Pre-computed quantized weights for every (layer, bitwidth) pair.

The sensitivity sweep of Algorithm 1 evaluates ``O(P * M * |B|)`` mixed-precision
configurations. Re-running GPTQ for each of them would be prohibitive, so every
layer is quantized once per candidate bitwidth and the resulting codes are
cached here. Storage is dominated by the integer codes, so holding all of
``B = {2..8}`` costs roughly ``sum(B)/16 ~ 2.2x`` the FP16 model rather than
``|B|x``; the bank can also be parked on CPU or on disk.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from slq.quant.gptq import GPTQQuantizer
from slq.quant.grid import QuantConfig, dequantize, quantize

__all__ = ["QuantizedWeight", "WeightBank"]


@dataclass
class QuantizedWeight:
    """Integer codes plus grid parameters for one layer at one bitwidth."""

    codes: torch.Tensor  # uint8 for b <= 8, int16 otherwise
    scale: torch.Tensor
    zero: torch.Tensor
    cfg: QuantConfig
    shape: tuple[int, ...]

    def materialize(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct ``W_hat`` on the requested device and dtype."""
        codes = self.codes.to(device=device, dtype=torch.float32)
        scale = self.scale.to(device=device, dtype=torch.float32)
        zero = self.zero.to(device=device, dtype=torch.float32)
        return dequantize(codes, scale, zero, self.cfg).to(dtype)

    @property
    def nbytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero.numel() * self.zero.element_size()
        )


class WeightBank:
    """Caches quantized weights keyed by ``(layer_name, bits)``.

    Args:
        bitwidths: Candidate bitwidths to populate, e.g. ``(2, ..., 8)``.
        group_size: Quantization group size (128 in the paper's setup).
        symmetric: Use symmetric grids. The paper shows asymmetric is a
            prerequisite for distribution-lossless fidelity (Section 3.1).
        fmt: ``"int"`` or ``"fp"``; affects storage accounting only.
        store_device: Where cached codes live (``"cpu"`` keeps accelerator
            memory free).
        method: ``"gptq"`` (paper default) or ``"rtn"`` (round-to-nearest).
    """

    def __init__(
        self,
        bitwidths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
        group_size: int = 128,
        symmetric: bool = False,
        fmt: str = "int",
        store_device: str = "cpu",
        method: str = "gptq",
        gptq_block_size: int = 128,
        gptq_percdamp: float = 0.01,
        gptq_act_order: bool = False,
    ) -> None:
        if method not in ("gptq", "rtn"):
            raise ValueError(f"method must be 'gptq' or 'rtn', got {method!r}")
        self.bitwidths = tuple(sorted(bitwidths))
        self.group_size = group_size
        self.symmetric = symmetric
        self.fmt = fmt
        self.store_device = store_device
        self.method = method
        self.gptq_block_size = gptq_block_size
        self.gptq_percdamp = gptq_percdamp
        self.gptq_act_order = gptq_act_order
        self._store: dict[tuple[str, int], QuantizedWeight] = {}

    def config(self, bits: int) -> QuantConfig:
        return QuantConfig(
            bits=bits, group_size=self.group_size, symmetric=self.symmetric, fmt=self.fmt
        )

    # ------------------------------------------------------------------ #
    # Population
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def add_layer(
        self,
        name: str,
        weight: torch.Tensor,
        hessian: torch.Tensor | None = None,
        bitwidths: Iterable[int] | None = None,
    ) -> None:
        """Quantize one layer at every candidate bitwidth and cache the result.

        Args:
            name: Layer identifier.
            weight: The FP weight matrix ``[out, in]``.
            hessian: Optional GPTQ Hessian ``2 X X^T``. Required for
                ``method="gptq"``; ignored for round-to-nearest.
            bitwidths: Override the bank's default bitwidth set.
        """
        w = weight.detach().to(torch.float32)
        targets = tuple(bitwidths) if bitwidths is not None else self.bitwidths

        for b in targets:
            cfg = self.config(b)
            if self.method == "gptq":
                if hessian is None:
                    raise ValueError(
                        f"layer {name!r}: method='gptq' requires a Hessian; "
                        "run a calibration pass first or use method='rtn'"
                    )
                gptq = GPTQQuantizer(w)
                gptq.hessian = hessian.to(torch.float32)
                gptq.n_samples = 1
                w_hat = gptq.quantize(
                    cfg,
                    block_size=self.gptq_block_size,
                    percdamp=self.gptq_percdamp,
                    act_order=self.gptq_act_order,
                )
                # Re-encode W_hat onto the grid so only codes need storing.
                codes, scale, zero = quantize(w_hat, cfg)
            else:
                codes, scale, zero = quantize(w, cfg)

            self._store[(name, b)] = QuantizedWeight(
                codes=codes.to(device=self.store_device, dtype=_code_dtype(b)),
                scale=scale.to(device=self.store_device, dtype=torch.float32),
                zero=zero.to(device=self.store_device, dtype=torch.float32),
                cfg=cfg,
                shape=tuple(w.shape),
            )

    def get(self, name: str, bits: int) -> QuantizedWeight:
        try:
            return self._store[(name, bits)]
        except KeyError:
            raise KeyError(
                f"no cached weight for layer {name!r} at {bits} bits; "
                f"available bitwidths: {sorted({b for (n, b) in self._store if n == name})}"
            ) from None

    def has(self, name: str, bits: int) -> bool:
        return (name, bits) in self._store

    @property
    def layers(self) -> list[str]:
        return sorted({n for (n, _) in self._store})

    @property
    def nbytes(self) -> int:
        return sum(q.nbytes for q in self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serialize the bank to a ``.pt`` file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "meta": {
                "bitwidths": self.bitwidths,
                "group_size": self.group_size,
                "symmetric": self.symmetric,
                "fmt": self.fmt,
                "method": self.method,
            },
            "store": {
                f"{name}|{bits}": {
                    "codes": q.codes,
                    "scale": q.scale,
                    "zero": q.zero,
                    "bits": q.cfg.bits,
                    "shape": q.shape,
                }
                for (name, bits), q in self._store.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, store_device: str = "cpu") -> WeightBank:
        """Restore a bank written by :meth:`save`."""
        payload = torch.load(path, map_location=store_device, weights_only=False)
        meta = payload["meta"]
        bank = cls(
            bitwidths=meta["bitwidths"],
            group_size=meta["group_size"],
            symmetric=meta["symmetric"],
            fmt=meta["fmt"],
            store_device=store_device,
            method=meta["method"],
        )
        for key, rec in payload["store"].items():
            name, bits_s = key.rsplit("|", 1)
            bits = int(bits_s)
            bank._store[(name, bits)] = QuantizedWeight(
                codes=rec["codes"],
                scale=rec["scale"],
                zero=rec["zero"],
                cfg=bank.config(bits),
                shape=tuple(rec["shape"]),
            )
        return bank


def _code_dtype(bits: int) -> torch.dtype:
    return torch.uint8 if bits <= 8 else torch.int16
