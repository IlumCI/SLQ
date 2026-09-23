"""Activation-scale collection for SmoothQuant-style migration.

Follows Xiao et al. (2023): for each linear layer, record the per-input-channel
maximum absolute activation over a calibration set,

    act_scale[j] = max over calibration tokens of |X[:, j]|.

Upstream SmoothQuant hardcodes the module layout of each supported
architecture; this collector instead hooks every ``nn.Linear`` and is therefore
model-agnostic.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

__all__ = ["ActScaleCollector", "collect_act_scales"]


class ActScaleCollector:
    """Hooks every ``nn.Linear`` and accumulates per-input-channel absmax.

    Example::

        with ActScaleCollector(model) as c:
            for batch in calib:
                model(batch)
        scales = c.scales
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float32) -> None:
        self.model = model
        self.dtype = dtype
        self.scales: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, name: str):
        def fn(_module: nn.Module, inputs: tuple, _output: torch.Tensor) -> None:
            x = inputs[0]
            if isinstance(x, tuple):
                x = x[0]
            flat = x.detach().reshape(-1, x.shape[-1]).abs().to(self.dtype)
            cur = flat.amax(dim=0).cpu()
            prev = self.scales.get(name)
            self.scales[name] = cur if prev is None else torch.maximum(prev, cur)

        return fn

    def __enter__(self) -> ActScaleCollector:
        for name, m in self.model.named_modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def collect_act_scales(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    forward: callable | None = None,
) -> dict[str, torch.Tensor]:
    """Run calibration batches and return per-layer activation scales.

    Args:
        model: The model to instrument.
        batches: Calibration inputs, passed to ``forward``.
        forward: How to invoke the model on a batch; defaults to ``model(batch)``.

    Returns:
        Maps each linear layer's name to its per-input-channel absmax vector.
    """
    fwd = forward or (lambda b: model(b))
    was_training = model.training
    model.eval()
    try:
        with ActScaleCollector(model) as c:
            for batch in batches:
                fwd(batch)
            return dict(c.scales)
    finally:
        model.train(was_training)
