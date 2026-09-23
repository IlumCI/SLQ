"""The object the SLQ search operates on: a model whose per-group bitwidths
can be reassigned cheaply and whose output fidelity can be measured.

Responsibilities:

1. Capture GPTQ Hessians on a calibration set (one pass).
2. Populate a :class:`~slq.model.bank.WeightBank` with every (layer, bitwidth)
   pair, once.
3. Apply an arbitrary assignment of bitwidths to groups in O(number of layers)
   tensor copies -- no re-quantization.
4. Evaluate EAR / KL against the cached full-precision reference logits.

Step 3 is what makes Algorithm 1 tractable: the Shapley sweep needs
``O(P * M * |B|)`` configurations, and each one must cost a forward pass, not a
quantization run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from slq.metrics.fidelity import DEFAULT_TOPK, FidelityMeter, FidelityResult
from slq.model.bank import WeightBank
from slq.model.grouping import GroupingPolicy, LayerGroup, build_groups
from slq.quant.act import ActQuantConfig, quantize_activation
from slq.quant.grid import effective_bits

__all__ = ["QuantizableModel", "CalibrationData"]

CalibrationData = Sequence[torch.Tensor]


@dataclass
class _LayerRecord:
    name: str
    module: nn.Linear
    fp_weight: torch.Tensor  # full-precision master copy
    numel: int


class QuantizableModel:
    """Wraps an ``nn.Module`` for mixed-precision bitwidth search.

    Args:
        model: The model to quantize. Modified in place when assignments are
            applied; the original weights are retained and restorable.
        calibration: Calibration batches. Used for Hessians, reference logits
            and all fidelity measurements.
        policy: How layers are partitioned into allocation groups.
        bank: Quantized-weight cache. One is created if omitted.
        forward: How to invoke the model on a batch; defaults to ``model(batch)``.
        topk: Truncation ``K`` for EAR and KL (10 in the paper).
        act_quant: Activation quantizer for W+A configurations. Disabled by
            default, matching the paper's weight-only main results.
        master_dtype: Dtype for the retained full-precision master weights.
            These are a second copy of every quantizable weight, so on a model
            near the memory limit they are worth storing in half precision:
            ``torch.float16`` halves their cost at the price of a small
            round-trip error when restoring. ``None`` keeps the model's dtype.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration: CalibrationData,
        policy: GroupingPolicy | None = None,
        bank: WeightBank | None = None,
        forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
        topk: int = DEFAULT_TOPK,
        act_quant: ActQuantConfig | None = None,
        master_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.calibration = list(calibration)
        if not self.calibration:
            raise ValueError("at least one calibration batch is required")
        self.policy = policy if policy is not None else GroupingPolicy()
        # An empty WeightBank is falsy (__len__ == 0), so `bank or WeightBank()`
        # would silently discard a caller's freshly-built bank and substitute a
        # default one -- different bitwidths, group size and quantizer than
        # asked for. Test identity explicitly.
        self.bank = bank if bank is not None else WeightBank()
        self.forward_fn = forward if forward is not None else (lambda b: model(b))
        self.topk = topk
        self.act_quant = act_quant if act_quant is not None else ActQuantConfig(bits=None)

        self.model.eval()
        self.master_dtype = master_dtype
        self._layers: dict[str, _LayerRecord] = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and not self.policy.is_excluded(name):
                master = m.weight.detach().clone()
                if master_dtype is not None:
                    master = master.to(master_dtype)
                self._layers[name] = _LayerRecord(
                    name=name,
                    module=m,
                    fp_weight=master,
                    numel=m.weight.numel(),
                )
        if not self._layers:
            raise ValueError("no quantizable nn.Linear layers found after exclusions")

        self.groups: list[LayerGroup] = build_groups(
            ((n, r.numel) for n, r in self._layers.items()), self.policy
        )
        self._group_of: dict[str, str] = {
            layer: g.name for g in self.groups for layer in g.layers
        }
        self._current: dict[str, int | None] = {g.name: None for g in self.groups}
        self._reference_logits: list[torch.Tensor] | None = None
        self._act_hooks: list[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def layer_names(self) -> list[str]:
        return list(self._layers)

    @property
    def group_names(self) -> list[str]:
        return [g.name for g in self.groups]

    @property
    def total_params(self) -> int:
        return sum(g.numel for g in self.groups)

    def group_numel(self) -> dict[str, int]:
        return {g.name: g.numel for g in self.groups}

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    def memory_report(self) -> dict[str, float]:
        """Approximate resident bytes, which is what decides feasibility.

        Three copies of the weights coexist during a search: the model's own
        parameters, the full-precision masters kept for restoration, and the
        bank's integer codes (one byte per parameter per candidate bitwidth).
        On a 1.7B model at four candidate bitwidths that is roughly
        6.8 + 5.6 + 5.6 GB, which will not fit in 16 GB. Check this before
        starting a long run rather than discovering it through an OOM kill.
        """
        params = sum(r.numel for r in self._layers.values())
        model_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        master_bytes = sum(
            r.fp_weight.numel() * r.fp_weight.element_size()
            for r in self._layers.values()
        )
        bank_bytes = params * len(self.bank.bitwidths)  # uint8 codes
        return {
            "model_gb": model_bytes / 1e9,
            "masters_gb": master_bytes / 1e9,
            "bank_gb_estimated": bank_bytes / 1e9,
            "total_gb_estimated": (model_bytes + master_bytes + bank_bytes) / 1e9,
        }

    @torch.no_grad()
    def capture_hessians(self) -> dict[str, torch.Tensor]:
        """Accumulate ``H = 2 X X^T`` for every quantizable layer in one pass.

        Memory warning: this holds one ``[in, in]`` float32 matrix per layer
        simultaneously. On a model with wide MLPs that is gigabytes; use
        ``method="rtn"`` on the bank if it does not fit.
        """
        from slq.quant.gptq import GPTQQuantizer

        quantizers = {n: GPTQQuantizer(r.fp_weight.float()) for n, r in self._layers.items()}
        handles = []

        def hook(name: str):
            def fn(_m: nn.Module, inputs: tuple, _o: torch.Tensor) -> None:
                x = inputs[0]
                if isinstance(x, tuple):
                    x = x[0]
                quantizers[name].add_batch(x)

            return fn

        try:
            for name, rec in self._layers.items():
                handles.append(rec.module.register_forward_hook(hook(name)))
            for batch in self.calibration:
                self.forward_fn(batch)
        finally:
            for h in handles:
                h.remove()
        return {n: q.hessian for n, q in quantizers.items()}

    @torch.no_grad()
    def build_bank(self, hessians: dict[str, torch.Tensor] | None = None) -> WeightBank:
        """Quantize every layer at every candidate bitwidth into the bank.

        Args:
            hessians: Pre-computed Hessians. Captured automatically when the
                bank's method is ``"gptq"`` and none are supplied.

        Returns:
            The populated bank.
        """
        if self.bank.method == "gptq" and hessians is None:
            hessians = self.capture_hessians()
        for name, rec in self._layers.items():
            h = hessians.get(name) if hessians else None
            self.bank.add_layer(name, rec.fp_weight.float(), hessian=h)
            if h is not None:
                hessians[name] = torch.zeros((0, 0))  # free as we go
        return self.bank

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def apply(self, assignment: dict[str, int | None]) -> None:
        """Set each group's bitwidth. ``None`` restores full precision.

        Only groups whose bitwidth actually changed are touched, so walking a
        Shapley permutation costs one layer update per step.
        """
        for gname, bits in assignment.items():
            if gname not in self._current:
                raise KeyError(f"unknown group {gname!r}")
            if self._current[gname] == bits:
                continue
            for layer in self._group_for(gname):
                rec = self._layers[layer]
                if bits is None:
                    rec.module.weight.data.copy_(rec.fp_weight.to(rec.module.weight.dtype))
                else:
                    q = self.bank.get(layer, bits)
                    rec.module.weight.data.copy_(
                        q.materialize(rec.module.weight.device, rec.module.weight.dtype)
                    )
            self._current[gname] = bits

    def _group_for(self, gname: str) -> list[str]:
        for g in self.groups:
            if g.name == gname:
                return g.layers
        raise KeyError(f"unknown group {gname!r}")

    def restore(self) -> None:
        """Return every layer to full precision."""
        self.apply({g.name: None for g in self.groups})

    def __enter__(self) -> QuantizableModel:
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore full precision, so a wrapped model is safe to reuse."""
        self.restore()

    def uniform(self, bits: int | None) -> dict[str, int | None]:
        """An assignment placing every group at the same bitwidth."""
        return {g.name: bits for g in self.groups}

    def current_assignment(self) -> dict[str, int | None]:
        return dict(self._current)

    # ------------------------------------------------------------------ #
    # Bitwidth accounting
    # ------------------------------------------------------------------ #

    def average_bits(self, assignment: dict[str, int | None], effective: bool = True) -> float:
        """Parameter-weighted average bitwidth ``b_bar`` (Section 3.2).

        Args:
            assignment: Group-to-bitwidth mapping; ``None`` counts as 16 bits.
            effective: Include per-group scale / zero-point overhead, matching
                the paper's reported bits per parameter (Table 7).
        """
        total = self.total_params
        if total == 0:
            return float("nan")
        acc = 0.0
        for g in self.groups:
            bits = assignment.get(g.name)
            if bits is None:
                acc += 16.0 * g.numel
            elif effective:
                acc += effective_bits(self.bank.config(bits)) * g.numel
            else:
                acc += float(bits) * g.numel
        return acc / total

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _logits(self, batches: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        self._install_act_hooks()
        try:
            return [self.forward_fn(b).detach().float() for b in batches]
        finally:
            self._remove_act_hooks()

    def _install_act_hooks(self) -> None:
        """Fake-quantize layer inputs for W+A configurations."""
        if not self.act_quant.enabled:
            return
        cfg = self.act_quant

        def pre_hook(_m: nn.Module, inputs: tuple):
            x = inputs[0]
            if isinstance(x, tuple):
                x = x[0]
            return (quantize_activation(x, cfg),) + inputs[1:]

        for rec in self._layers.values():
            self._act_hooks.append(rec.module.register_forward_pre_hook(pre_hook))

    def _remove_act_hooks(self) -> None:
        for h in self._act_hooks:
            h.remove()
        self._act_hooks.clear()

    @torch.no_grad()
    def reference_logits(self, recompute: bool = False) -> list[torch.Tensor]:
        """Full-precision logits on the calibration set, cached.

        Activation quantization is disabled while these are taken, so the
        reference is the true BF16/FP32 model in every configuration.
        """
        if self._reference_logits is None or recompute:
            saved_assignment = self.current_assignment()
            saved_act = self.act_quant
            self.act_quant = ActQuantConfig(bits=None)
            self.restore()
            try:
                self._reference_logits = self._logits(self.calibration)
            finally:
                self.act_quant = saved_act
                self.apply(saved_assignment)
        return self._reference_logits

    @torch.no_grad()
    def evaluate(
        self, assignment: dict[str, int | None] | None = None, max_batches: int | None = None
    ) -> FidelityResult:
        """Measure EAR, KL, flip rate and disagreement margin.

        Args:
            assignment: Applied before evaluating; the current state is used
                if omitted.
            max_batches: Evaluate on a prefix of the calibration set. The
                evolutionary search uses this for multi-stage selection.

        Returns:
            Fidelity of the quantized model against the full-precision reference.
        """
        if assignment is not None:
            self.apply(assignment)
        ref = self.reference_logits()
        batches = self.calibration if max_batches is None else self.calibration[:max_batches]
        ref = ref if max_batches is None else ref[:max_batches]

        meter = FidelityMeter(topk=self.topk)
        for r, cand in zip(ref, self._logits(batches), strict=True):
            meter.update(r, cand)
        return meter.compute()
