"""End-to-end SLQ pipeline, with optional SmoothQuant difficulty migration.

The full sequence:

1. **Smooth** (optional) -- collect per-channel activation scales, discover
   normalization/linear fusion sites by tracing, and migrate activation
   difficulty into the weights. Required for usable W+A configurations;
   a no-op for weight-only ones unless explicitly enabled.
2. **Calibrate** -- capture GPTQ Hessians in one pass.
3. **Bank** -- quantize every layer at every candidate bitwidth, once.
4. **Estimate** -- build the sensitivity database (Shapley or linear).
5. **Search** -- find the minimum average bitwidth meeting a DL or TL target.
6. **Apply** -- install the winning assignment and verify it by measurement.

Step 3 is what keeps step 4 affordable: applying a configuration is a tensor
copy, not a re-quantization.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from slq.alloc.search import (
    BenchmarkFn,
    SearchResult,
    search_distribution_lossless,
    search_task_lossless,
)
from slq.metrics.fidelity import FidelityResult
from slq.model.bank import WeightBank
from slq.model.grouping import GroupingPolicy
from slq.model.wrapper import QuantizableModel
from slq.quant.act import ActQuantConfig
from slq.quant.grid import QuantConfig, effective_bits
from slq.sensitivity.database import SensitivityDatabase
from slq.sensitivity.linear import linear_sensitivity
from slq.sensitivity.shapley import shapley_sensitivity
from slq.smooth.scales import collect_act_scales
from slq.smooth.smooth import SmoothReport, discover_sites, smooth_model

__all__ = ["SLQConfig", "SLQResult", "run_slq"]


@dataclass
class SLQConfig:
    """Configuration for a full SLQ run.

    Args:
        bitwidths: Candidate bitwidths ``B``. The paper uses ``{2,...,8}``.
        group_size: Weight quantization group size (128 in the paper).
        symmetric: Symmetric weight grids. The paper shows asymmetric is a
            prerequisite for distribution-losslessness (Section 3.1), so this
            defaults to ``False`` and exists mainly to reproduce that ablation.
        fmt: ``"int"`` or ``"fp"``; affects storage accounting (Table 7).
        quantizer: ``"gptq"`` (paper default) or ``"rtn"``.
        act_bits: Activation bitwidth for W+A runs; ``None`` for weight-only.
        act_granularity: Activation quantization granularity.
        act_symmetric: Symmetric activation grids. ``False`` applies the
            gamma-squared argument to the activation path.
        estimator: ``"shapley"`` (Algorithm 1) or ``"linear"`` (Appendix A.2).
        permutations: ``P`` for the Shapley estimator.
        target: ``"dl"`` for distribution-lossless, ``"tl"`` for task-lossless.
        target_ear: The DL constraint.
        target_recovery: The TL constraint.
        calibration_bits: TL anchor budget (Algorithm 2).
        smooth: Apply SmoothQuant difficulty migration.
        smooth_alpha: Migration strength, shared or per site.
        grouping: Layer partitioning policy.
        topk: Truncation ``K`` for EAR and KL.
        kl_mode: ``"truncated"`` (paper) or ``"renormalized"``.
        solver: Allocation solver.
        seed: Seed for permutation sampling.
    """

    bitwidths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8)
    group_size: int = 128
    symmetric: bool = False
    fmt: str = "int"
    quantizer: str = "gptq"

    act_bits: int | None = None
    act_granularity: str = "per_token"
    act_symmetric: bool = False

    estimator: str = "shapley"
    permutations: int = 8

    target: str = "dl"
    target_ear: float = 0.99
    target_recovery: float = 0.99
    calibration_bits: float = 4.0

    smooth: bool = False
    smooth_alpha: float | dict[str, float] = 0.5

    grouping: GroupingPolicy = field(default_factory=GroupingPolicy)
    topk: int = 10
    kl_mode: str = "truncated"
    solver: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.target not in ("dl", "tl"):
            raise ValueError(f"target must be 'dl' or 'tl', got {self.target!r}")
        if self.estimator not in ("shapley", "linear"):
            raise ValueError(f"estimator must be 'shapley' or 'linear', got {self.estimator!r}")

    def effective_bits_fn(self) -> Callable[[int], float]:
        """Nominal-to-realized bitwidth map for this configuration (Table 7)."""

        def f(b: int) -> float:
            return effective_bits(
                QuantConfig(
                    bits=b, group_size=self.group_size, symmetric=self.symmetric, fmt=self.fmt
                )
            )

        return f


@dataclass
class SLQResult:
    """Everything a run produced."""

    assignment: dict[str, int]
    average_bits: float
    search: SearchResult
    database: SensitivityDatabase
    measured: FidelityResult
    model: QuantizableModel
    smooth_report: SmoothReport | None = None

    def summary(self) -> dict:
        d = {
            "average_bits": self.average_bits,
            "target": self.search.target,
            "satisfied": self.search.satisfied,
            "measured_ear": self.measured.ear,
            "measured_kl": self.measured.kl,
            "flip_rate": self.measured.flip_rate,
            "topk_mass": self.measured.topk_mass,
            "groups": len(self.assignment),
        }
        if self.smooth_report is not None:
            d["smooth"] = self.smooth_report.summary()
        return d

    def bitwidth_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for b in self.assignment.values():
            hist[b] = hist.get(b, 0) + 1
        return dict(sorted(hist.items()))


def run_slq(
    model: nn.Module,
    calibration: Sequence[torch.Tensor],
    config: SLQConfig | None = None,
    forward: Callable[[torch.Tensor], torch.Tensor] | None = None,
    benchmark: BenchmarkFn | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> SLQResult:
    """Run the full SLQ pipeline and leave the model at the chosen precision.

    Args:
        model: The model to quantize, modified in place.
        calibration: Calibration batches.
        config: Run configuration; defaults are the paper's DL setup.
        forward: How to invoke the model; defaults to ``model(batch)``.
        benchmark: Required for ``target="tl"``. Called with an assignment, or
            ``None`` for the full-precision baseline, and returns a score.
        progress: Optional ``(stage, done, total)`` callback.

    Returns:
        An :class:`SLQResult`. The model is left carrying the winning
        assignment, and ``result.model`` holds the full-precision masters.

    Warning:
        ``model`` is modified in place: smoothing rewrites weights and the
        winning assignment is installed on exit. Do not pass the same model
        object to a second run -- the second wrapper would capture the
        *quantized* weights as its full-precision reference and silently
        measure against the wrong baseline. Call ``result.model.restore()``
        first, or load a fresh model per run.
    """
    cfg = config if config is not None else SLQConfig()
    fwd = forward if forward is not None else (lambda b: model(b))

    # -- 1. Difficulty migration ------------------------------------------- #
    report: SmoothReport | None = None
    if cfg.smooth:
        act_scales = collect_act_scales(model, calibration, forward=fwd)
        sites = discover_sites(model, calibration[0], forward=fwd)
        report = smooth_model(
            model, act_scales, sites, alpha=cfg.smooth_alpha, group_size=cfg.group_size
        )

    # -- 2-3. Calibrate and bank ------------------------------------------- #
    bank = WeightBank(
        bitwidths=cfg.bitwidths,
        group_size=cfg.group_size,
        symmetric=cfg.symmetric,
        fmt=cfg.fmt,
        method=cfg.quantizer,
    )
    act_cfg = ActQuantConfig(
        bits=cfg.act_bits, granularity=cfg.act_granularity, symmetric=cfg.act_symmetric
    )
    qmodel = QuantizableModel(
        model,
        calibration,
        policy=cfg.grouping,
        bank=bank,
        forward=fwd,
        topk=cfg.topk,
        act_quant=act_cfg,
    )
    qmodel.build_bank()

    # -- 4. Sensitivity ----------------------------------------------------- #
    if cfg.estimator == "shapley":
        db = shapley_sensitivity(
            qmodel,
            bitwidths=cfg.bitwidths,
            permutations=cfg.permutations,
            seed=cfg.seed,
            progress=progress,
        )
    else:
        db = linear_sensitivity(
            qmodel, bitwidths=cfg.bitwidths, progress=progress
        )

    # -- 5. Search ---------------------------------------------------------- #
    eff = cfg.effective_bits_fn()

    def measure(assignment: dict[str, int]) -> tuple[float, float]:
        r = qmodel.evaluate(assignment)
        return r.ear, r.kl

    if cfg.target == "dl":
        search = search_distribution_lossless(
            db,
            target_ear=cfg.target_ear,
            effective_bits_fn=eff,
            solver=cfg.solver,
            measure=measure,
        )
    else:
        if benchmark is None:
            raise ValueError("target='tl' requires a benchmark callable")

        def measure_kl(assignment: dict[str, int]) -> float:
            return qmodel.evaluate(assignment).kl

        search = search_task_lossless(
            db,
            measure_kl=measure_kl,
            benchmark=benchmark,
            target_recovery=cfg.target_recovery,
            calibration_bits=cfg.calibration_bits,
            effective_bits_fn=eff,
            solver=cfg.solver,
        )

    # -- 6. Apply and verify ------------------------------------------------ #
    qmodel.apply(search.assignment)
    measured = qmodel.evaluate()

    return SLQResult(
        assignment=search.assignment,
        average_bits=qmodel.average_bits(search.assignment),
        search=search,
        database=db,
        measured=measured,
        model=qmodel,
        smooth_report=report,
    )
