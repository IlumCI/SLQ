"""Linear sensitivity estimation (Section 3.3, Appendix A.2).

The cheap baseline, following Malinovskii et al. (2025). Metric degradation is
modelled as a weighted sum of per-group normalized reconstruction errors:

    e_m^(b)      = (1/|G_m|) sum_{l in G_m} ||W_l - W_hat_l^(b)||_F^2 / ||W_l||_F^2
    Delta_KL(b)  = sum_m e_m^(b_m) * alpha_m^KL                        (Eq. 5)
    Delta_EAR(b) = sum_m e_m^(b_m) * alpha_m^EAR                       (Eq. 6)

The coefficients ``alpha_m`` are fitted by noise injection: perturb one group at
a time and regress the observed metric change on the injected error. Cost is
``O(T * M)`` forward passes against Shapley's ``O(P * M * |B|)`` -- 10-100x
cheaper, independent of model size. Appendix E notes the trade-off is compute
against *bitwidth*, not against fidelity: both estimators reach the same
fidelity target, Shapley just gets there with a slightly smaller model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from slq.model.wrapper import QuantizableModel
from slq.sensitivity.database import SensitivityDatabase

__all__ = ["linear_sensitivity", "reconstruction_errors"]

ProgressFn = Callable[[str, int, int], None]


@torch.no_grad()
def reconstruction_errors(
    qmodel: QuantizableModel, bitwidths: Sequence[int]
) -> dict[str, dict[int, float]]:
    """Normalized reconstruction error ``e_m^(b)`` per group and bitwidth."""
    errors: dict[str, dict[int, float]] = {}
    for group in qmodel.groups:
        per_bits: dict[int, float] = {}
        for b in bitwidths:
            acc = 0.0
            for layer in group.layers:
                w = qmodel._layers[layer].fp_weight.float()
                w_hat = qmodel.bank.get(layer, b).materialize(w.device, torch.float32)
                denom = w.pow(2).sum().clamp_min(1e-12)
                acc += float((w - w_hat).pow(2).sum() / denom)
            per_bits[b] = acc / max(len(group.layers), 1)
        errors[group.name] = per_bits
    return errors


@torch.no_grad()
def linear_sensitivity(
    qmodel: QuantizableModel,
    bitwidths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    probe_bits: Sequence[int] | None = None,
    max_batches: int | None = None,
    progress: ProgressFn | None = None,
) -> SensitivityDatabase:
    """Fit per-group sensitivity coefficients by noise injection.

    Args:
        qmodel: A model whose weight bank already covers ``bitwidths``.
        bitwidths: Candidate bitwidths ``B``.
        probe_bits: Bitwidths used as noise-injection probes when fitting
            ``alpha_m``. Defaults to the two most aggressive bitwidths, which
            give the largest measurable signal.
        max_batches: Evaluate on a prefix of the calibration set.
        progress: Optional callback ``(stage, done, total)``.

    Returns:
        A :class:`SensitivityDatabase` with ``method="linear"``.
    """
    bits = sorted(bitwidths)
    b_max = bits[-1]
    groups = qmodel.group_names
    probes = list(probe_bits) if probe_bits else bits[: min(2, len(bits) - 1)]

    errors = reconstruction_errors(qmodel, bits)
    db = SensitivityDatabase.empty(groups, bits, qmodel.group_numel(), method="linear")

    # Baseline: everything at the reference bitwidth.
    base_assignment = dict.fromkeys(groups, b_max)
    qmodel.apply(base_assignment)
    base = qmodel.evaluate(max_batches=max_batches)
    db.baseline_ear = base.ear
    db.baseline_kl = base.kl

    total = len(groups) * len(probes)
    done = 0
    for g in groups:
        # Least-squares slope through the origin over the probe points:
        # alpha = sum(e * delta) / sum(e^2).
        num_kl = num_ear = denom = 0.0
        for pb in probes:
            assignment = dict(base_assignment)
            assignment[g] = pb
            qmodel.apply(assignment)
            r = qmodel.evaluate(max_batches=max_batches)

            e = errors[g][pb] - errors[g][b_max]
            num_kl += e * (r.kl - base.kl)
            num_ear += e * (base.ear - r.ear)
            denom += e * e

            done += 1
            if progress is not None:
                progress("linear", done, total)

        alpha_kl = num_kl / denom if denom > 0 else 0.0
        alpha_ear = num_ear / denom if denom > 0 else 0.0
        for b in bits:
            e_b = errors[g][b] - errors[g][b_max]
            db.kl[g][b] = alpha_kl * e_b
            db.ear_drop[g][b] = alpha_ear * e_b

    db.meta = {
        "probe_bits": probes,
        "b_max": b_max,
        "max_batches": max_batches,
        "recon_error": errors,
    }
    qmodel.restore()
    return db.enforce_monotonic()
