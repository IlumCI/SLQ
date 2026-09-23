"""SmoothQuant migration plus SLQ's asymmetric activation grids.

Shows the two ways the combination pays off:

1. Migration makes low-bit activations viable at all, which is what unlocks
   weight-and-activation configurations -- the limitation SLQ's Appendix E names.
2. SLQ's gamma-squared argument, applied to the activation path, says a
   zero-anchored activation grid costs gamma^2 in noise. Post-SiLU activations
   are bounded below and long-tailed above, so gamma is large and the asymmetric
   grid wins by a wide margin.

    python examples/03_smoothquant_activations.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch

from slq.data import build_calibration
from slq.metrics.fidelity import fidelity
from slq.model.bank import WeightBank
from slq.model.wrapper import QuantizableModel
from slq.quant.act import ActQuantConfig
from slq.smooth import collect_act_scales, discover_sites, smooth_model


def measure(model, calib, ref, act_cfg, weight_bits):
    qm = QuantizableModel(
        model, calib, forward=lambda b: model(b).logits,
        bank=WeightBank(bitwidths=(weight_bits,), method="rtn"),
        act_quant=act_cfg,
    )
    qm.build_bank()
    qm.apply(qm.uniform(weight_bits))
    with torch.no_grad():
        cand = qm._logits(calib)
    qm.restore()
    flat = lambda xs: torch.cat([x.reshape(-1, x.shape[-1]) for x in xs])  # noqa: E731
    return fidelity(flat(ref), flat(cand))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--weight-bits", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    calib = build_calibration(tok, n_samples=args.samples, seq_len=args.seq_len)
    fwd = lambda b: model(b).logits  # noqa: E731

    with torch.no_grad():
        ref = [fwd(b).float() for b in calib]

    print(f"W{args.weight_bits}A{args.act_bits}, alpha={args.alpha}\n")
    print(f"{'configuration':<34} {'EAR':>9} {'KL':>10} {'flips':>8}")

    for label, cfg in [
        ("no smoothing, symmetric acts", ActQuantConfig(args.act_bits, "per_token", True)),
        ("no smoothing, asymmetric acts", ActQuantConfig(args.act_bits, "per_token", False)),
    ]:
        r = measure(model, calib, ref, cfg, args.weight_bits)
        print(f"{label:<34} {r.ear:>9.5f} {r.kl:>10.5f} {r.flip_rate:>8.4f}")

    scales = collect_act_scales(model, calib, forward=fwd)
    sites = discover_sites(model, calib[0], forward=fwd)
    report = smooth_model(model, scales, sites, alpha=args.alpha)
    print(f"\nsmoothed {len(sites)} sites; mean gamma "
          f"{report.summary()['mean_gamma_before']:.4f} -> "
          f"{report.summary()['mean_gamma_after']:.4f}\n")

    for label, cfg in [
        ("smoothed, symmetric acts", ActQuantConfig(args.act_bits, "per_token", True)),
        ("smoothed, asymmetric acts", ActQuantConfig(args.act_bits, "per_token", False)),
    ]:
        r = measure(model, calib, ref, cfg, args.weight_bits)
        print(f"{label:<34} {r.ear:>9.5f} {r.kl:>10.5f} {r.flip_rate:>8.4f}")


if __name__ == "__main__":
    main()
