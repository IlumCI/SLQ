"""Validate the gamma-squared variance law (Lemma 3.2) on real weights.

The paper's central theoretical claim is that a zero-anchored (symmetric) grid
inflates quantization noise variance by ``gamma^2 = (2M/R)^2`` relative to an
asymmetric one. This script measures that ratio directly.

    python examples/01_gamma_squared_law.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse

import torch

from slq.quant.grid import QuantConfig, centering_inefficiency, fake_quantize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--bits", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)

    layers = [
        (n, m.weight.data.float())
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and "layers." in n
    ]
    print(f"{args.model}: {len(layers)} transformer linear layers\n")

    gammas = torch.tensor([float(centering_inefficiency(w, args.group_size).mean())
                           for _, w in layers])
    print(f"centering inefficiency gamma = 2M/R (group size {args.group_size})")
    print(f"  mean {gammas.mean():.4f}   min {gammas.min():.4f}   max {gammas.max():.4f}")
    print(f"  => predicted noise inflation gamma^2 = {gammas.mean() ** 2:.4f}\n")

    print(f"{'bits':>5} {'predicted gamma^2':>18} {'measured MSE ratio':>19} {'error':>8}")
    for bits in args.bits:
        ratios, g2 = [], []
        for _, w in layers:
            cfg_a = QuantConfig(bits, args.group_size, symmetric=False)
            cfg_s = QuantConfig(bits, args.group_size, symmetric=True)
            mse_a = float((w - fake_quantize(w, cfg_a)).pow(2).mean())
            mse_s = float((w - fake_quantize(w, cfg_s)).pow(2).mean())
            ratios.append(mse_s / mse_a)
            g2.append(float(centering_inefficiency(w, args.group_size).mean()) ** 2)
        pred = sum(g2) / len(g2)
        meas = sum(ratios) / len(ratios)
        print(f"{bits:>5} {pred:>18.4f} {meas:>19.4f} {abs(meas - pred) / pred:>7.2%}")

    print(
        "\nThe law is a high-rate (Bennett) result, so agreement tightens as the\n"
        "bitwidth grows and loosens at 2-3 bits where the bins stop being narrow\n"
        "relative to the density."
    )


if __name__ == "__main__":
    main()
