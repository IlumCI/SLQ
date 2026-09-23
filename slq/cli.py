"""Command-line interface for SLQ.

    slq sweep    --model Qwen/Qwen3-0.6B            # uniform bitwidth sweep
    slq gamma    --model Qwen/Qwen3-0.6B            # gamma-squared law check
    slq run      --model Qwen/Qwen3-0.6B --target dl --ear 0.99
    slq smooth   --model Qwen/Qwen3-0.6B --alpha-search
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence

import torch

from slq.data import build_calibration
from slq.metrics.fidelity import fidelity
from slq.model.grouping import GroupingPolicy
from slq.model.wrapper import QuantizableModel
from slq.model.bank import WeightBank
from slq.pipeline import SLQConfig, run_slq
from slq.quant.grid import QuantConfig, centering_inefficiency, fake_quantize

__all__ = ["main"]

DEFAULT_BITS = (2, 3, 4, 5, 6, 7, 8)


def _load(model_id: str, dtype: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("this command needs 'transformers'; install slq[hf]") from e
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.eval()
    return model, tok


def _calibration(tok, args) -> list[torch.Tensor]:
    return build_calibration(
        tok, n_samples=args.samples, seq_len=args.seq_len, batch_size=args.batch_size,
        seed=args.seed,
    )


def _wrap(model, tok, args, bitwidths: Sequence[int]) -> QuantizableModel:
    calib = _calibration(tok, args)
    bank = WeightBank(
        bitwidths=bitwidths,
        group_size=args.group_size,
        symmetric=args.symmetric,
        method=args.quantizer,
    )
    return QuantizableModel(
        model,
        calib,
        policy=GroupingPolicy(granularity=args.granularity),
        bank=bank,
        forward=lambda b: model(b).logits,
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_gamma(args: argparse.Namespace) -> int:
    """Check the gamma-squared variance law (Lemma 3.2) on real weights."""
    model, _ = _load(args.model, args.dtype)
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or m.weight.ndim != 2:
            continue
        if any(s in name for s in ("lm_head", "embed")):
            continue
        w = m.weight.data.float()
        gamma = float(centering_inefficiency(w, args.group_size).mean())
        cfg_a = QuantConfig(args.bits, args.group_size, symmetric=False)
        cfg_s = QuantConfig(args.bits, args.group_size, symmetric=True)
        mse_a = float((w - fake_quantize(w, cfg_a)).pow(2).mean())
        mse_s = float((w - fake_quantize(w, cfg_s)).pow(2).mean())
        rows.append(
            {"layer": name, "gamma": gamma, "gamma_sq": gamma**2,
             "mse_ratio": mse_s / mse_a if mse_a > 0 else float("nan")}
        )
        if args.limit and len(rows) >= args.limit:
            break

    mean_g2 = sum(r["gamma_sq"] for r in rows) / len(rows)
    mean_ratio = sum(r["mse_ratio"] for r in rows) / len(rows)
    print(f"layers={len(rows)}  bits={args.bits}  group_size={args.group_size}")
    print(f"mean gamma^2        = {mean_g2:.4f}   (predicted MSE_sym/MSE_asym)")
    print(f"mean measured ratio = {mean_ratio:.4f}")
    print(f"relative error      = {abs(mean_ratio - mean_g2) / mean_g2:.2%}")
    if args.json:
        _dump(args.json, {"mean_gamma_sq": mean_g2, "mean_ratio": mean_ratio, "layers": rows})
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Measure EAR / KL at each uniform bitwidth."""
    model, tok = _load(args.model, args.dtype)
    bits = tuple(args.bits)
    qm = _wrap(model, tok, args, bits)
    t = time.time()
    qm.build_bank()
    print(f"bank: {len(qm.bank)} entries, {qm.bank.nbytes / 1e9:.2f} GB, {time.time() - t:.0f}s")
    print(f"{'bits':>5} {'bpp':>7} {'EAR':>8} {'KL':>9} {'flips':>7} {'top-K':>7}")
    rows = []
    for b in sorted(bits, reverse=True):
        a = qm.uniform(b)
        r = qm.evaluate(a)
        bpp = qm.average_bits(a)
        print(f"{b:>5} {bpp:>7.3f} {r.ear:>8.5f} {r.kl:>9.5f} {r.flip_rate:>7.4f} "
              f"{r.topk_mass:>7.4f}")
        rows.append({"bits": b, "bpp": bpp, **r.as_dict()})
    if args.json:
        _dump(args.json, {"model": args.model, "rows": rows})
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full SLQ search."""
    model, tok = _load(args.model, args.dtype)
    calib = _calibration(tok, args)
    cfg = SLQConfig(
        bitwidths=tuple(args.bits),
        group_size=args.group_size,
        symmetric=args.symmetric,
        quantizer=args.quantizer,
        estimator=args.estimator,
        permutations=args.permutations,
        target=args.target,
        target_ear=args.ear,
        target_recovery=args.recovery,
        act_bits=args.act_bits,
        act_symmetric=args.act_symmetric,
        smooth=args.smooth,
        smooth_alpha=args.alpha,
        grouping=GroupingPolicy(granularity=args.granularity),
        seed=args.seed,
    )

    def progress(stage: str, done: int, total: int) -> None:
        if done % max(1, total // 20) == 0 or done == total:
            print(f"\r  {stage}: {done}/{total}", end="", file=sys.stderr, flush=True)

    benchmark = None
    if args.target == "tl":
        raise SystemExit(
            "target='tl' needs a benchmark callable; use the Python API "
            "(slq.pipeline.run_slq) and pass one, since benchmark choice is task-specific"
        )

    t = time.time()
    res = run_slq(model, calib, cfg, forward=lambda b: model(b).logits,
                  benchmark=benchmark, progress=progress)
    print(file=sys.stderr)
    print(f"completed in {time.time() - t:.0f}s")
    print(json.dumps(res.summary(), indent=2))
    print("bitwidth histogram:", res.bitwidth_histogram())
    if args.json:
        _dump(args.json, {
            "summary": res.summary(),
            "assignment": res.assignment,
            "histogram": res.bitwidth_histogram(),
            "search": res.search.as_dict(),
        })
    if args.save_db:
        res.database.save(args.save_db)
        print(f"sensitivity database -> {args.save_db}")
    return 0


def cmd_smooth(args: argparse.Namespace) -> int:
    """Apply SmoothQuant migration and report its effect on gamma and EAR."""
    from slq.smooth import collect_act_scales, discover_sites, smooth_model

    model, tok = _load(args.model, args.dtype)
    calib = _calibration(tok, args)
    fwd = lambda b: model(b).logits  # noqa: E731

    with torch.no_grad():
        ref = [fwd(b).float() for b in calib]

    scales = collect_act_scales(model, calib, forward=fwd)
    sites = discover_sites(model, calib[0], forward=fwd)
    print(f"discovered {len(sites)} smoothing sites")

    report = smooth_model(model, scales, sites, alpha=args.alpha, group_size=args.group_size)
    print(json.dumps({k: round(v, 5) for k, v in report.summary().items()}, indent=2))

    with torch.no_grad():
        after = [fwd(b).float() for b in calib]
    drift = max(float((a - b).abs().max()) for a, b in zip(ref, after, strict=True))
    print(f"max |logit drift| from smoothing: {drift:.3e} (should be ~0)")

    if args.act_bits:
        for sym in (True, False):
            qm = QuantizableModel(
                model, calib, forward=fwd,
                bank=WeightBank(bitwidths=(8,), group_size=args.group_size),
                act_quant=__import__("slq.quant.act", fromlist=["ActQuantConfig"]).ActQuantConfig(
                    bits=args.act_bits, symmetric=sym
                ),
            )
            qm._reference_logits = ref
            r = fidelity(torch.cat([x.reshape(-1, x.shape[-1]) for x in ref]),
                         torch.cat([x.reshape(-1, x.shape[-1]) for x in qm._logits(calib)]))
            print(f"  A{args.act_bits} {'sym ' if sym else 'asym'}: EAR={r.ear:.5f} KL={r.kl:.5f}")
    return 0


def _dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="HuggingFace model id or local path")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--samples", type=int, default=32, help="calibration windows")
    p.add_argument("--seq-len", type=int, default=512, help="tokens per window")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--symmetric", action="store_true",
                   help="symmetric weight grids (the paper's ablation, not its method)")
    p.add_argument("--quantizer", default="gptq", choices=["gptq", "rtn"])
    p.add_argument("--granularity", default="layer",
                   choices=["layer", "block", "module_type"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="write results to this JSON path")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="slq", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gamma", help="validate the gamma-squared variance law")
    _common(g)
    g.add_argument("--bits", type=int, default=4)
    g.add_argument("--limit", type=int, default=0, help="stop after N layers (0 = all)")
    g.set_defaults(func=cmd_gamma)

    s = sub.add_parser("sweep", help="EAR/KL at each uniform bitwidth")
    _common(s)
    s.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    s.set_defaults(func=cmd_sweep)

    r = sub.add_parser("run", help="run the full SLQ search")
    _common(r)
    r.add_argument("--bits", type=int, nargs="+", default=list(DEFAULT_BITS))
    r.add_argument("--target", default="dl", choices=["dl", "tl"])
    r.add_argument("--ear", type=float, default=0.99, help="DL target")
    r.add_argument("--recovery", type=float, default=0.99, help="TL target")
    r.add_argument("--estimator", default="shapley", choices=["shapley", "linear"])
    r.add_argument("--permutations", type=int, default=8)
    r.add_argument("--act-bits", type=int, default=None, help="enable W+A at this bitwidth")
    r.add_argument("--act-symmetric", action="store_true")
    r.add_argument("--smooth", action="store_true", help="SmoothQuant migration first")
    r.add_argument("--alpha", type=float, default=0.5)
    r.add_argument("--save-db", help="write the sensitivity database here")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("smooth", help="apply SmoothQuant migration and report")
    _common(m)
    m.add_argument("--alpha", type=float, default=0.5)
    m.add_argument("--act-bits", type=int, default=None)
    m.add_argument("--alpha-search", action="store_true",
                   help="reserved: use the Python API for per-site search")
    m.set_defaults(func=cmd_smooth)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
