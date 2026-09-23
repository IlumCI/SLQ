"""Run the whole SLQ pipeline on the built-in reference model, on CPU.

No downloads, no GPU: this exercises calibration, GPTQ, the Shapley sweep, the
ILP allocation and the DL search in about a minute, which makes it the quickest
way to check an installation or to step through the method in a debugger.

    python examples/02_end_to_end_reference.py
"""

from __future__ import annotations

import time

import torch

from slq.model.grouping import GroupingPolicy
from slq.model.reference import make_peaked_reference
from slq.pipeline import SLQConfig, run_slq


def main() -> None:
    torch.manual_seed(0)

    t = time.time()
    model, sampler = make_peaked_reference()
    print(f"reference model trained in {time.time() - t:.0f}s")

    calibration = [sampler(2, 48) for _ in range(6)]

    config = SLQConfig(
        bitwidths=(2, 3, 4, 6, 8),
        permutations=3,
        target="dl",
        target_ear=0.99,
        grouping=GroupingPolicy(granularity="layer"),
    )

    t = time.time()
    result = run_slq(model, calibration, config)
    print(f"SLQ search completed in {time.time() - t:.0f}s\n")

    for k, v in result.summary().items():
        print(f"  {k:<16} {v:.5f}" if isinstance(v, float) else f"  {k:<16} {v}")
    print(f"\n  bitwidth histogram: {result.bitwidth_histogram()}")

    print("\n  per-group assignment (first 10):")
    for name, bits in list(result.assignment.items())[:10]:
        print(f"    {name:<28} {bits} bits")


if __name__ == "__main__":
    main()
