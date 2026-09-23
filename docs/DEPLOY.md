# Deploying an SLQ-allocated model with llama.cpp

SLQ decides which tensor gets which bitwidth. It does not need to do the
quantization itself: `llama-quantize` accepts per-tensor type overrides, so the
allocation can be handed to it as a small text file. That matters at scale --
the allocation is kilobytes of decisions, while holding a 27B model in memory
to quantize it is not possible on a 16 GB machine. llama.cpp streams the
weights; SLQ supplies the policy.

## The constraint, stated honestly

Weights alone, for a dense 27B model:

| quant | bits/param | size |
|---|---|---|
| Q6_K | 6.56 | 21.98 GB |
| Q5_K_M | 5.67 | 19.77 GB |
| Q4_K_M | 4.83 | 16.46 GB |
| IQ4_XS | 4.25 | 14.25 GB |
| Q3_K_XL | 3.90 | 13.15 GB |

On 16 GB RAM + 4 GB VRAM the usable budget is roughly **16.5 GB** once the OS
and a KV cache are accounted for. Q6_K needs 22 GB. No allocation strategy
closes a 5.5 GB gap; SLQ's measured advantage is 0.13-0.42 bits per parameter,
which on 27B is 0.5-1.4 GB. It buys roughly half a quant level, not two.

**A dense 27B at Q6 does not fit on that hardware.** What SLQ can do is make
the ~14 GB configuration that *does* fit as good as it can be.

### Consider a mixture-of-experts model instead

CPU inference is bandwidth-bound: every token reads the weights it activates.
A dense model reads all of them.

| model | size | read/token | rough speed |
|---|---|---|---|
| dense 27B @ IQ4_XS | 14.25 GB | ~14 GB | 3-5 tok/s |
| 30B-A3B (3B active) @ Q3_K_M | 14.71 GB | ~2-3 GB | 12-20 tok/s |

Same footprint, several times the throughput, because only the routed experts
are touched. The paper's Appendix F confirms the SLQ pipeline transfers to MoE
models unchanged.

## Procedure

### 1. Convert the model to an unquantized GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py --outtype f16 \
    --outfile model-f16.gguf /path/to/hf/model
```

This needs disk for the F16 copy (~54 GB for 27B) but not RAM.

### 2. Build a sensitivity database

On a model small enough to hold, run the real estimator:

```python
from slq.model import QuantizableModel, WeightBank, GroupingPolicy
from slq.sensitivity import shapley_sensitivity
from slq.data import build_calibration

calib = build_calibration(tokenizer, n_samples=512, seq_len=512)
qm = QuantizableModel(model, calib, forward=lambda b: model(b).logits,
                      bank=WeightBank(bitwidths=(4, 5, 6, 8)),
                      policy=GroupingPolicy(granularity="layer"))
qm.build_bank()
db = shapley_sensitivity(qm, bitwidths=(4, 5, 6, 8), permutations=4)
db.save("sensitivity.json")
```

Use **per-layer** granularity. Block-level grouping forces every projection in
a block to share a bitwidth, which removes the within-block variation the
method depends on and loses to uniform quantization outright (see
`RESULTS.md`).

Keep the candidate bitwidths at 4 and above. The additive prediction of
Equations 7-8 is accurate to ~0.02 EAR down to about 5 bpp and then degrades
sharply; at 3.2 bpp it overestimates EAR by 0.52 and will confidently choose a
configuration that destroys the model.

For a model too large to hold, the transferable part is the *shape* of the
sensitivity -- which projection types matter -- measured on a smaller sibling
of the same family. That is an approximation, and it should be stated as one.

### 3. Allocate under a memory budget

```python
from slq.alloc import search_memory_budget
from slq.quant.grid import QuantConfig, effective_bits

eff = lambda b: effective_bits(QuantConfig(b, 128))
result = search_memory_budget(
    db, budget_bytes=13.5e9, effective_bits_fn=eff, overhead_fraction=0.12,
)
print(result.average_bits, result.notes["realized_gigabytes"])
```

`overhead_fraction` reserves room for the KV cache and activations. At 8k
context on a 27B model the KV cache alone is on the order of a gigabyte, so
0.10-0.15 is a reasonable starting point.

### 4. Export and quantize

```python
from slq.export import write_tensor_type_file, quantize_command
write_tensor_type_file("slq_types.txt", result.assignment)
print(" ".join(quantize_command("model-f16.gguf", "model-slq.gguf",
                                "slq_types.txt", base_type="q4_k_m")))
```

```bash
llama-quantize --output-tensor-type q6_K --token-embedding-type q6_K \
    --tensor-type-file slq_types.txt \
    model-f16.gguf model-slq.gguf q4_k_m 8
```

The tensor-type file must contain nothing but `pattern=type` lines. llama.cpp
tokenizes it on whitespace and has no comment syntax, so a `#` header fails
with `malformed tensor type '#'` and aborts the run.

Add `--imatrix` if you have an importance matrix; it materially helps the
sub-4-bit types and costs nothing at inference.

### 5. Run with GPU offload

```bash
llama-cli -m model-slq.gguf -ngl 12 -c 4096 -t 8
```

`-ngl` is the number of layers placed on the GPU. On 4 GB of VRAM, start low
and raise it until allocation fails. Because SLQ has already assigned the
sensitive K/V projections a high bitwidth, the layers you offload carry their
precision with them.

## What to expect

Measured on Qwen3-0.6B with this exact path, against a plain `q4_k_m` build of
the same model: the allocation loads, runs, and generates at 24 tok/s on two
CPU threads. Size-matched perplexity figures are in `RESULTS.md`.

The honest summary for a 27B target on 16 GB: expect a ~13-14 GB model of
roughly Q4 class, allocated better than a uniform build of the same size.
Expect neither Q6 quality nor anything close to it.
