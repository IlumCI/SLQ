# SLQ allocations for Qwen/Qwen3.8-27B

Per-tensor quantization assignments for `llama-quantize --tensor-type-file`,
targeting **16 GB RAM + 4 GB VRAM**.

| file | total | avg bits | character |
|---|---|---|---|
| `slq-14.5gb-safe.txt` | ~14.5 GB | 4.18 bpp | nothing below 4 bits |
| `slq-14.5gb-aggressive.txt` | ~14.5 GB | 4.18 bpp | MLP down to 3 bits, more bits to attention |

Both cover all 496 quantizable tensors; each pattern matches exactly one tensor.

## What this model actually is

Qwen3.8-27B is **not a plain transformer**, which matters because a naive
name-based allocation silently misses most of it:

- 64 blocks, of which **48 are linear-attention/SSM** (`attn_qkv`, `attn_gate`,
  `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_conv1d`) and **16 are standard
  attention** (blocks 3, 7, 11, ... `attn_q/k/v/output`).
- All 64 have an MLP, which is **70% of all quantizable parameters**.
- A vision tower ships separately as `mmproj-*.gguf`, so it is not in the main
  model file.
- 27.78B total, of which 24.35B are SLQ-quantizable 2-D language weights.

Tensor names come from llama.cpp's own `gguf-py` mapper (`MODEL_ARCH.QWEN35`),
not a hand-written table, because the SSM name mapping
(`in_proj_a -> ssm_alpha`, `in_proj_z -> attn_gate`) is not guessable.

## Why the allocation looks the way it does

`attn_k` and `attn_v` are 0.3% of parameters each, and `ssm_alpha`/`ssm_beta`
are rounding error. Protecting the most sensitive tensors at 8 bits therefore
costs almost nothing, and the budget is decided almost entirely by the MLP.
That asymmetry is what makes non-uniform allocation worth doing here.

## Usage

```bash
# 1. Convert (needs ~56 GB of disk for the F16 copy, but little RAM)
python llama.cpp/convert_hf_to_gguf.py --outtype f16 \
    --outfile qwen38-f16.gguf /path/to/Qwen3.8-27B

# 2. Quantize with the SLQ allocation
llama-quantize \
    --token-embedding-type q4_K \
    --output-tensor-type q5_K \
    --tensor-type-file slq-14.5gb-safe.txt \
    qwen38-f16.gguf qwen38-slq.gguf q4_k_m 8

# 3. Run, offloading what fits on the 3050
llama-cli -m qwen38-slq.gguf -ngl 10 -c 4096 -t 8
```

Add `--imatrix` if you have an importance matrix; it helps the 3-bit tensors in
the aggressive file appreciably and costs nothing at inference. Raise `-ngl`
until VRAM allocation fails, then back off one.

## Honest status of these numbers

**Measured in this repository**, on Qwen3-0.6B: the sensitivity ordering of
standard-attention and MLP projections (v > k > up > down > q = gate > o), and
that non-uniform allocation beats uniform by 0.005-0.016 EAR at equal size.

**Assumed, not measured**: the sensitivity of the linear-attention and SSM
tensors (`attn_qkv`, `attn_gate`, `ssm_out`, `ssm_alpha`, `ssm_beta`). Qwen3-0.6B
has no counterpart to them, so their coefficients are priors, and they are
marked with `*` wherever this repository prints them. They drive the allocation
of ~23% of parameters.

**Not verified at all**: that this allocation achieves any particular quality on
the 27B model. Confirming that needs a calibration run against the real
checkpoint, which needs more disk and RAM than the machine this was built on.

### On the quality target

At 4.18 bpp this sits inside the paper's **task-lossless** band (3.3-4.7 bpp,
defined as >=99% of BF16 benchmark accuracy). It does **not** reach the paper's
**distribution-lossless** band (5.0-6.6 bpp, EAR >= 0.99), which for this model
would be 17.4-22.9 GB and does not fit in 16 GB + 4 GB.

So: task-lossless is plausibly in reach at this size; output-indistinguishable
is not, and neither is Q6_K, which needs 21.98 GB.
