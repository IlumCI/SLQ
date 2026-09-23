# SLQ

Implementation of **"Statistically-Lossless Quantization of Large Language Models"**
([arXiv:2605.02404](https://arxiv.org/abs/2605.02404), Helcig, Kurtić & Alistarh),
integrated with **SmoothQuant**
([Xiao et al., 2023](https://github.com/mit-han-lab/smoothquant)) for
weight-and-activation quantization.

SLQ inverts the usual framing of post-training quantization. Rather than fixing
a bitwidth and reporting the damage, it fixes a *fidelity target* and searches
for the minimum average bitwidth that meets it.

---

## The method in one page

**Two notions of losslessness.**

- *Task-lossless* (TL): zero-shot benchmark accuracy preserved within the
  run-to-run variance a model already shows under stochastic sampling.
- *Distribution-lossless* (DL): the next-token distribution is practically
  indistinguishable from the original.

**Expected Acceptance Rate (EAR)** is the DL metric. Borrowed from speculative
decoding, it is the probability mass overlap between the original and quantized
next-token distributions:

```
EAR = (1/N) Σ_i Σ_{k ∈ T_i} min(p_i(k), q_i(k))        # Eq. 3
```

EAR equals `1 − d_TV(p, q)`: under optimal coupling, the maximum probability the
two models emit the same token. `EAR = 0.99` means 99% agreement. `T_i` is the
top-`K = 10` support of the reference distribution.

**The γ² variance law.** For a weight group spanning `[L, U]`, write
`R = U − L`, `M = max(|L|, |U|)`, and define the *centering inefficiency*
`γ = 2M/R`. A symmetric grid is anchored at zero and so spans `[−M, M]`, wasting
capacity whenever the weights are offset. Its step size is `γ` times larger, and
under Bennett's high-rate approximation (`σ² = Δ²/12`) the noise variance is:

```
σ²_sym = γ² · σ²_asym                                   # Lemma 3.2
```

Asymmetric quantization is therefore a *prerequisite* for DL — but not for TL,
where stochastic sampling absorbs the extra token flips.

**The pipeline.** Layers are partitioned into groups (respecting fused kernels
such as vLLM's QKV). Multi-bitwidth Shapley estimation runs a separate binary
game per target bitwidth, producing a sensitivity database `φ_m^(b)` over the
full range `B = {2,…,8}`. Allocation is then a multiple-choice knapsack solved
by ILP, wrapped in a binary search over the bitwidth budget.

---

## Install

```bash
pip install -e .            # core: torch, numpy, scipy
pip install -e ".[hf]"      # + transformers/datasets for real models
pip install -e ".[dev]"     # + pytest, ruff
```

## Quick start

No download, no GPU — the full pipeline on a built-in reference model:

```bash
python examples/02_end_to_end_reference.py
```

On a real model:

```bash
slq gamma --model Qwen/Qwen3-0.6B                  # check the γ² law
slq sweep --model Qwen/Qwen3-0.6B --bits 2 3 4 6 8 # EAR/KL per uniform bitwidth
slq run   --model Qwen/Qwen3-0.6B --target dl --ear 0.99
```

Python API:

```python
from slq.pipeline import SLQConfig, run_slq
from slq.data import build_calibration

calibration = build_calibration(tokenizer, n_samples=512, seq_len=512)
result = run_slq(
    model, calibration,
    SLQConfig(target="dl", target_ear=0.99, estimator="shapley", permutations=8),
    forward=lambda b: model(b).logits,
)
print(result.summary(), result.bitwidth_histogram())
```

Task-lossless search needs a benchmark callable, since the choice of benchmark
is task-specific:

```python
SLQConfig(target="tl", target_recovery=0.99, calibration_bits=4.0)
# run_slq(..., benchmark=lambda assignment: my_eval(model))   # None = BF16 baseline
```

---

## What the combination with SmoothQuant adds

SLQ's Appendix E names weight-and-activation quantization as its principal open
limitation: the method is agnostic to it, but the paper reports W+A only at
small scale, needing 6.50–6.97 bpp via the expensive evolutionary search.
SmoothQuant is the missing piece, and the two compose more tightly than they
first appear.

**1. Migration changes γ, so the two methods interact through SLQ's own theory.**
SmoothQuant rescales weight columns by `s_j = max|X_j|^α / max|W_j|^(1−α)`,
which changes each quantization group's `[L, U]` and therefore its centering
inefficiency. `smooth_model` measures and reports `γ` before and after rather
than leaving the interaction implicit.

**2. The γ² law transfers to the activation path.** SmoothQuant quantizes
activations with a symmetric absmax grid. Post-SiLU activations are bounded
below and long-tailed above, so `γ` is large and Lemma 3.2 predicts a
substantial penalty. Measured here on such activations: `γ = 1.941`, predicted
`γ² = 3.77`, observed MSE ratios **3.69** (8-bit) and **3.73** (6-bit), drifting
to 2.68 at 4-bit as the high-rate assumption weakens. Asymmetric activation
grids are consequently the default in `slq.quant.act`.

**3. α becomes searchable under a cheap, principled metric.** SmoothQuant picks
one global α and validates it by downstream accuracy. EAR is bounded,
interpretable and measured on the calibration set, so `slq.smooth.alpha` selects
α *per site* by coordinate descent — attention and MLP sites do not want the
same migration strength.

**4. Site discovery is traced, not hardcoded.** Upstream SmoothQuant enumerates
`OPTDecoderLayer`, `LlamaDecoderLayer`, `BloomBlock`, `FalconDecoderLayer` and
so on by type, and a new architecture needs new code. `discover_sites` runs one
forward pass and records which linear layers consume each normalization output,
recovering exactly the same QKV and gate/up fusion groups on Llama-style models
while working unmodified on architectures nobody has written a branch for.

---

## Deviations from the paper, and why

Running the method surfaced three points where a literal reading does not work.
Each is documented at the point of implementation, and each reduces to the
paper's formula in the regime the paper operates in.

**1. EAR and KL are conditioned on the top-`K` support, not merely truncated.**
The paper writes the metrics as sums over `T_i` without renormalizing. Taken
literally, `EAR ≤ Σ_{k∈T_i} p_i(k)`, so EAR is capped by the top-`K` mass —
measured at **0.751** on Qwen3-0.6B over WikiText-2. The paper's `EAR ≥ 0.99`
target would then be unreachable by *any* quantizer, including a perfect one,
and `EAR(p, p) ≠ 1`. The paper also states EAR equals `1 − d_TV` and describes
it as a coupling probability; both identities require normalized distributions.
So "restricted to the top-`K` tokens" is implemented as conditioning. Pass
`normalize=False` for the literal form; `topk_mass` is always reported.

**2. Metric prediction is anchored on the measured baseline.** Equations 7–8
read `D̂_KL = Σ_m φ_m` and `EÂR = 1 − Σ_m φ_m`, which take the all-`b_max`
configuration to have EAR exactly 1. That is fine for a sharply peaked LLM, but
otherwise inflates every prediction by `1 − EAR(b_max)` — enough to make the DL
binary search undershoot its bitwidth. Predictions are anchored on the measured
reference instead, which is identical when the baseline is 1.

**3. Sensitivity costs are made monotone in bitwidth.** Shapley sampling noise
can leave a group marginally cheaper at `b` than at `b+1`, which lets the ILP
"buy" fidelity by *removing* bits. A running minimum from `b_max` downward
removes the artifact.

---

## Layout

```
slq/
  quant/grid.py       affine grids, γ, step sizes, Table 7 bit accounting
  quant/gptq.py       GPTQ with Cholesky inverse Hessian and lazy block updates
  quant/act.py        activation quantizers (per-token/tensor/channel, sym/asym)
  metrics/fidelity.py EAR, top-K KL, decision flips by entropy, margin
  smooth/scales.py    per-input-channel activation absmax collection
  smooth/smooth.py    traced site discovery, difficulty migration, γ reporting
  smooth/alpha.py     EAR-driven α search, global or per site
  model/grouping.py   layer partitioning with fused-kernel constraints
  model/bank.py       quantized-weight cache over every (layer, bitwidth)
  model/wrapper.py    apply assignments, measure fidelity
  model/reference.py  self-contained transformer for CPU tests
  sensitivity/        multi-bitwidth Shapley (Alg. 1), linear baseline (A.2)
  alloc/ilp.py        multiple-choice knapsack (HiGHS + Lagrangian fallback)
  alloc/search.py     DL binary search; TL single-point calibration (Alg. 2)
  alloc/evolution.py  constraint-based evolutionary search (Alg. 3)
  pipeline.py         end-to-end run
  data.py             calibration set construction
  cli.py              slq gamma / sweep / run / smooth
```

### Design notes

- **The weight bank is what makes the search affordable.** Algorithm 1 evaluates
  `O(P·M·|B|)` configurations; each must cost a forward pass, not a
  re-quantization. Every layer is quantized once per candidate bitwidth up
  front, after which applying a configuration is a tensor copy. Codes are stored
  as `uint8`, so a real run should bit-pack them if memory is tight.
- **GPTQ shares one Cholesky across a layer's bitwidths.** The inverse-Hessian
  factor depends only on calibration statistics.
- **GPTQ's column loop is pinned to one thread.** The error compensation is
  sequential over rank-1 updates too small to amortize thread dispatch; on a
  few-core machine the contention cost 100× (20.7s → 0.21s for one 256×512 layer
  at five bitwidths, with identical numerics).
- **The ILP budget is parameter-weighted**, matching `b̄ = Σ b_ℓ|W_ℓ| / Σ |W_ℓ|`.
  An unweighted budget lets the solver spend bits freely on the largest tensors.

## Validated results

Measured with this implementation on `Qwen/Qwen3-0.6B` (CPU, WikiText-2
calibration). Full tables in [`docs/RESULTS.md`](docs/RESULTS.md).

- **The γ² law holds on real weights.** Mean `γ ≈ 1.10` across projections,
  predicting a 1.19–1.24× noise penalty for symmetric grids; measured
  symmetric/asymmetric MSE ratios are 1.22–1.28.
- **It transfers to activations.** On post-SiLU activations (`γ = 1.941`,
  `γ² = 3.77`) the measured ratios are 3.69 at 8-bit and 3.73 at 6-bit.
- **The uniform sweep is monotone over four orders of magnitude of KL**, from
  EAR 0.992 / KL 0.0011 at 8-bit to complete collapse at 2-bit.

| bits | bpp | EAR | KL | flips | top-10 mass |
|---|---|---|---|---|---|
| 8 | 8.156 | 0.99234 | 0.00106 | 0.011 | 0.7874 |
| 6 | 6.156 | 0.97107 | 0.00908 | 0.042 | 0.7874 |
| 4 | 4.156 | 0.88783 | 0.07196 | 0.167 | 0.7874 |
| 3 | 3.156 | 0.73773 | 0.37770 | 0.350 | 0.7874 |
| 2 | 2.156 | 0.22528 | 3.62519 | 1.000 | 0.7874 |

The constant top-10 mass of 0.787 is what settles the normalization question in
deviation 1 above: unnormalized, every EAR in this table would be clipped at
0.787, the 8- and 6-bit rows would be indistinguishable, and `EAR ≥ 0.99` would
be unreachable by a lossless quantizer.

**Non-uniform allocation beats uniform**, same quantizer and calibration, with
uniform interpolated to SLQ's exact bpp:

| bpp | SLQ EAR | uniform at same bpp | gain |
|---|---|---|---|
| 4.399 | 0.92078 | 0.90448 | +0.0163 |
| 4.799 | 0.93434 | 0.92002 | +0.0143 |
| 5.199 | 0.94664 | 0.93556 | +0.0111 |
| 5.599 | 0.95629 | 0.95110 | +0.0052 |

Uniform needs 0.13–0.42 more bits per parameter for the same fidelity. The
solver also independently recovers the paper's K/V-are-most-sensitive finding
(v_proj 6.93 and k_proj 6.50 mean bits, against 4.4–4.5 for q/o/gate).

Two caveats are documented in full in `docs/RESULTS.md`, because either one
reverses the result: grouping coarser than the sensitivity structure (block
rather than per-layer) loses to uniform outright, and the additive prediction
of Eq. 7–8 is invalid below ~5 bpp, where it overestimates EAR by up to 0.52.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers grid geometry against Table 7 and the paper's own worked
examples, the γ² law analytically and empirically, EAR's `1 − d_TV` identity,
the parameter-weighted knapsack, the TL search recovering a known slope, and
SmoothQuant's output-preserving property.

## Citation

```bibtex
@article{helcig2026slq,
  title={Statistically-Lossless Quantization of Large Language Models},
  author={Helcig, Michael and Kurti{\'c}, Eldar and Alistarh, Dan},
  journal={arXiv preprint arXiv:2605.02404},
  year={2026}
}
@inproceedings{xiao2023smoothquant,
  title={SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## License

Apache-2.0.
