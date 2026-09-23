# Validation results

Measured with this implementation. Everything here was produced on CPU
(4 cores, no GPU), which bounds the model size but not the conclusions.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3-0.6B` (28 layers, 196 quantizable linears, 440M quantized params) |
| Precision | float32 |
| Calibration | WikiText-2 raw, 8 windows x 256 tokens |
| Quantizer | GPTQ, group size 128, asymmetric |
| Metrics | EAR / KL at top-`K` = 10, conditioned on the top-`K` support |

The paper uses 512 calibration samples on 8B-70B models; these runs are
deliberately smaller, so the absolute numbers are noisier than the paper's and
a 0.6B model is more sensitive to quantization than the models it evaluates.

---

## 1. The gamma-squared variance law (Lemma 3.2)

`gamma = 2M/R` is the centering inefficiency; Lemma 3.2 predicts that a
symmetric grid inflates quantization noise variance by `gamma^2` relative to an
asymmetric one. Per-projection means over all 28 blocks, group size 128:

| projection | mean gamma | gamma^2 (predicted) |
|---|---|---|
| q_proj | 1.0969 | 1.2032 |
| k_proj | 1.0948 | 1.1987 |
| v_proj | 1.0980 | 1.2056 |
| o_proj | 1.0916 | 1.1915 |
| gate_proj | 1.0930 | 1.1947 |
| up_proj | 1.0945 | 1.1980 |
| down_proj | 1.1068 | 1.2250 |

Measured symmetric/asymmetric MSE ratio at 4-bit against the prediction:

| layer | gamma^2 | measured ratio |
|---|---|---|
| `layers.0.self_attn.q_proj` | 1.2013 | 1.2475 |
| `layers.5.mlp.gate_proj` | 1.1890 | 1.2175 |
| `layers.14.mlp.down_proj` | 1.2421 | 1.2841 |
| `layers.27.self_attn.v_proj` | 1.1926 | 1.2233 |

The law holds, with the measured ratio running ~3% above the prediction. That
direction is expected: Bennett's high-rate approximation is asymptotic in the
number of levels, and the asymmetric grid additionally clamps its range to
include zero, which costs it a little resolution the idealization does not model.

Real Qwen3 weights are only mildly off-centre (`gamma ~ 1.10`, so a ~22% error
penalty), against the paper's illustrative `gamma = 1.2` giving 44%.

## 2. The law transfers to the activation path

SmoothQuant quantizes activations with a symmetric absmax grid. Post-SiLU
activations are bounded below and long-tailed above, so `gamma` is far larger
there than it is for weights. On such a tensor (`gamma = 1.941`, predicted
`gamma^2 = 3.77`), per-token grids:

| bits | symmetric MSE | asymmetric MSE | ratio |
|---|---|---|---|
| 8 | 1.947e-04 | 5.275e-05 | **3.69** |
| 6 | 3.311e-03 | 8.869e-04 | **3.73** |
| 4 | 5.271e-02 | 1.967e-02 | 2.68 |

The prediction is accurate at 8 and 6 bits and degrades at 4, where the
high-rate assumption stops holding. This is why `slq.quant.act` defaults to
asymmetric activation grids.

## 3. Uniform bitwidth sweep

| bits | bpp | EAR | KL | flip rate | top-10 mass |
|---|---|---|---|---|---|
| 8 | 8.156 | 0.99234 | 0.00106 | 0.0107 | 0.7874 |
| 6 | 6.156 | 0.97107 | 0.00908 | 0.0420 | 0.7874 |
| 4 | 4.156 | 0.88783 | 0.07196 | 0.1670 | 0.7874 |
| 3 | 3.156 | 0.73773 | 0.37770 | 0.3501 | 0.7874 |
| 2 | 2.156 | 0.22528 | 3.62519 | 1.0000 | 0.7874 |

Both metrics are monotone across four orders of magnitude of KL, and at 2 bits
the model collapses completely (every argmax changes). For scale, the paper's
Table 5 reports QTIP at uniform 4-bit on Llama-3.1-8B at KL 0.0172 / EAR 0.9519;
a 0.6B model doing worse at the same bitwidth is the expected ordering.

Qwen3-0.6B needs roughly 8 bits to clear the DL threshold of `EAR >= 0.99`,
against the 5.0-6.6 bpp the paper reports for 8B-70B models. Smaller models are
more sensitive, so they need *more* bits, not fewer.

### Why this sweep settles the normalization question

The top-10 mass is **0.7874** at every bitwidth. Under a literal reading of the
paper's formulae -- truncating to the top-`K` support without renormalizing --
EAR is bounded above by that mass. Every row in the table would be clipped at
0.787: the 8-bit and 6-bit configurations would be indistinguishable, and the
paper's own `EAR >= 0.99` target would be unreachable by any quantizer, including
a lossless one. Conditioning on the support is therefore the only reading under
which the metric does the job the paper asks of it, and it is the reading its
stated identity `EAR = 1 - d_TV` requires. See the README for the full argument;
`normalize=False` recovers the literal form, and `topk_mass` is always reported.

## 4. The search end to end

On the built-in reference model (28 groups, CPU, seconds per run), after the
sensitivity database is built with 2 Shapley permutations.

**Prediction accuracy.** Equations 7-8 predict a configuration's EAR from the
database alone, with no forward pass. Against measurement:

| bits | predicted EAR | measured EAR | error |
|---|---|---|---|
| 8 | 0.99997 | 0.99997 | +0.00000 |
| 6 | 0.99985 | 0.99986 | -0.00001 |
| 4 | 0.99946 | 0.99947 | -0.00001 |
| 3 | 0.99846 | 0.99852 | -0.00006 |
| 2 | 0.99396 | 0.99414 | -0.00018 |

Agreement to within 2e-4 is what makes the binary search over bitwidth budgets
free: it runs entirely on the database.

**Allocation.** Tightening the DL target produces progressively more
conservative, and progressively less uniform, allocations:

| target EAR | bpp | measured EAR | bitwidth histogram |
|---|---|---|---|
| 0.9950 | 2.206 | 0.99772 | {2: 27, 3: 1} |
| 0.9985 | 2.356 | 0.99846 | {2: 25, 3: 2, 4: 1} |
| 0.9995 | 2.906 | 0.99931 | {2: 16, 3: 9, 4: 2, 6: 1} |
| 0.9999 | 4.256 | 0.99972 | {2: 10, 3: 8, 4: 4, 6: 3, 8: 3} |

This is the behaviour the paper's Section 4.1 argues for: the full bitwidth
range gets used, and a binary `{4, 8}` restriction would have to spend more
average bits to hit the same constraint.

These numbers also make two of the fixed bugs visible. With the sensitivity
database flattened by the wrong monotonicity direction, the predicted column
above read a constant 1.0001 and every row of the allocation table collapsed to
the minimum bitwidth regardless of target.

## 5. Does non-uniform allocation actually beat uniform?

This is SLQ's central claim, so it gets a direct test: same model, same
quantizer (our GPTQ), same calibration, same bit budget -- only the allocation
differs. Uniform is interpolated to SLQ's exact bpp, since a uniform
configuration only exists at integer bitwidths.

Per-layer groups (196), bitwidths `{4, 6, 8}`, 1 Shapley permutation:

| bpp | SLQ EAR | uniform at same bpp | gain |
|---|---|---|---|
| 4.399 | 0.92078 | 0.90448 | **+0.0163** |
| 4.799 | 0.93434 | 0.92002 | **+0.0143** |
| 5.199 | 0.94664 | 0.93556 | **+0.0111** |
| 5.599 | 0.95629 | 0.95110 | **+0.0052** |

Equivalently, uniform needs **0.13-0.42 more bits per parameter** to reach the
same fidelity -- a 2-9% larger model. The margin narrows as the budget grows,
which is what one expects: there is less to gain from reallocation once every
layer is already well served.

### The allocation reproduces the paper's Figure 6, partly

Mean assigned bits by projection type at a 5.2 bpp budget:

| projection | mean bits |
|---|---|
| v_proj | 6.93 |
| k_proj | 6.50 |
| up_proj | 5.36 |
| down_proj | 4.93 |
| q_proj | 4.50 |
| gate_proj | 4.50 |
| o_proj | 4.43 |

Appendix C.1 reports that the solver "assigns 8 bits to the most sensitive K/V
projections, 6-7 bits to Q and output projections, and 4-5 bits to the more
robust MLP layers". K and V coming out clearly on top is reproduced here
without any architectural prior -- the solver was given only measured
sensitivities. The Q/O placement is not: the paper puts them mid-range, this
model puts them at the bottom alongside the MLP. That may be a size effect
(0.6B against 8B-70B) or a grouped-query-attention head-ratio effect; it is
recorded as a discrepancy rather than explained.

### Two ways to get this wrong

Both were hit before arriving at the table above, and both are worth stating
because either one inverts the conclusion.

**Coarse grouping destroys the effect.** An earlier run grouped at *block*
granularity, forcing all seven projections of a transformer block to share a
bitwidth. The result was a decisive loss to uniform (EAR 0.553 against 0.895 at
4.156 bpp). That is expected in hindsight: the gain comes from *within-block*
variation -- V at ~7 bits beside O at ~4.4 -- and block grouping cannot express
it. Group at least as finely as the sensitivity structure varies.

**The additive model has a validity floor.** Predicted against measured EAR
for allocations over the full `{2,3,4,6,8}` range:

| budget | predicted | measured | error |
|---|---|---|---|
| 3.2 | 0.842 | 0.324 | **+0.518** |
| 4.156 | 0.909 | 0.647 | **+0.262** |
| 5.0 | 0.944 | 0.923 | +0.021 |
| 6.0 | 0.973 | 0.958 | +0.015 |
| 7.0 | 0.986 | 0.982 | +0.004 |

Equations 7-8 sum per-group costs, but each group's Shapley value is measured
largely against a backdrop of *other groups at* `b_max`. One group at 2 bits is
survivable in that context; ten groups at 2 bits are not, and the sum has no
way to know. Below ~5 bpp the prediction is not merely noisy, it is wrong by
more than half an EAR, and the search confidently selects configurations that
destroy the model.

Restricting the candidate set to `{4, 6, 8}` holds the error at or below 0.02
everywhere. The paper's own operating range sits inside the valid region -- its
kernels are 4-8 bit and it reports DL at 5.0-6.6 bpp -- so it never encounters
this, but the boundary is not stated there and it matters for anyone extending
`B` downward.

As a correctness check, prediction is *exact* at uniform bitwidths
(error 0.00000 at 8, 6 and 4 bits), which is Shapley's efficiency property:
the marginals of a full switch telescope to the total.

## 6. Cost

| stage | time |
|---|---|
| Weight bank, 196 layers x 7 bitwidths, GPTQ | 1119 s (3.28 GB) |
| One fidelity evaluation (8 x 256 tokens) | 10-15 s uncontended |

GPTQ's column loop is pinned to a single thread; on this 4-core machine that is
worth ~100x (one 256x512 layer at five bitwidths: 20.7 s -> 0.21 s, identical
numerics). Bank entries are stored as `uint8` codes, so a production run on a
larger model should bit-pack them.

## Reproducing

```bash
python examples/01_gamma_squared_law.py --model Qwen/Qwen3-0.6B   # sections 1
slq sweep --model Qwen/Qwen3-0.6B --bits 2 3 4 6 8 --samples 8 --seq-len 256
python examples/03_smoothquant_activations.py                      # section 2
```
