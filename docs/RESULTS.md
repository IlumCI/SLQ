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

## 5. Cost

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
