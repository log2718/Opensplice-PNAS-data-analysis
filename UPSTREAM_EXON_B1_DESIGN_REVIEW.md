# Upstream-exon-body (B1) → AG mean-logit: design review

Inspection-only report. No code written, no model trained, no AlphaGenome calls, nothing committed.

Repositories inspected:

| Tag | Source | State inspected |
|---|---|---|
| **orig** | `github.com/regev-lab/interpretable-splicing-model-torch` | `main` @ `6535902` (HEAD) |
| **fork** | `github.com/log2718/interpretable-splicing-model-torch` | `main` @ `b51615e`, 29 commits ahead of orig |
| **osplice** | `github.com/log2718/Opensplice-PNAS-data-analysis` + local working dir | `main` @ `e8a2746`, clean |

Throughout: **[FACT]** = verified from code or data in this session. **[REC]** = my architectural recommendation. **[OPEN]** = open scientific or technical question that code cannot answer.

---

## 0. Headline findings

1. **[FACT] `osplice/PNAS_model/` is a byte-identical copy of orig HEAD.** MD5 of all seven files (`model.py`, `predict.py`, `utils.py`, `train.py`, `calibration.py`, `prepare_dataset.py`, `model_weights.pt`) matches orig exactly. There is no OpenSplice-local divergence to reconcile, and full training infrastructure is already present — not inference-only.
2. **[FACT] B1 is the largest single-block main effect in the corrected 8-block run**, and it is ~11× larger than B2 (the collaborator's variable). From `outputs/synthetic_context_8block/factorial/*_main_pairwise_effects.csv`.
3. **[FACT] The existing AG data shows no upper saturation.** Across all 204,800 scored cells, acceptor and donor probabilities never exceed 0.99; only 1.2 % / 1.4 % exceed 0.9; `AG_mean_logit` spans −9.45 to +4.49 with a median of −4.34. The prior experiment lived in the *weak* regime, not on a plateau. The saturation question is genuinely open and must be measured, not assumed.
4. **[REC] Do not transfer the pretrained checkpoint.** Train from scratch; use the original weights only as an explicitly-labelled comparison arm.
5. **[REC] Build a new module inside OpenSplice (Option C)** that imports the AlphaGenome and block-partition code unchanged and re-implements a small PNAS-style network with a regression head.

---

## A. Architecture comparison

### A.1 The original model, traced end to end

All line numbers are `orig/model.py` unless noted; `osplice/PNAS_model/model.py` is identical.

```
x_seq      (N, 4, L)          one-hot ACGT
x_struct   (N, 3, L)          one-hot dot-bracket ". ( )"
x_wobble   (N, 1, L)          1 where position is in a paired G·U / U·G
```

| Stage | Code | Operation | Shape out |
|---|---|---|---|
| Sequence conv | `model.py:260-261` | two **separate** `Conv1d(4 → 20, k=6, padding=0)` — `conv_incl`, `conv_skip` | `(N, 20, L-5)` each |
| Sequence position bias | `model.py:264-266`, applied `358-359` | learned `nn.Parameter(20, L-5)` **added** to each conv output | `(N, 20, L-5)` |
| Structure input | `model.py:362` | `cat([x_seq, x_struct, x_wobble], dim=1)` → **8 channels** | `(N, 8, L)` |
| Structure conv | `model.py:270-271` | two `Conv1d(8 → 8, k=30, padding='same')` | `(N, 8, L)` each |
| Structure position bias | `model.py:272-273`, applied `363-364` | learned `nn.Parameter(8, L)` | `(N, 8, L)` |
| Crop | `model.py:367-368` | `[:, :, 2:-3]` to align with the valid-padded sequence branch | `(N, 8, L-5)` |
| Activation | `model.py:279-280`, `371-372` | `Softplus` on the concatenation of both branches | `(N, 28, L-5)` |
| Aggregation (**pre-tuner**) | `SumDiff`, `model.py:69-90` | `w · (Σ_all incl − Σ_all skip) + b` — sums over **both** filters and positions | `(N,)` |
| Tuner | `ResidualTuner`, `model.py:103-166`, built at `283` | `Linear(1,4) → ReLU → BN → Linear(4,4) → ReLU → BN → Linear(4,1)`, then `+ input` | `(N, 1)` |
| Output | `model.py:284`, `385` | `Sigmoid` → PSI | `(N,)` |

**[FACT] Key structural properties:**

- **The pre-tuner is a scalar.** All 28 filters × (L−5) positions collapse into one number by a signed sum. The tuner is a scalar→scalar function (`Linear(1, …)` at `model.py:119`). This is the whole source of the model's interpretability: everything upstream is an additive, position-resolved energy, and the only non-additive part is a 1-D warp at the very end.
- **Interpretability consequence:** `∂(pre-tuner)/∂(activation at filter f, position p)` is the constant `±w`. A motif's contribution is therefore readable directly off the activation map.
- **Positional information enters only as an additive learned bias per (filter, position).** There is no positional encoding in the convolution, no pooling, and no attention. Pooling is absent entirely — the sum in `SumDiff` is the only reduction.
- **SR balance** (`model.py:508-521`) = `Σ_filters mean_positions softplus(conv_incl + bias_incl) − Σ_filters mean_positions softplus(conv_skip + bias_skip)`. **Sequence branch only** — structure is excluded by construction.
- **Variable-length support: yes, with a caveat.** `PNASModel(input_length=L)` builds correctly-sized position biases for any `L`, and the overridden `load_state_dict` (`model.py:523-597`) Lanczos-resamples checkpoint position biases to the new length (`_resample_position_bias`, `654-683`). It preserves `pad_seq = 15` edge values verbatim at each end and resamples only the middle. Convolution weights are length-agnostic. So the model runs at any length, but the *learned positional prior* is being stretched, which is a modelling assumption, not a free lunch.
- **Encoding** (`orig/utils.py`): `str_to_vector` (`:39-63`) one-hot over `ACGT`, strict — raises on any other character; `folding_to_vector` (`:65-78`) over `".()"`; `compute_wobble_indicator` (`:344-378`) walks the dot-bracket bijection and marks G·U pairs. Structure comes from a ViennaRNA `RNAfold` subprocess (`:133-221`), default `-T 37`, `maxBPspan=0`.
- **Flanks** (`orig/utils.py:17-18`): `LEFT_FLANK = "CATCCAGGTT"`, `RIGHT_FLANK = "CAGGTCTGAC"`, 10 nt each. A 70-nt exon → `L = 90`. This is the default and matches the checkpoint.
- **Training** (`orig/train.py`): target is `metadata_PSI` ∈ [0,1]; loss is `kl_divergence_from_logits` (`:44-49`) = `BCEWithLogits(logits, targets) − BCE(targets, targets)`, i.e. the Bernoulli KL, zero at a perfect fit; Adam, `val_split=0.1`, early stopping on val loss, best-checkpoint saving. README notes BatchNorm misbehaves on validation data and recommends `--no-batchnorm`.
- **Checkpoint** `model_weights.pt`: a plain `state_dict`, 20 tensors, `L = 90` — `position_bias_{skip,incl}` are `(20, 85)` and `position_bias_{skip,incl}_struct` are `(8, 90)`. No optimizer state, no wrapper dict.

### A.2 Comparison table

| Dimension | orig (= osplice copy) | fork |
|---|---|---|
| Files changed vs orig | — | `model.py`, `utils.py`, `train.py`, `prepare_dataset.py`. `predict.py` + `calibration.py` **byte-identical** |
| Flanks | 10 nt / 10 nt, `L = 90` | **40 nt / 30 nt**, `L = 140` default (`model.py:234`); 150-nt-left variant also trained |
| Sequence branch | `Conv1d(4→20,k=6)` ×2 + pos bias | identical |
| Structure branch | `Conv1d(8→8,k=30,'same')` ×2 + pos bias, crop `[2:-3]` | identical, but **bypassable** — stage 1 skips it entirely so no gradient flows |
| Aggregation | `SumDiff` scalar | identical |
| Output head | `ResidualTuner(4)` → sigmoid | same, **plus** stages 1/2 replace it with bare `SumDiff → sigmoid` |
| Uncertainty | none | **`variance_bottleneck = Linear(56→1)`** in `PNASModel` + **`VarianceTuner`**: `Linear(1,16)→ReLU→Linear(16,16)→ReLU→Linear(16,1)→Softplus+1e-6` → `(log_var, var)`. Input is the position-mean of the 56 = 2×28 incl/skip activations |
| `forward` signature | `(x_seq, x_struct, x_wobble, return_logits=False)` | `+ return_uncertainty=False` → `(pred, log_var, var)` |
| Stage control | — | `self.stage ∈ {1,2,3}`, a **plain Python attribute**, not in the state dict |
| Losses | Bernoulli KL only | KL, **`gaussian_nll_const_var`** (logit-space MSE, targets clamped to [1e-2, 1−1e-2]), **`gaussian_nll_learned_var`** (`0.5·(log σ² + (Δ)²/σ²)`) |
| Regularization | none | **L1 on filters** (summed over filters+positions, mean over batch), **`_l2_smooth_loss`** = squared first differences of position-bias tensors |
| Training schedule | single loop | **5 stages** via `_freeze_for_stage` (`train.py:219-246`): 1 seq-only → 2 +struct → 3 full tuner → 4 variance-only (filters frozen) → 5 full joint. Plus `train_monitored.py`, ~15 SLURM scripts, L1 grid search |
| Preprocessing extras | — | **G-quadruplex** (`RNAfold -g`, second regex for `+~`), per-call `--left-flank` / `--right-flank` overrides, annotated-CSV export |
| Analysis infra | — | `performance/` package: `enrich.py`, `analyze.py`, `compare.py`, `lib/{kl,rnafold,plots}.py`; 29 checkpoint directories |
| Checkpoints | one, `L = 90` | many; e.g. `flank_40_30/` is `L = 140` (`position_bias_skip (20,135)`, struct `(8,140)`) and contains **no** variance keys |

### A.3 Fork concerns worth knowing before reusing anything

- **[FACT] Bare `squeeze()` in the fork's forward.** Orig uses `tuner_out.squeeze(-1)`; the fork uses `tuner_out.squeeze()`. With `batch_size == 1` this collapses the batch dimension too and returns a 0-d tensor. Harmless at batch 64, a real bug for single-sequence inference.
- **[FACT] The resampling anchor changed.** `load_state_dict` uses `F = 10` in orig and `F = 30` in the fork (`model.py:594`). This changes how many edge positions are preserved verbatim when resampling a position bias. The same checkpoint loaded at the same target length gives **different** position biases through the two files. Any cross-repo comparison must state which `model.py` did the load.
- **[FACT] `self.stage` is not serialized.** It defaults to 3, so a fresh model behaves like orig, but a staged training run's stage is not recoverable from its checkpoint.
- **[FACT] The fork's uncertainty work is complete and runnable** (losses, variance init to σ²≈1, freeze logic, stage 4/5, SLURM scripts), but the `flank_40_30` checkpoint I inspected has no variance parameters — so not every saved run used it. Which runs are uncertainty runs must be read off `hparams` in each checkpoint, not assumed.
- **[REC] Exclude the uncertainty head from this project initially.** It adds a 56→1→16→16→1 branch and a second loss term to a task where we do not yet know the signal exists. Heteroscedastic variance is worth revisiting *after* a point-prediction baseline works, and the fork's stage-4 recipe (freeze filters, train variance only) will then transfer cleanly. What *is* worth taking now: the L1 and position-bias smoothness penalties, and the freeze/stage pattern.

---

## B. The OpenSplice repository

### B.1 Answers to the six questions asked

1. **How much of the PNAS architecture is present?** All of it.
2. **Identical to upstream?** **[FACT]** Yes — byte-identical across all seven files (MD5-verified).
3. **Do the pre-tuner calculations match the original?** **[FACT]** Yes. `analysis_script/predict_pretuner.py` re-implements `forward` up to `SumDiff` (`get_pretuner`, same convs / biases / `[2:-3]` crop / softplus / stack) and independently recomputes it as `w·(seq_diff + struct_diff) + b` (`get_seq_struct_sumdiff`), printing the max discrepancy as a self-check. The pre-tuner is the `SumDiff` output — the scalar before `ResidualTuner`.
4. **Inference-only vs full training?** Full training is present (`PNAS_model/train.py` is orig's), as are `data/train_data.npz` (47 MB) and `data/test_data.npz` (12 MB). Nothing is inference-crippled.
5. **What training infrastructure is missing?** For *this* task, essentially everything task-specific: a regression target path (the existing loop is hard-wired to `metadata_PSI` with a Bernoulli KL), a dataset builder keyed on an upstream-exon window, and the regularizers from the fork. Generic plumbing (Dataset, epoch loops, early stopping, checkpointing) is reusable.
6. **Can the AlphaGenome utilities be reused directly?** **[FACT] Yes, unchanged.** See below.

### B.2 The AlphaGenome path — already exactly what the new experiment needs

**[FACT]** `analysis_script/synthetic_context_lib.py::score_core` computes the target metric verbatim:

```python
padded, left_pad, _ = ag.center_pad(ag.clean_seq(core))      # → 16,384, 'N' padding, centred
acc_pos = left_pad + exon_start
don_pos = left_pad + exon_start + exon_len - 1
pred = model.predict_sequence(sequence=padded,
                              organism=dna_client.Organism.HOMO_SAPIENS,
                              requested_outputs=[dna_client.OutputType.SPLICE_SITES],
                              ontology_terms=["CL:0002518"])
donor_track, acceptor_track = ag.get_positive_strand_tracks(pred)
...
"AG_mean_logit": (_logit(acc) + _logit(don)) / 2.0
```

with `_logit(p, eps=1e-7)` clipping to `[1e-7, 1−1e-7]`, and `CENSOR_VALUE = log(1e-7/(1−1e-7)) ≈ −16.1` recorded as the lower clip. Supporting helpers live in `alphagenome_local_prediction_pipeline_minigene.py`: `create_model` (reads `.env.alphagenome` via dotenv), `clean_seq`, `center_pad` (`TARGET_LEN = 2**14`), `get_positive_strand_tracks` (filters to positive strand, indexes track names `donor` / `acceptor`).

**[FACT]** `run_scoring` is a production-grade harness: single AlphaGenome stream (quota-safe by design), resumable and keyed on `row_id`, per-call watchdog thread with `call_timeout=90` because the gRPC client retries but never times out, client rebuild on timeout, `flush_every=50`, separate failed-rows CSV. `run_chimera8_experiment.py` adds deterministic contiguous sharding and **re-verifies each construct's SHA-256 against the manifest before spending a call**. This is the single most valuable thing in the repository for the new experiment and should be reused without modification.

### B.3 How the current PNAS model is used — and why the new task is different

**[FACT]** Current usage, two patterns:

- `synthetic_context_lib.pnas_input(exon, up7, dn7)` = `up7 + exon + dn7`, with `PNAS_COMMON_UP7 = PNAS_INTRON1[-7:]` and `PNAS_COMMON_DN7 = PNAS_INTRON2[:7]` — the exact 90-nt input for a 76-nt middle exon (`GTT` + 70-mer + `CAG`).
- `predict_pretuner.py` passes `exon_with_splice_site` directly with `add_flanks=False`, grouping rows by length and instantiating a fresh `PNASModel(input_length=L)` per group so the position biases are resampled.

So in every existing use, **the model's input window is the middle exon plus ~7–10 nt of immediately adjacent intron**, and the prediction is about that same exon. The model has never seen, and has no coordinate frame for, the upstream exon.

The new experiment inverts the relationship:

| | Existing use | New experiment |
|---|---|---|
| Input | middle exon + short intronic flanks | **upstream exon body (B1)**, ~900 nt away from the middle exon's acceptor |
| Target | that exon's own PSI / pre-tuner | AlphaGenome's recognition of a **different, fixed** exon |
| Mechanism | cis-elements inside the regulated exon | trans-acting-at-a-distance effect of a neighbouring exon's sequence |

**[REC] This is a different learning problem, not a different input to the same problem.** The original model's inductive bias — "additive energies from motifs inside the window I am scoring, with a learned prior over position *within that exon*" — is a hypothesis about upstream-exon effects, not a given. That is precisely what makes it interesting to test, and precisely why pretrained position biases should not be inherited.

### B.4 Block geometry and the empirical B1 result

**[FACT]** `analysis_script/splice_site_partition.py` defines the corrected partition. Lengths in the PNAS context, computed from `block_table()`:

| Block | Role | Length (PNAS) | Sequence |
|---|---|---|---|
| **B1** | **upstream exon body** | **113** | — |
| B2 | upstream exon 5′SS (3 exonic + 6 intronic) | 9 | `CAGGTTGGT` |
| B3 | upstream intron body | 101 | — |
| B4 | middle exon 3′SS (20 intronic + `GTT`) | 23 | `TGTCTTTCTCTGCCATCCAGGTT` |
| — | `synthetic70` | 70 | **fixed, not a block** |
| B5 | middle exon 5′SS (`CAG` + 6 intronic) | 9 | `CAGGTCTGA` |
| B6 | downstream intron body | 823 | — |
| B7 | downstream exon 3′SS | 23 | `CTCTTATCTTCCTCCCACAGCTC` |
| B8 | downstream exon body | 138 | — |

`B1 = PNAS_EXON1[:-3]`, i.e. the upstream exon minus the 3 exonic nucleotides that belong to its donor. **B1 has a single fixed length of 113 nt in the PNAS context** — a fixed-length design is available for free if the experiment stays in that context. `assemble8(label, synthetic70, pb, fb)` returns `(core, exon_start)`, and `exon_start` is derived from the assembled left side, so **the middle exon's AG read positions stay correct even if a swapped block changes length** — `score_core` consumes that `exon_start` directly.

**[FACT] Main effects on `AG_mean_logit`** from `factorial/{e6,e7}_main_pairwise_effects.csv` (effect = mean over the 128 contexts with that block F minus the 128 with it P, across 400 exons):

| Block | TP53_e6 arm | TP53_e7 arm |
|---|---|---|
| **B1 (upstream exon body)** | **−1.285** | **−1.365** |
| B4 (middle 3′SS) | −1.198 | −0.866 |
| B8 (downstream exon body) | −1.053 | −1.073 |
| B6 (downstream intron body) | −0.504 | −0.608 |
| B3 (upstream intron body) | −0.197 | +0.461 |
| B5 (middle 5′SS) | −0.104 | +0.375 |
| B7 (downstream 3′SS) | −0.119 | −0.054 |
| B2 (upstream 5′SS) | −0.116 | −0.089 |

B1 is the **largest** main effect in both arms. This confirms the premise.

**[FACT] B1 and B2 are nearly separable**, which matters for the two-researcher split: the `B1:B2` interaction is +0.134 (e6) / +0.132 (e7) with sign flipping in only 11 % / 16 % of the 64 backgrounds — an order of magnitude below B1's main effect. The two studies can proceed independently, with the caveat that neither will capture the small genuine B1×B2 coupling.

### B.5 The saturation question, measured

**[FACT]** My own computation over all 204,800 rows of `chimera8_ag_results.csv`:

| Quantity | Value |
|---|---|
| `AG_acceptor` max / fraction > 0.9 / > 0.99 | 0.984 / 1.19 % / **0.00 %** |
| `AG_donor` max / fraction > 0.9 / > 0.99 | 0.992 / 1.44 % / **0.00 %** |
| `AG_mean_logit` min / median / max | −9.45 / −4.34 / **+4.49** |
| `AG_mean_logit` fraction > 4 | 0.02 % |
| `PPPPPPPP` (native PNAS context), n=800: median / p90 | −2.54 / +3.20 |
| `FPPPPPPP` (B1 swapped only), n=800: median / p90 | −3.23 / +4.29 |

**[FACT] Nothing in the existing dataset is saturated.** The metric never came within 0.01 of the probability ceiling, and the eps clip at ±16.1 was never approached from either side. The prior 400 exons were Dirichlet-random 70-mers and sat overwhelmingly in the *weak* regime.

**[OPEN] Therefore the premise "high PNAS SR balance → AlphaGenome saturation" is untested, and the existing data gives no reason to expect it.** It also gives no reason to rule it out, because no high-SR exon was ever tried. **[FACT]** Nothing in `analysis_script/` references SR balance at all (grep: zero hits) — exon selection by SR balance is new work. **[REC] A small saturation probe must come before any large batch** (step 1 in §F). Note also that `FPPPPPPP` has a *higher* p90 than `PPPPPPPP` (+4.29 vs +3.20), so a B1 swap already reaches further toward the ceiling in the strong tail than the native context does — the dynamic range for B1 at the top end is real but thin in this exon set.

---

## C. Recommended architecture

### C.1 Option evaluation

| Criterion | **A: reuse orig with minimal changes** | **B: adapt the fork** | **C: new module borrowing components** |
|---|---|---|---|
| Suitability for upstream-exon input | Poor. Position biases are shaped `(20, L−5)` for a window centred on the scored exon; nothing is wrong mechanically, but the *semantics* of "position" are silently redefined | Same, and compounded by the 40/30-flank and `F=30` resampling assumptions baked into `utils.py` / `model.py` | Good. Position axis is defined once, explicitly, for the upstream window |
| Interpretability | Inherited and excellent, but the output is forced through sigmoid-PSI semantics the new target does not have | Same | Inherited; `SumDiff` kept as the bottleneck, sigmoid dropped |
| Learning sequence-dependent effects | Yes — see C.3 | Yes, plus L1/smoothness which help on a small dataset | Yes, same conv core |
| Structure / position features still meaningful | Position: redefined. Structure: folding an isolated upstream exon ignores the long-range pairing that would matter | Same | Same physics, but the choice is made explicitly and can be ablated |
| Fixed vs variable length | Handles both via resampling | Both | Fixed 113 nt by design; variable supported if needed |
| Pretrained weights genuinely reusable | No (§E) | No, and its own checkpoints are `L=140` with extra keys | No, and does not pretend to |
| Training complexity / maintainability | Low effort, but requires editing `PNAS_model/train.py`'s target path — **modifies existing inference behaviour**, which you asked to avoid | High: 29 checkpoint dirs, 5-stage schedule, SLURM scripts, variance branch to disable | Moderate; self-contained, existing analyses untouched |

### C.2 **[REC] Option C.** A new `upstream_exon_model/` package.

Rationale: it is the only option that satisfies the hard constraint — *preserve the existing PNAS implementation and all completed analyses, do not modify existing inference behaviour* — while letting the position axis, the output head, and the loss all be defined correctly for a regression target. Options A and B both require editing files that current results depend on. And because the conv core is only ~40 lines, "borrowing components" costs very little duplication relative to the clarity gained.

### C.3 Can the PNAS convolution capture upstream-exon motifs?

**[REC] Yes, and the architecture is unusually well matched to the question** — with one caveat.

Why it fits: `Conv1d(4→20, k=6)` + softplus + signed sum is a learned, position-weighted 6-mer motif counter with an inclusion/skipping polarity. Splicing regulatory elements (SR-protein ESEs, hnRNP ESSs) are 5–8 nt degenerate motifs, so `k=6` is the right scale, and `num_filters=20` is enough for a motif vocabulary while small enough to read off by hand. Crucially, because the pre-tuner is `w·(Σincl − Σskip) + b`, each filter's learned PWM and its position-bias profile are *directly interpretable* — you can ask "which 6-mers in the upstream exon body push the middle exon's recognition up, and does it matter where in the exon they sit?" That is exactly the scientific question.

The caveat: the architecture is **purely additive over positions**. If the upstream exon acts through a mechanism that is not a sum of local motif contributions — competition for a shared spliceosome, exon-definition kinetics, co-transcriptional ordering, or an RNA structure spanning B1 and the middle exon — a sum of 6-mer counts cannot represent it, and the model will underfit in an informative way. **[REC] Treat that underfit as a result, not a failure**, and bracket the interpretable model with a less-constrained baseline (see §F step 3) so you can tell "no signal" from "signal the additive model cannot express".

### C.4 Input representation — what to feed

Separate two things cleanly, as you asked:

- **Scientifically relevant variable:** B1 only, 113 nt. This is what gets randomized. Everything else in the 1,309-nt construct is held fixed.
- **Model input window:** a *computational* choice about how much fixed context the network sees.

Of the three options you listed:

- **(1) Randomized B1 alone (113 nt).** **[REC] This is the right baseline.** The input is exactly the experimental variable; every input position varies across the dataset, so every position bias is identifiable. No constant columns.
- **(2) Full upstream exon including its fixed 5′SS (B1 + B2 = 122 nt).** **[REC] Worth one run as a secondary arm, not the baseline.** B2 is constant across the entire dataset, so its 9 columns are a constant input: the convolution can only learn a bias term from them, and those 9 positions' biases are unidentifiable from the data. It costs nothing and it lets the filters see the junction context that a real donor-proximal element would sit next to — but it buys no additional learnable signal, and it is the collaborator's variable.
- **(3) B1 plus a fixed margin of surrounding sequence.** **[REC] Only if you later want cross-context transfer.** Same identifiability objection as (2), scaled up. Within a single fixed context it adds constant columns and dilutes the position-bias budget.

**[REC] Final recommendation:** baseline on **B1 alone, fixed 113 nt**; pre-register option (2) as a sensitivity check. State in the write-up that the input window is a modelling choice and that B1-only is the choice that makes every parameter identifiable from the randomization.

### C.5 Proposed network

```
input    x_seq (N, 4, 113)                       one-hot ACGT, B1 only
         [optional arm] x_struct (N,3,113), x_wobble (N,1,113)

         conv_incl : Conv1d(4 → 20, k=6, padding=0)   + position_bias_incl (20, 108)
         conv_skip : Conv1d(4 → 20, k=6, padding=0)   + position_bias_skip (20, 108)
                                   ↓ Softplus
         SumDiff:  z = w · ( Σ_{f,p} a_incl − Σ_{f,p} a_skip ) + b        → (N,)
                                   ↓
         head:     ŷ = α·z + β       (Linear(1,1), identity output)       → (N,)

target   AG_mean_logit  ∈ ℝ  (unbounded)
loss     MSE, or Huber if the tails are heavy
```

**[REC] Design choices and why:**

- **Keep `SumDiff` as a scalar bottleneck.** It is the entire interpretability story. Widening it to a vector would buy accuracy and lose the reason to use this architecture at all.
- **Replace `ResidualTuner → Sigmoid` with a plain `Linear(1,1)`.** The target is an unbounded logit, not a probability; a sigmoid would impose a [0,1] range the target does not have, and `ResidualTuner` was fitted to calibrate PSI. A linear rescale keeps `z` monotone in the prediction, which preserves "pre-tuner = interpretable energy". **[REC]** If the pilot shows a genuinely non-linear, compressive relationship (i.e. real saturation), reintroduce a *fresh* `ResidualTuner(hidden_units=4, use_batchnorm=False)` — that is precisely what it is for, and it will then be a *measurement* of saturation rather than an assumption. Use `--no-batchnorm`, per the original README's own warning.
- **Loss: MSE on `AG_mean_logit`.** Not the Bernoulli KL — the target is not a probability. The fork's `gaussian_nll_const_var` is mathematically MSE-in-logit-space and is the closest existing precedent, but it clamps PSI then logit-transforms; our target *is already* a logit, so plain MSE on it is the correct simplification. **[REC]** Exclude or flag any row at the eps clip (`|AG_mean_logit| ≈ 16.1`) — those are censored, not measured, and `CENSOR_VALUE` already exists for exactly this.
- **Regularization: borrow the fork's two penalties.** L1 on filter weights (sparser, more readable motifs) and `_l2_smooth_loss` on position biases (a smooth positional profile is both more plausible and far easier to plot). On a dataset of a few thousand sequences with 20×108 = 2,160 position-bias parameters, the smoothness prior is close to essential.

### C.6 Structure and position — what is lost by omitting them

**[REC] Baseline sequence-only; add structure as a deliberate second arm.**

- **Omitting RNA structure** loses any contribution from local secondary structure occluding or exposing a motif in the upstream exon. Cost to include is low (`compute_structure` + `compute_wobbles` already exist, RNAfold on 113-mers is fast). But **[OPEN]** there is a real question about *what to fold*: folding B1 in isolation gives a structure that does not exist in the 1,309-nt transcript, and the mechanism most likely to matter — long-range pairing between the upstream exon and the middle exon region — is invisible to any single-sequence fold of B1. Folding the full construct and slicing out B1's channels is more faithful but much more expensive and introduces a dependence on the fixed context. There is no existing code for either choice at this scale. I would not let this block the baseline.
- **Omitting position information** loses the ability to say "elements near the upstream donor matter more than elements at the 5′ end of the upstream exon" — which is one of the more interesting things this experiment could show. **[REC] Keep the position bias.** It is cheap (2,160 parameters), interpretable as a plottable curve, and with B1 uniformly randomized it is identifiable. A position-bias-free ablation is a useful comparison, not a default.
- **[REC] An interpretable sequence-only baseline is not just useful, it is the right primary model.** It is the smallest model that can answer the question, every parameter is readable, and it isolates the sequence hypothesis from structure confounds. Add structure and variance only if the sequence-only model shows signal.

---

## D. Pretrained weights vs training from scratch

**[REC] Train from scratch. Do not initialize from `model_weights.pt`.** Reasons, in order of force:

1. **The position biases are not transferable in any meaningful sense.** **[FACT]** The checkpoint's `position_bias_{incl,skip}` is `(20, 85)` for `L=90`, learned over a window that is *the regulated exon plus 7–10 nt of flank*. Position `p` in that tensor means "p nucleotides from the start of the exon being scored". In the new task, position `p` means "p nucleotides into an upstream exon that sits ~900 nt from the exon being scored". `load_state_dict`'s Lanczos resampling will happily map `(20,85) → (20,108)`, and the tensors will align — but the coordinate frames are unrelated. **This is exactly the "dimensions happen to match" case you warned against.**
2. **The output head must be replaced regardless.** **[FACT]** `ResidualTuner` + sigmoid was fitted to map a pre-tuner energy to a PSI in [0,1] for the PNAS minigene library, under a Bernoulli KL. The new target is an unbounded AlphaGenome logit. Transferring it would be transferring a calibration to a different measurement on a different scale.
3. **The convolution filters are the only genuinely arguable case, and the argument is a scientific hypothesis.** **[OPEN]** The 20 incl / 20 skip 6-mer filters encode ESE/ESS-like motifs learned from exonic sequence. *If* upstream-exon effects run through the same SR/hnRNP vocabulary, they are a good prior; *if* the mechanism is different (structure, exon-definition kinetics, spliceosome competition), they are a misleading one. Which is true is one of the things this experiment is for. Initializing from them pre-commits the answer and makes a negative result uninterpretable.
4. **Distribution shift in the input itself.** The checkpoint saw real/mutagenized minigene exons under selection. Our B1 will be uniform- or composition-matched random 113-mers. Filters tuned to natural exonic composition are not obviously a good starting point for that.

**[REC] The defensible use of the checkpoint is as a labelled comparison arm, not an initialization.** Specifically:

- **Arm 1 (primary):** from scratch, random init.
- **Arm 2 (hypothesis test):** load *only* `conv_incl.weight/bias` and `conv_skip.weight/bias` — **not** the position biases, **not** `SumDiff`, **not** the tuner — and either freeze them or fine-tune at a low LR. If Arm 2 beats Arm 1, that is evidence that the upstream exon acts through the same motif vocabulary as within-exon regulation, which is a genuine finding. If it does not, you have learned something too.
- **[FACT]** The machinery for Arm 2 already exists: `PNASModel.load_partial_state_dict` (`model.py:599-652`) loads subsets with `strict=False` and logs each parameter as `[LOAD]` / `[INIT]` / `[SKIP]`. Strip the checkpoint to the four conv keys and hand it over. The fork's `_freeze_for_stage` pattern (`train.py:219-246`) is the model for the freeze.

**[REC] Fine-tuning the full checkpoint is not appropriate** as a first move: with a replaced head, a redefined position axis, and a new loss, "fine-tuning" is just training from scratch with a biased, hard-to-describe initialization.

---

## E. Recommended repository structure

Keep everything in OpenSplice, add one self-contained package, touch nothing that exists.

```
Opensplice-analysis/
├── PNAS_model/                      ← UNTOUCHED. byte-identical to upstream; all
│                                      existing pre-tuner results stay reproducible
├── analysis_script/                 ← UNTOUCHED. imported, never edited
│   ├── splice_site_partition.py           → import: pnas_blocks8, assemble8, validate
│   ├── synthetic_context_lib.py           → import: score_core, run_scoring, _logit,
│   │                                               CENSOR_VALUE, PNAS_* constants
│   └── alphagenome_local_prediction_pipeline_minigene.py
│                                          → import: create_model, clean_seq,
│                                            center_pad, get_positive_strand_tracks
│
├── upstream_exon_model/             ← NEW, the whole project lives here
│   ├── __init__.py
│   ├── b1_blocks.py        # thin wrapper over splice_site_partition: substitute a
│   │                       # given 113-mer for B1, return (core, exon_start)
│   ├── randomize.py        # B1 generators: uniform; composition-matched to native
│   │                       # B1; dinucleotide-preserving shuffle; motif-seeded
│   ├── build_manifest.py   # mirrors build_chimera8_manifest.py: enumerate, SHA-256,
│   │                       # de-duplicate, shard, write validation report
│   ├── score_b1.py         # wraps synthetic_context_lib.run_scoring (UNMODIFIED)
│   ├── dataset.py          # B1 string → seq_oh / (optional) struct_oh / wobble → .npz
│   ├── model.py            # UpstreamExonModel (§C.5), borrowed conv+bias+SumDiff
│   ├── train.py            # MSE/Huber regression loop + L1 + position smoothness
│   └── analyze.py          # filter PWM logos, position-bias curves, pre-tuner vs target
│
├── outputs/
│   ├── synthetic_context_8block/    ← UNTOUCHED
│   └── upstream_exon_b1/            ← NEW namespace, all new artefacts
│       ├── pilot/                   # step 1: exon selection + saturation probe
│       ├── manifest/
│       ├── shards/
│       └── models/
└── run_b1_*.sbatch                  ← NEW, modelled on run_chimera8_*.sbatch
```

### Code reuse map

**Import unchanged — never copy, never edit:**

| Path | Symbols | Why |
|---|---|---|
| `analysis_script/synthetic_context_lib.py` | `score_core`, `run_scoring`, `_logit`, `CENSOR_VALUE`, `PNAS_EXON1/INTRON1/INTRON2/EXON3`, `EXON_PREFIX`, `EXON_SUFFIX`, `SYN_LEN` | The target metric and the resumable, watchdogged, quota-safe AG harness. Reusing it verbatim is what makes the new targets numerically comparable to the 8-block run. |
| `analysis_script/splice_site_partition.py` | `pnas_blocks8`, `assemble8`, `BLOCK_ORDER_8`, `BLOCK_ROLE`, `validate` | Canonical B1 definition and the construct assembler that tracks `exon_start` correctly. |
| `analysis_script/alphagenome_local_prediction_pipeline_minigene.py` | `create_model`, `clean_seq`, `center_pad`, `get_positive_strand_tracks`, `TARGET_LEN` | AG client, padding, track extraction. |
| `PNAS_model/utils.py` | `str_to_vector`, `one_hot_batch`, `folding_to_vector`, `compute_structure`, `compute_wobbles`, `compute_wobble_indicator` | Encoders. Note `DNA_ALPHABET` ordering `ACGT` must match whatever the filters were ever compared against. |
| `PNAS_model/model.py` | `PNASModel` (for SR-balance scoring of candidate middle exons at `L=90`), `compute_sr_balance`, `load_partial_state_dict` | Step 1 exon selection, and Arm 2's partial load. **Read-only.** |

**Copy and adapt into `upstream_exon_model/model.py`** (small, and the semantics change, so a shared base class would be a false abstraction):

| Source | What to take |
|---|---|
| `PNAS_model/model.py:260-266` | sequence conv pair + position-bias parameter pattern |
| `PNAS_model/model.py:69-90` | `SumDiff` — take as-is |
| `PNAS_model/model.py:270-273, 362-368` | structure branch + the `[2:-3]` crop, **only** for the optional structure arm |
| `PNAS_model/model.py:103-166` | `ResidualTuner` — only if the pilot shows real non-linearity; instantiate fresh with `use_batchnorm=False` |

**Copy and adapt into `upstream_exon_model/train.py`:**

| Source | What to take |
|---|---|
| `PNAS_model/train.py:24-40` | `PSIDataset` → rename, re-target to a float regression label |
| `PNAS_model/train.py:58-220` | epoch loops, early stopping, best-checkpoint saving — swap the loss |
| **fork** `train.py:74-81` | `_l2_smooth_loss` (position-bias first differences) |
| **fork** `train.py` L1 block (`--l1-lambda`) | filter L1, summed over filters+positions, mean over batch |
| **fork** `train.py:219-246` | `_freeze_for_stage` pattern, for Arm 2's frozen-filter run |

**Explicitly do not bring over:** the fork's `VarianceTuner` / `variance_bottleneck` / `return_uncertainty` / stage-4-5 schedule (defer, §A.3); the fork's `utils.py` 40/30 flanks and `F=30` resampling anchor (wrong geometry for this task); `calibration.py` (bias-only PSI calibration, not applicable to a logit regression).

---

## F. Proposed data flow

```
 ┌─ exon selection (step 1) ──────────────────────────────────────────────┐
 │  candidate middle exons ──► PNASModel(L=90).compute_sr_balance ──►     │
 │  pick a panel spanning low → very high SR balance                      │
 └────────────────────────────────────────────────────────────────────────┘
                                   │  chosen synthetic70 (FIXED thereafter)
                                   ▼
   randomize.py ──► B1 variant (113 nt)          [THE ONLY THING THAT VARIES]
                                   │
                                   ▼
   b1_blocks.py:  pb = pnas_blocks8();  pb["B1"] = b1_variant
                  core, exon_start = assemble8("PPPPPPPP", synthetic70, pb, pb)
                  sha256(core) ──► manifest                 [1,309 nt construct]
                                   │
         ┌─────────────────────────┴──────────────────────────┐
         │ TARGET branch (AlphaGenome)                        │ INPUT branch (model)
         ▼                                                    ▼
  center_pad(core) → 16,384, 'N', centred            b1_variant (113 nt string)
  predict_sequence(SPLICE_SITES, CL:0002518)                  │
  acc = acceptor[left_pad + exon_start]              str_to_vector / one_hot_batch
  don = donor[left_pad + exon_start + exon_len - 1]           │
         │                                           x_seq (N, 4, 113)
  AG_mean_logit = (logit(acc) + logit(don)) / 2       [+ optional x_struct, x_wobble]
         │                                                    │
         └──────────── join on row_id / b1_sha256 ────────────┘
                                   │
                                   ▼
                    UpstreamExonModel  (§C.5)
          conv_incl/skip(4→20,k=6) + position bias → Softplus
                        → SumDiff  (scalar z)
                        → Linear(1,1)
                                   ▼
                     ŷ = predicted AG_mean_logit
                     loss = MSE(ŷ, AG_mean_logit)
```

**Pairing inputs to targets. [REC]** Key the join on a content hash, never on a label — exactly the discipline `build_chimera8_manifest.py` already enforces ("everything is decided by SEQUENCE IDENTITY, never by label"):

1. `randomize.py` emits each B1 variant with `b1_sha256 = sha256(b1_variant)` and a `row_id`.
2. `build_manifest.py` assembles the full core, records `core_sha256`, and de-duplicates: identical B1 strings (collisions are unlikely at 113 nt but free to check) collapse to one AG call.
3. `score_b1.py` passes `row_id`/`core`/`exon_start`/`exon_len` to `run_scoring`, which re-verifies `core_sha256` before spending a call and checkpoints by `row_id`.
4. `dataset.py` joins the results CSV back to the B1 strings on `row_id`, re-asserts `sha256(b1) == b1_sha256`, and writes the `.npz`. The target column is `AG_mean_logit` straight from `score_core` — no re-derivation, so the numbers stay identical to the 8-block run's convention.
5. Split by `b1_sha256` so no duplicate sequence can straddle train/val.

---

## G. Key technical uncertainties to resolve before implementation

**Must confirm first:**

1. **[OPEN] Is saturation actually present for a high-SR exon?** Measured answer for the existing exon set: no (§B.5). Unknown for a deliberately strong exon. **Blocking** — it determines whether the output head needs `ResidualTuner` and whether the target needs a censoring-aware loss. Resolved by step 1.
2. **[OPEN] Single-position AG read vs a small window.** `score_core` reads one index for acceptor and one for donor. Prior experiments held the construct geometry nearly fixed; randomizing 113 nt is a larger perturbation. If AG's predicted peak shifts by ±1–2 nt the point read would understate it. **[REC]** In the pilot, pull a ±5 nt window around both positions and check that the argmax stays put. If it moves, decide on a rule *before* the big batch — and keep reporting the exact-index value too, for comparability with the 8-block data.
3. **[OPEN] Fixed 113 nt, or variable-length B1?** B1 is 113 nt in the PNAS context. **[REC]** Stay fixed-length and PNAS-context-only for the first experiment: it removes the need for position-bias resampling, makes every parameter identifiable, and keeps the readout geometry constant. Length as a variable is a good second experiment, and `assemble8` already handles it correctly.
4. **[OPEN] How many B1 variants, given AG quota?** The 8-block run spent ~155k new calls across 12 sequential SLURM shards, so the scale is proven feasible — but budget and wall-clock are yours to confirm. **[REC]** 113 nt × 20 filters × 108 positions argues for ≥ a few thousand; I would plan ~4,000–8,000 for the main arm plus ~1,000 for composition-matched and dinucleotide-shuffled controls.

**Confirm before the structure arm (not blocking the baseline):**

5. **[OPEN] What sequence to fold for the structure channels?** Isolated B1 (fast, physically unrealistic), B1 + fixed flanks (arbitrary window), or the whole 1,309-nt core with B1's channels sliced out (faithful, expensive, context-dependent). No existing code covers the third at scale. This is the single largest unresolved modelling choice.
6. **[OPEN] Does `RNAfold` on random 113-mers behave?** The fork hit output-parsing issues severe enough to need a second regex for G-quadruplex notation. Random sequence may produce structures the `".()"` encoder handles fine but which are near-degenerate (high ensemble diversity), making the channel mostly noise. Worth a quick distributional check.

**Verified, so not uncertainties:**

7. **[FACT] Original training code exists and is complete** — `PNAS_model/train.py` is upstream's, with data. Nothing needs to be reconstructed.
8. **[FACT] Checkpoint compatibility is understood.** `model_weights.pt` is a 20-key `state_dict` at `L=90`; it loads into `PNASModel(input_length=90)` strictly, and into other lengths via Lanczos resampling. Fork checkpoints are `L=140` and some carry variance keys, so they will **not** load strictly into orig's `model.py`. Per §D we are not transferring either way, except the four conv tensors in Arm 2.
9. **[FACT] AG output localization is settled for the metric as previously defined** — `left_pad + exon_start` and `left_pad + exon_start + exon_len − 1`, positive strand, `SPLICE_SITES`, `CL:0002518`. Item 2 above asks only whether to *additionally* check for peak drift.

---

## H. First three implementation steps

### Step 1 — Exon selection and saturation probe (small pilot, ~100–150 AG calls)

**No training. No new architecture. Cheap, and it decides the rest of the design.**

1. Score PNAS SR balance for candidate middle exons: `PNASModel(input_length=90)`, load `model_weights.pt`, feed `pnas_input(GTT + syn70 + CAG, PNAS_COMMON_UP7, PNAS_COMMON_DN7)`, call `compute_sr_balance(x_seq, agg='mean')`. Draw candidates from the existing `outputs/synthetic_context/synthetic_exon_library.csv` (10,000 exons, already built) and, if its SR range is too narrow, generate SR-enriched 70-mers.
2. Pick a panel of ~5 exons spanning low → mid → high → very high SR balance.
3. For each, score the native PNAS construct (`PPPPPPPP`) plus ~20 uniform-random B1 replacements, via `assemble8` + `score_core`. ≈ 105 calls.
4. Also pull a ±5 nt window around both read positions (uncertainty 2).

**Deliverables:** (a) the empirical SR-balance → `AG_mean_logit` relationship — tested, not assumed; (b) for each exon, the spread of `AG_mean_logit` across 20 random B1s, i.e. whether B1 still moves a strong exon or the response has flattened; (c) a peak-drift check; (d) the chosen exon, with its regime stated.

**Decision gate:** if the chosen exon's 20-variant spread is comparable to measurement noise, the exon is genuinely saturated — which is itself a publishable observation, and the right response is to add a mid-SR exon as a second arm rather than to push on with a flat target.

### Step 2 — Full B1 library, manifest, and sharded scoring

1. `randomize.py`: generators for uniform-random 113-mers (main arm), composition-matched to native B1, and dinucleotide-preserving shuffles of native B1 (controls that separate "composition" from "motif order").
2. `build_manifest.py`, mirroring `build_chimera8_manifest.py`: assemble every core, SHA-256 it, de-duplicate, shard deterministically, and emit a validation report that asserts — for every construct — that `synthetic70` appears exactly once, that `core[exon_start : exon_start+exon_len]` equals `GTT + syn70 + CAG`, that blocks B2–B8 are byte-identical to `pnas_blocks8()`, and that only B1 differs from the native construct.
3. Score with `synthetic_context_lib.run_scoring` **unmodified**, one shard per SLURM job, `--array=0-N%1` to keep a single stream.
4. Sanity check: the subset of variants whose B1 equals native B1 must reproduce the existing `PPPPPPPP` value for that exon from `chimera8_ag_results.csv` to within AG's own determinism. **This is a free end-to-end validation against completed work — do not skip it.**

### Step 3 — Baseline model, against honest comparators

1. `dataset.py` → `.npz` with `seq_oh (N,4,113)` and `AG_mean_logit`; split by `b1_sha256`.
2. Train the §C.5 sequence-only model from scratch: MSE, Adam, early stopping, `--no-batchnorm`, with L1 and position-bias smoothness; sweep both penalties on a small grid.
3. Compare against three references **before** interpreting anything:
   - **mean predictor** (floor);
   - **k-mer ridge regression** (6-mer counts → `AG_mean_logit`) — a position-blind linear comparator. If ridge matches the conv model, the CNN is adding no predictive value over position-blind motif counting, and you should say so. **[CORRECTION]** an earlier version of this line claimed it would show "position adds nothing"; that overstates it. The two models differ in architecture, fitting and capacity, not only in whether position is used, so this comparison cannot isolate the positional bias. Isolating it requires an ablation of the same CNN with and without its position bias.
   - **Arm 2**: the same model with `conv_incl` / `conv_skip` loaded from `model_weights.pt` via `load_partial_state_dict` and frozen — the direct test of whether upstream-exon effects use the within-exon motif vocabulary (§D).
4. Then interpret: filter PWMs as logos, position-bias curves, and the pre-tuner `z` against the target to read off whether the relationship is linear or compressive (the latter being the quantitative signature of saturation).

Structure channels and the fork's variance head come after step 3 shows signal — not before.

---

## Summary of what is asserted on what basis

| Claim | Basis |
|---|---|
| OpenSplice `PNAS_model/` ≡ upstream HEAD | **[FACT]** MD5, 7/7 files |
| Pre-tuner = `SumDiff` output, and OpenSplice computes it correctly | **[FACT]** `model.py:69-90`, `predict_pretuner.py` with self-check |
| Fork adds uncertainty head, 5-stage training, L1 + smoothness, 40/30 flanks, gquad | **[FACT]** diff vs orig; checkpoint tensor shapes |
| Fork has a bare-`squeeze()` batch-1 bug and a changed resampling anchor | **[FACT]** diff vs orig |
| B1 is the largest 8-block main effect; B1×B2 is ~10× smaller | **[FACT]** `factorial/*_main_pairwise_effects.csv` |
| B1 = 113 nt, fixed, in the PNAS context | **[FACT]** `splice_site_partition.block_table()` |
| Existing AG data is not saturated (max prob 0.99, max mean-logit 4.49) | **[FACT]** computed over 204,800 rows |
| AG utilities reusable unchanged, including the exact `AG_mean_logit` formula | **[FACT]** `score_core`, `run_scoring` |
| SR balance is unused in OpenSplice today | **[FACT]** grep, zero hits |
| Option C (new module) over A or B | **[REC]** preserves existing behaviour; lets position axis, head, and loss be redefined correctly |
| Train from scratch; transfer at most the 4 conv tensors, as a labelled arm | **[REC]** position biases are in an unrelated coordinate frame; head and loss both change |
| Feed B1 alone (113 nt); B1+B2 as a sensitivity arm | **[REC]** identifiability — B2 is constant across the dataset |
| Sequence-only baseline; structure and variance deferred | **[REC]** isolate the sequence hypothesis; structure's correct folding window is unresolved |
| Whether a high-SR exon saturates in AG mean-logit | **[OPEN]** step 1 |
| Whether the AG peak drifts when B1 is randomized | **[OPEN]** step 1 |
| What sequence to fold for structure channels | **[OPEN]** blocking only the structure arm |
| Whether upstream-exon effects use the within-exon motif vocabulary | **[OPEN]** the experiment's actual scientific question; Arm 2 tests it |
