# full_v2: training-integrity verification and scientific analysis

Analysis of the completed upstream-exon (B1) CNN run. **Read-only**: nothing in
`outputs/upstream_exon/models/full_v2/` was modified, no model was retrained,
no AlphaGenome calls were made. All outputs live in this directory.

- Run: `full_v2`, 984 epochs, best checkpoint epoch 944, early-stopped on patience 40
- Target: `AG_mean_logit` = (logit p_acceptor + logit p_donor)/2 for the fixed middle
  exon `syn_a5.0_00482`, **predicted by AlphaGenome — not measured PSI**
- Input: 113-nt randomized upstream-exon body (B1); everything else in the construct fixed

---

## Part 1 — Training integrity

Every check passed. Details below; the headline is that the run is sound and the
reported numbers are reproducible from the saved files.

| # | Check | Result |
|---|---|---|
| 1 | Checkpoints load | `best.pt` and `last.pt` both load `strict=True` into `UpstreamExonModel`, 5,324 params, 10 expected tensors |
| 2 | Resume was genuine | Optimizer step count **104,304 = 984 × 106** (106 = ceil(27000/256)); history contiguous 1–984, no duplicates; epochs 1/100/400 identical to the pre-resume copy |
| 3 | Early stopping correct | Best val loss 0.33387 at epoch 944; `bad_epochs=40` at 984 — stopped exactly on patience |
| 4 | Eval used **best**, not last | Predictions recomputed from `best.pt` match `predictions.csv` to **2.4e-05**; from `last.pt` they differ by **0.15** |
| 5 | Standardization | `mu=3.850885660613`, `sd=0.677390122681` recomputed from train only, matching to 12 digits; inverse transform reproduces `y_true` to 3e-07 |
| 6 | Group-disjoint splits | **0 of 10,000** groups span >1 split; all groups exactly 4 members; 0 repeated sequence hashes; α split 20/40/40 in every split |
| 7 | Baselines clean | Refit from scratch on train only, reproducing every saved R²: 6-mer 0.7087, composition 0.2589, constant −0.0003 |
| 8 | Metrics reproduce | Test R² recomputed 0.667626 vs 0.667626 in `metrics.json`; MSE, within-group r all match |

### Verified by independent recomputation

`regression_metrics`, `within_group_metrics` and all three baselines were
recomputed from `predictions.csv` without reusing the training code's results.
No discrepancy above 1e-6.

### Architecture consistent with the PNAS-derived interpretation

Forward pass confirmed against `model_upstream.py` and the imported, unmodified
`SumDiff`:

```
conv_{incl,skip}(4→20, k=6, valid) + position_bias → Softplus
  → SumDiff: w·(Σ incl − Σ skip) + b → Linear(1,1)
```

One-hot encoding is **ACGT** (`PNAS_model/utils.DNA_ALPHABET`), so logo letter
assignment is correct.

**Sign convention — important and counterintuitive.** The learned head weight is
negative (`head.weight = −0.7589`, `SumDiff.w = +0.09699`), giving an effective
coefficient `c = −0.0736`. So the arm *named* `incl` **lowers** the predicted
AlphaGenome logit, and `skip` **raises** it. The names come from the PNAS
parentage, not from anything learned here; nothing constrains the sign during
regression training. Any reading of these filters must use the effective sign,
which the figures in this directory do.

### Three caveats on the verification

1. **SLURM logs are absent from the Mac** (`logs/b1_full_*.out`), as is the
   `full_v2_e400` pre-resume copy. The resume was instead verified from the
   optimizer step count, which is stronger evidence than a log line anyway.
2. **`assert_no_leakage` only checks sequence identity**, not group membership.
   Group disjointness holds (verified separately above), but the guard in the
   code would not have caught a violation.
3. **The 6-mer ridge's α was at its grid edge.** `RidgeCV` selected α=1000, the
   largest value offered. Re-running with a wider grid selects α=10,000 and
   gives test R² **0.7135** (vs 0.7087). The baseline was mildly
   under-regularized — in the ridge's favour to fix, which strengthens rather
   than weakens the conclusions below.

---

## Part 2 — Predictive performance

### Reported numbers confirmed

| Model | Test R² | Status |
|---|---:|---|
| Composition ridge (linear) | 0.2589 | ✅ reproduced |
| CNN at 400 epochs | 0.5669 | ✅ reproduced |
| CNN, extended (best, ep 944) | **0.6676** | ✅ reproduced |
| 6-mer ridge | 0.7087 | ✅ reproduced (0.7135 with wider α grid) |

### The CNN converged — it is not simply under-trained

| epochs | val loss slope / 100 ep |
|---|---|
| 1–100 | −0.0572 |
| 301–400 | −0.0354 |
| 601–700 | −0.0089 |
| 901–944 | −0.0099 |
| 945–984 | **+0.0039** |

Train−val gap at the best epoch is **+0.0035** (train loss marginally *higher*
than val). There is no overfitting and no remaining downward trend. At 400
epochs the model was genuinely still improving; by 944 it had plateaued.

**This is evidence of convergence under this configuration** (Adam, lr 1e-3
constant, batch 256, `l2_smooth=1e-3`) — not proof of an architectural ceiling.
A different learning-rate schedule, width, or regularization could still move
it. Equally, the diagnostics below suggest the bottleneck is structural rather
than a matter of more epochs. Neither is established by this run alone.

### The key finding: the target is mostly a nonlinear function of base composition

The composition ridge scored only 0.259 — but it is **linear in 3 numbers**.
Giving the same 3 numbers (A, C, G frequencies; T implied) to a nonlinear model
changes the picture completely. All fit on train, scored once on test:

| predictor | features | test R² |
|---|---|---:|
| composition ridge (linear) | 3 | 0.2589 |
| composition, quadratic | 10 terms | 0.4206 |
| composition, cubic | 20 terms | 0.5749 |
| **CNN (5,324 params)** | full sequence | **0.6676** |
| composition, degree-6 poly | 84 terms | 0.7213 |
| 6-mer ridge | 4,096 | 0.7087 |
| **composition, gradient boosting** | **3** | **0.7214** |
| base + dinucleotide freqs, GBM | 19 | 0.7790 |

**A model that cannot see a single motif — only how much A, C and G the
sequence contains — beats both the CNN and the 6-mer ridge.** The k-mer sweep
tells the same story: test R² rises 0.179 → 0.447 → 0.552 → 0.646 → 0.700 →
0.713 for k = 1…6, saturating right at the composition-only ceiling rather than
climbing past it.

So the 6-mer ridge's advantage over the CNN is **not** evidence that it reads
motifs better. Both are largely recovering a smooth nonlinear surface over base
composition, and the ridge's 4,096 correlated features approximate that surface
more flexibly than this CNN's single-scalar bottleneck does.

Structurally, this is baked into the design: the Dirichlet probability vector
sets each sequence's expected composition, and **74% of test variance lies
between probability-vector groups** against 26% within. The experiment's
dominant axis of variation is composition.

### Does the CNN capture anything beyond composition? Yes, but modestly

- With a cubic composition fit (train-fit) removed from both sides, the CNN's
  residual correlates **r = 0.485** with the truth's residual (R² ≈ 0.23).
- A composition-preserving **shuffle test** — permuting each test sequence's
  bases, holding composition exactly fixed — moves the CNN's prediction by only
  0.148 logit SD against 0.521 across sequences. **About 71% of the CNN's output
  is determined by composition alone**, and corr(original, shuffled) = 0.92.
- Within-group (the 4 sequences sharing one probability vector, where
  composition is nearly constant by construction), the CNN reaches r = 0.510,
  slope 0.79, R² 0.242 on test. A composition-only GBM reaches r = 0.458 on the
  same contrast — so the CNN's within-group margin over pure composition is
  real but small.

### By sampling regime and composition

| α | n | true SD | CNN R² | 6-mer R² | composition ridge R² |
|---|---|---|---|---|---|
| 0.5 | 2,000 | 0.604 | 0.636 | 0.739 | 0.061 |
| 1.0 | 4,000 | 0.655 | 0.666 | 0.713 | 0.265 |
| 2.0 | 4,000 | 0.702 | 0.669 | 0.684 | 0.301 |

The ridge's lead is largest at α=0.5 (the most skewed, extreme compositions) and
nearly gone at α=2. By GC decile, the CNN tracks the ridge closely except at
**GC > 0.8**, where it falls well behind (0.456 vs 0.638).

### Tail behaviour: both models shrink, the CNN more

| | bottom 5% | top 5% |
|---|---|---|
| CNN bias | **+0.651** | **−0.417** |
| 6-mer ridge bias | +0.517 | −0.335 |

Predicted SD / true SD is 0.809 (CNN) vs 0.879 (ridge). The slope of true on
predicted is 1.011, so this is variance shrinkage, not a calibration slope error
— ordinary regression-to-the-mean, stronger in the more constrained model.

One target property worth noting: AlphaGenome's output is heavily **quantized**
— 274 distinct values across 10,000 test rows, with the 10 most common covering
70% of them. This places a ceiling on attainable R² that none of these models
reach, so it does not change the ranking.

---

## Part 3 — What the CNN actually learned

### It learned to count bases, not to recognize motifs

The dominant filter, **S04**, carries 69% of the non-compositional prediction
variance. Its weights:

```
      p1     p2     p3     p4     p5     p6
A   0.122  0.057  0.047  0.044  0.084 -0.007
C  -1.904 -1.734 -1.535 -1.407 -1.506 -1.827
G   1.404  1.235  1.030  1.068  1.274  1.475
T   0.378  0.442  0.458  0.295  0.148  0.359
```

Every one of the six positions prefers G and disfavours C, by nearly the same
amount. Its first singular component holds **99.6%** of the weight variance and
the mean correlation between positions is **0.995**. This is not a 6-mer motif
— it is a **(G − C) counter** implemented in a convolution. The second-largest,
S06, is the same shape (C/T versus A, rank-1 share 0.954).

Across all 40 filters, the contribution-weighted mean rank-1 share is **0.836**.
Weighted by influence, the filter bank is dominated by position-independent
base-preference detectors. Consistent with this, **each filter's contribution is
96–99.9% predictable from base counts alone** (`tables/filter_summary.csv`).

### Massive internal cancellation makes per-filter readings unreliable

The per-filter contributions have SDs summing to **33.2 logit units**, while the
final prediction's SD is **0.54** — a **61×** ratio. S04 alone correlates
**−0.998** with the sum of all other filters. The model works by near-exact
cancellation of large opposing terms, so no single filter's contribution
survives in isolation, and reading individual filters as independent "motifs"
is not supportable here. This is a property of the `SumDiff` scalar bottleneck
(2 × 20 × 108 = 4,320 softplus terms summed into one number), not a training
failure.

### The learned positional weights carry almost nothing

Post-hoc interventions, each **affine-recalibrated on train** so only ranking
matters:

| intervention | test R² | corr with full |
|---|---:|---:|
| none | 0.6684 | 1.000 |
| flatten each filter's profile to its own mean | 0.6653 | 0.996 |
| shuffle positions within each filter | 0.6645 | 0.995 |
| zero the position biases entirely | 0.0149 | 0.152 |

**Positional *dependence* is worth ~0.003 R².** Flattening the profile — keeping
each filter's mean offset but destroying all positional structure — costs almost
nothing. Zeroing the biases is catastrophic only because that *also* removes the
per-filter offsets that set the softplus operating points; it is not evidence
that position matters.

The positional-bias heatmaps show this directly: near-horizontal stripes (per-filter
offsets) with little position-to-position structure. Per-filter SD across
position is 0.055 (incl) and 0.084 (skip), against conv weights reaching 3.175.

**These are post-hoc interventions, not retrained ablations.** They show the
learned solution does not use positional information; they do not show that a
model retrained without position biases would perform the same. That remains the
proper ablation and is not implemented here.

### Interpretation caveat

Everything above describes **features of a model fitted to AlphaGenome's
predictions**. These are not experimentally validated binding motifs, and
AlphaGenome's output is itself a model prediction, not a measurement. A (G−C)
preference recovered here is a statement about AlphaGenome's response surface
over random 113-nt upstream exons — nothing more.

---

## Part 4 — Figures

In `figures/`, all from saved predictions and `best.pt` weights:

| file | content |
|---|---|
| `01_learning_curves.png` | loss and R² vs epoch, best epoch and resume point marked |
| `02_predictions_and_residuals.png` | true vs predicted, residuals vs true, CNN vs ridge |
| `03_by_alpha_and_composition.png` | R² by α, by GC decile, and the raw GC dependence |
| `04_within_group.png` | within-group deviations and between-group means |
| `05_filter_contributions.png` | per-filter contribution SD vs prediction SD; non-compositional remainder |
| `06_logos_incl.png`, `06_logos_skip.png` | all 40 filter logos, effective-sign corrected, logit units |
| `07_positional_weights.png` | positional-bias heatmaps, mean contribution by position, intervention results |
| `08_model_comparison.png` | full model ladder |
| `09_kmer_order_sweep.png` | k-mer ridge test R² vs k against the composition-only ceiling |

Tables in `tables/`: `filter_summary.csv`, `filter_positional_uniformity.csv`,
`kmer_order_sweep.csv`.

---

## Part 5 — What this means for the original question

**How does upstream-exon sequence influence AlphaGenome's predicted splice-site
strength for a fixed, high-SR-balance synthetic middle exon?**

1. **Strongly, and mostly through base composition.** Randomizing only the
   113-nt upstream exon body moves AlphaGenome's mean logit over a range of
   1.40–5.54, and a nonlinear function of three base frequencies predicts 72% of
   test variance. The upstream exon matters, and composition is the main channel.

2. **There is a real but secondary order-dependent component.** Holding
   composition exactly fixed and shuffling, the CNN's prediction still moves
   (~29% of its variation), and within shared-composition groups it reaches
   r = 0.51 — above the 0.46 a composition-only model achieves. Adding
   dinucleotide frequencies lifts test R² from 0.721 to 0.779, which also points
   to short-range order carrying signal. Something beyond composition is present;
   its magnitude is modest.

3. **The strong 6-mer ridge does not imply motif logic.** Since a 3-feature
   composition model matches it, the ridge's performance is better read as
   flexible nonlinear fitting of composition than as motif detection. The CNN's
   deficit against it is likewise not evidence that the CNN missed motifs.

4. **The interpretability goal is not met by this run.** The model is dominated
   by a (G−C) counter, its filters are position-independent base-preference
   detectors, its positional weights contribute ~0.003 R², and its internal terms
   cancel at 61:1. The architecture was chosen to make motif and positional
   structure legible; on this data it found almost none to represent.

5. **Scope.** The response variable is AlphaGenome's predicted splice-site logit,
   not measured PSI. Conclusions concern AlphaGenome's behaviour on this
   construct family. One middle exon, one construct backbone, uniform-random B1.

---

## Part 6 — Recommended next steps

Ordered by information gained per unit of work. The first two are diagnostics on
existing data, needing no new AlphaGenome calls.

### 1. The positional-bias ablation, done properly (highest value, cheap)

Retrain the same CNN with `position_bias_*` fixed at zero and excluded from the
optimizer, same seed, same splits. Settles whether positional biases earn their
2,160 parameters. The post-hoc evidence predicts ~no loss — worth confirming,
because a clean null here is publishable and simplifies every later model.

### 2. Condition on composition by design, not after the fact

Current results are dominated by between-group composition variance (74%). Train
and evaluate on the **within-group residual** — the deviation of each sequence
from its own probability-vector group mean — so composition is removed by
construction rather than regressed out. This turns the secondary question into
the primary one, and uses the grouped design as intended. It can be done
entirely on the existing 40,000 labels.

### 3. A fair-capacity comparison before judging the architecture

The CNN (5,324 params, scalar bottleneck) lost to a GBM on 3 features. Before
concluding anything architectural, train (a) this CNN with a wider head, and (b)
a small CNN without the `SumDiff` bottleneck, on identical splits. This separates
"the inductive bias is wrong" from "the bottleneck is too tight" — a distinction
the current run cannot make.

### 4. A composition-matched library (requires new AlphaGenome calls)

The decisive experiment for order effects: sample sequences at **fixed
composition** (e.g. exact 25/25/25/25 counts) and vary only arrangement. Any
remaining signal is order, with no composition confound at all. A few thousand
calls would be enough to size the effect, and the 74/26 variance split suggests
it should be run before any larger library.

### 5. Cross-exon generalization

Everything here concerns one middle exon, `syn_a5.0_00482`. Scoring the same B1
library against a second middle exon (the pilot's `syn_a0.5_00475` is the natural
choice) would show whether the composition response is a property of this
construct or of AlphaGenome's upstream-exon handling generally.

**Not recommended yet:** a larger or deeper sequence model. The limiting factor
is not model capacity — a 3-feature GBM already beats the CNN. It is that the
experimental design puts most variance on a composition axis, where there is
little order structure left to learn.
