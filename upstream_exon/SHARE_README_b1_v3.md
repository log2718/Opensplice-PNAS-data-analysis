# B1 upstream-exon dataset (v3) — 250,000 sequences

A supervised regression dataset: predict AlphaGenome's splice-site strength
for a **fixed** synthetic middle exon from the sequence of the **upstream exon
body** alone.

## The task

Every row shares an identical genomic context except for one 113-nt window
(`b1`, the upstream exon body). The target is AlphaGenome's prediction for the
middle exon's splice sites — which never change sequence. So the question the
dataset asks is: *how much does upstream exon body sequence shift the model's
read of a downstream splice site?*

Predict `AG_mean_logit` from `b1`. One sequence in, one scalar out.

## Files

`b1_v3_upstream_exon_250k.csv` (~67 MB, ~20 MB gzipped)

| column | meaning |
|---|---|
| `row_id`, `b1_id` | row identifiers |
| `b1` | **the input.** 113-nt upstream exon body, ACGT only |
| `split` | `train` (200,000) / `val` (10,000) / `test` (40,000) |
| `AG_mean_logit` | **the target.** `(logit p_acceptor + logit p_donor) / 2` |
| `AG_acceptor_probability`, `AG_donor_probability` | the raw AlphaGenome probabilities behind the target |
| `AG_mean_prob` | their arithmetic mean, in probability space |
| `gc`, `entropy_bits`, `max_homopolymer`, `count_{A,C,G,T}` | sequence QC, computed from `b1` |

**Use the provided `split`.** It is assigned by sequence hash upstream; a fresh
random split gives a different test set and the numbers below stop being
comparable.

## Two properties of this library that change how you model it

**1. Mononucleotide composition is constant by construction.** Every `b1` is a
permutation of the same multiset: 29 A / 28 C / 28 G / 28 T. So `count_*` and
`gc` carry no usable signal — they are the same for every row, and any model
that leans on base composition has nothing to learn. This is deliberate: it
forces models onto motif and positional structure rather than GC content.

Dinucleotide composition is *not* fixed, though — permuting a fixed multiset
does not fix adjacent-pair counts. A dinucleotide ridge gets R² ≈ 0.40, so
treat that as the floor a sequence model has to clear, not zero.

**2. The target is quantized.** AlphaGenome returns probabilities on a ~1/256
grid, and the target averages two of them, so the measured axis is a ladder of
389 distinct values with a median gap of ~0.005 logit. Scatter plots band
vertically; that is the assay, not your model. It also caps achievable R².

## Benchmarks on the provided test split (n=40,000)

| model | params | test R² | R² on the beyond-dinucleotide residual |
|---|---|---|---|
| constant | — | 0.000 | — |
| dinucleotide ridge | — | 0.396 | — |
| 6-mer ridge (α=1000) | 4,096 | 0.672 | — |
| interpretable CNN | 5,324 | 0.681 | 0.471 |
| BiLSTM (2×128, bidirectional) | 598,273 | **0.820** | **0.702** |

The BiLSTM is a capacity control, not a proposal — it says ~82% of the
variance is reachable from these 113 nt, so the gap below it is architectural.
It converged by early stopping (best epoch 28, patience 12), train R² 0.907 vs
val 0.824, so there is mild overfitting but no sign that much more is available
from scale alone.

Known weakness in all of the above: above ~4.5 measured logit the training data
thins out and every model flattens. The BiLSTM's maximum prediction is 5.29
against a measured maximum of 10.83. Anything that depends on the
strong-splice-site extreme is not supported by this dataset as-is.

## Caveat on what the labels are

These targets are **AlphaGenome predictions, not measurements.** Nothing here
was assayed in cells. A model fit to this data learns what AlphaGenome does
with upstream exon sequence, which is a question about the predictor, not
directly about splicing.
