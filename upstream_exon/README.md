# `upstream_exon/` — upstream-exon (B1) modeling experiment

Learns how the **upstream exon body** of the PNAS three-exon cassette changes
**AlphaGenome's recognition of a fixed middle exon**.

```
randomized B1 (113 nt)  ->  PNAS 1,309-nt cassette  ->  AlphaGenome SPLICE_SITES
                                                     ->  y = (logit p_acc + logit p_don) / 2
B1 one-hot (N,4,113)    ->  PNAS-derived CNN        ->  predicted y
```

Nothing in `PNAS_model/` or `analysis_script/` is modified — they are imported
read-only, so the original model, the pilot results and every completed
analysis stay reproducible.

---

## Datasets

| | `dataset_v1` | `dataset_v2` | **`dataset_v3`** (current) |
|---|---|---|---|
| Sequences | 10,000 | 40,000 | **250,000** |
| Sampling | one Dirichlet vector per sequence | one vector per group, 4 sequences each | **random permutation of a fixed multiset** |
| Composition | varies | varies (matched in expectation within a group) | **identical everywhere: 29 A / 28 C / 28 G / 28 T** |
| Groups | — | 10,000 | — |
| α mixture (20/40/40) | 2,000 / 4,000 / 4,000 seqs | 8,000 / 16,000 / 16,000 seqs | — (no α axis) |
| Splits | 80/10/10 by sequence | 27,000 / 3,000 / 10,000, by group | **200,000 / 10,000 / 40,000** |
| Labeled | 300 (preserved) | 40,000 (complete) | pilot 1,000, then full |

**v1 is preserved in full**, including its 300 labeled examples. v2 uses a
different seed stream and explicitly excludes every sequence already used by
the pilot or v1 (10,038 of them), so the two libraries are disjoint by
construction — verified, not assumed.

### Why grouped sampling

Four sequences drawn from one `p ~ Dirichlet(α,α,α,α)` share nucleotide
probabilities but not realized counts. Measured on the built library:
**within-group GC spread 0.0384 vs 0.2205 between groups** — a 5.7× separation.

So a *within-group* response difference isolates nucleotide **order** from
nucleotide **content**. This is what replaces a separate composition-matched
arm, and it is the sharpest available test that the CNN is not just a
composition readout. It matters: on the v1 data a composition-only ridge
already reaches test R² ≈ 0.35, and the CNN's R² on the composition
*residual* was **negative** at n = 247.

### Why fixed composition (v3)

The v2 analysis showed that composition, not motifs, dominated the response:
a gradient-boosted model on **three base frequencies reached test R² 0.721**,
beating both the CNN (0.668) and the 6-mer ridge (0.709), and the k-mer sweep
saturated at that same ceiling. The v2 grouped design matched composition only
*in expectation*; realized counts still varied within a group.

v3 removes the axis outright. Every sequence is a permutation of the same
multiset, so base counts, GC and sequence entropy are **constant across the
whole library** and cannot explain anything. Whatever is left is nucleotide
**order**: dinucleotide structure, k-mers, and position.

113 is not divisible by 4, so one base takes the extra copy. A is chosen so
that G and C stay equal at 28 each — the dominant v2 filter was a (G−C)
counter, and this denies it an asymmetry to read.

Dinucleotide frequencies are deliberately *not* fixed: they are part of the
order signal being measured. A library that also fixed them would be a
stricter, different experiment.

**Is there anything left to measure?** Estimated from data already paid for,
rather than assumed: among v2 sequences that happen to share exactly the same
base counts, the pooled SD of `AG_mean_logit` is **0.368** (6,468 sequences in
3,105 composition cells), about 30% of v2's total variance. AlphaGenome
quantizes probabilities to ~1/256, which near p ≈ 0.95 is ~0.041 logit, so
that residual is roughly **9 quantization steps** — measurable, but coarse
enough that the 1,000-sequence pilot checks it directly before the full run.

---

## Layout

| File | Purpose |
|---|---|
| `b1_dirichlet.py` | Dirichlet generators — `generate()` (v1 flat), `generate_grouped()` (v2) + QC stats |
| `build_b1_dataset.py` | v1 manifest and splits |
| `build_b1_dataset_v2.py` | **v2 manifest, grouped splits, 32 validation checks** |
| `score_b1_dataset.py` | AlphaGenome labeling via the existing `run_scoring` |
| `model_upstream.py` | `UpstreamExonModel` — PNAS-derived interpretable regressor |
| `dataset_upstream.py` | encoding, torch datasets, leakage assertions |
| `baselines.py` | constant / composition-ridge / k-mer-ridge comparators |
| `train_upstream.py` | training, early stopping, resume, metrics, plots |
| `b1_lib.py` | construct assembly + per-construct validation (shared) |
| `tests/test_upstream.py` | **46 unit tests**, no network or GPU |
| `slurm/*.sbatch` | connectivity check, scoring, GPU training |

Outputs under `outputs/upstream_exon/`:

```
dataset_v1/   10,000-row manifest + 300 labeled rows        [PRESERVED]
dataset_v2/   b1_primary_manifest.csv (40,000)
              b1_primary_manifest_meta.json
              b1_primary_validation_report.txt
              b1_primary_ag_results.csv      <- labels land here
models/<run>/ best.pt, last.pt, config.json, history.csv, predictions.csv,
              metrics.json, metrics_by_alpha.csv, metrics_by_composition.csv,
              baseline_comparison.csv, training_summary.png
```

---

## The model

The PNAS architecture with three changes, each forced by the task. `SumDiff`
and `ResidualTuner` are **imported from `PNAS_model/model.py`**, not
reimplemented.

```
x_seq (N,4,113)
  -> conv_incl / conv_skip : Conv1d(4 -> 20, k=6, valid)      (N,20,108)
  -> + position_bias_incl / position_bias_skip                 (20,108)
  -> Softplus
  -> SumDiff : w*(Σ incl − Σ skip) + b        <- the "pre-tuner" scalar
  -> Linear(1,1)                              <- replaces sigmoid
```

**5,324 parameters**, trained from scratch. Position biases start at zero:
index *p* now means "p nt into the upstream exon body", so the pretrained
biases would encode a prior about a different coordinate system.

Two initialization details, both regression-tested:

- `SumDiff` sums ~2 × 20 × 108 softplus terms, so an `N(0,1)` weight makes the
  untrained energy enormous and MSE unoptimizable. `w` starts at `1e-3`.
- `calibrate_head_()` least-squares-fits the affine head at init so training
  starts at the target's mean and scale.

The target is z-scored on **train statistics only**; every reported metric is
converted back to logit units, so model and baselines are directly comparable.

### Evaluation produced automatically

- overall MSE / MAE / R² / Pearson / Spearman, per split
- **per α** (0.5 / 1 / 2) — `metrics_by_alpha.csv`
- **per GC quantile bin** — `metrics_by_composition.csv`
- **within-group** (order sensitivity at matched composition) — in `metrics.json`
- **beyond-composition**: model R² on the composition-residualized target
- **relative**: pairwise sign agreement and pairwise correlation
- baselines: constant, composition ridge, k-mer-6 ridge

---

## Commands

### 1. Build and validate the library (no API calls)

```bash
python upstream_exon/build_b1_dataset_v2.py --validate-only
python upstream_exon/build_b1_dataset_v2.py
```

### 2. SLURM resources

The training job requests **no partition and no GPU** — only
`--account=torch_pr_352_general` — so it lands on the cluster default
partition. This is evidence-based, not assumed: every AlphaGenome scoring and
analysis job in this repo (`run_tp53_wide_hybrids.sbatch`,
`run_chimera8_score.sbatch`, `run_synthetic_chimeras.sbatch`, …) declares the
account alone, and those jobs ran to completion on CPU nodes
`cs655` / `cs667` / `cs695` / `cs787` (committed `slurm-177*.out` logs).

Two things to keep straight:

- `torch_pr_352_general` is a SLURM **account**, not a partition.
- `cpu_public` is **not** a partition here — submission fails with
  `Error partition 'cpu_public' is not valid for this job`.

`--partition=l40s_public --gres=gpu:l40s:1` also validates, but it reserves an
L40S that a CPU-only torch build never touches and it queues behind GPU
demand — job `21316880` sat pending with `Reason=None` for exactly that
reason. It remains the fallback if the default partition ever refuses the job:

```bash
sbatch --partition=l40s_public --gres=gpu:l40s:1 \
       upstream_exon/slurm/train_full.sbatch
```

To see the full picture for yourself (read-only, submits nothing):

```bash
bash upstream_exon/slurm/probe_slurm.sh | tee slurm_probe.txt
```

The **scoring** scripts still carry the `l40s_public` directives. Scoring is
finished (all 40,000 labels are in), so they were left untouched rather than
re-tested.

### 3. Submit the pipeline

```bash
cd /scratch/yl12316/Opensplice-PNAS-data-analysis
bash upstream_exon/slurm/submit_pipeline.sh          # dry run, prints the plan
bash upstream_exon/slurm/submit_pipeline.sh --go     # submit
```

| stage | script | resources |
|---|---|---|
| connectivity check | `check_connectivity.sbatch` | l40s_public, 4 CPU, 12G, 15 min |
| scoring × 4 chunks | `score_b1.sbatch` | l40s_public, 4 CPU, 12G, 4 h each |
| GPU training | `train_full.sbatch` | l40s_public, 4 CPU, 24G, 4 h |

With 1,800 of 40,000 already labeled, the plan is **4 scoring chunks of
10,000**, chained `afterok` so they run strictly one at a time — never
concurrently against the same results CSV. The scorer also takes an exclusive
lock on the results file as a second guard.

A failed `sbatch` aborts immediately and schedules nothing downstream
(verified). If a running stage fails, downstream jobs sit in
`DependencyNeverSatisfied`:

```bash
squeue -u $USER --format='%.10i %.12P %.14j %.8T %.10M %R'
scancel -u $USER --state=PENDING      # clear a stalled chain
```

Variants:

```bash
bash upstream_exon/slurm/submit_pipeline.sh --go --score-only
bash upstream_exon/slurm/submit_pipeline.sh --go --train-only
CHUNK=5000 bash upstream_exon/slurm/submit_pipeline.sh --go
```

### 4. Fallback: score locally, then transfer

Only if the connectivity check fails:

```bash
python upstream_exon/score_b1_dataset.py --dataset dataset_v2 --flush-every 200
# ~0.55 s/call => ~5.8 h for the remaining 38,200; resumable, safe to re-run

rsync -avP outputs/upstream_exon/dataset_v2/b1_primary_ag_results.csv \
  torch:/scratch/yl12316/Opensplice-PNAS-data-analysis/outputs/upstream_exon/dataset_v2/

ssh torch "wc -l /scratch/yl12316/Opensplice-PNAS-data-analysis/outputs/upstream_exon/dataset_v2/b1_primary_ag_results.csv"
# expect 40001 (40,000 rows + header)

bash upstream_exon/slurm/submit_pipeline.sh --go --train-only
```

`.env.alphagenome` must sit at the **repo root** on the cluster, not in the
dataset directory.

### 4b. The v3 fixed-composition experiment

The v3 manifest is ~150 MB and is **not committed**; it is reproducible from
the seed in ~20 s, and `manifest_fingerprint.json` (committed) proves a
rebuild is byte-identical.

```bash
# on Torch, after pulling
python upstream_exon/build_b1_dataset_v3.py --validate-only   # 9 check groups
python upstream_exon/build_b1_dataset_v3.py                   # writes manifest

# pilot: 1,000 calls (~9 min), drawn from TRAIN so the full run reuses them
mkdir -p logs
PILOT=1 sbatch --array=0-0 --time=01:00:00 upstream_exon/slurm/score_b1_v3.sbatch
python upstream_exon/analyze_b1_v3_pilot.py      # prints GO / NO-GO

# full scoring: 13 shards of 20,000, strictly one at a time (~35-46 h total)
sbatch --array=0-12%1 upstream_exon/slurm/score_b1_v3.sbatch

# training (resumable; resubmit to continue)
sbatch upstream_exon/slurm/train_full_v3.sbatch
```

The pilot's verdict is mechanical: it exits non-zero unless the response has
≥ 20 distinct values and an SD of ≥ 3 quantization steps, and neither splice
site is saturated at p ≈ 1.

Baselines change for v3 (`--baseline-set fixed_composition`). The composition
ridge is **degenerate** there — its features have zero variance, so it can
only predict the training mean — and is replaced by a **dinucleotide ridge**.
The 6-mer ridge uses a streaming Gram solver: a 200,000 × 4,096 design is
3.3 GB as float32 and sklearn's `StandardScaler` + `RidgeCV` path copies it
twice more, so the features are folded into `X'X` (134 MB) in chunks instead
and every penalty is solved against one eigendecomposition.

### 5. Tests

```bash
python -m pytest upstream_exon/tests/test_upstream.py -q       # 46 tests, ~6 s
python -m pytest upstream_exon/tests/test_upstream_v3.py -q    # 27 tests, ~2 s
```

---

## Scoring budget

One AlphaGenome call per sequence. Measured **~0.59 s/call**, single stream.

| | calls | wall clock |
|---|---|---|
| v2 full library | **40,000** | **~6.5 h** |
| v1 (already done) | 300 | ~3 min |
| Replication exon (later) | 10,000 (v2 test split) | ~1.6 h |

Fully resumable — `run_scoring` keys on `row_id`, so an interrupted job
re-reads the CSV and continues. Re-running never re-spends a call.

---

## Known caveats

1. **Composition is a strong predictor in any Dirichlet library.** Library GC
   spans the full range. The grouped design addresses this directly (within-group
   analysis), but *beating the constant baseline does not demonstrate motif
   learning* — only beating the composition and k-mer ridges, and a positive
   within-group correlation, do.
2. **Extreme sequences are kept, by instruction.** 7,298/40,000 have a
   homopolymer ≥ 10; 10,069 have a single base at ≥ 60%; 14,295 have < 30
   distinct 3-mers. Recorded as `flag_*` columns; nothing is filtered.
3. **Cryptic splice sites** were cleared on a 43-construct pilot subset
   (`B1_CRYPTIC_CHECK.md`, Spearman +0.162, p = 0.339). The v2 library is more
   compositionally extreme, so that screen is worth repeating on a sample.
4. **Partition and GPU names are unverified from here.** Torch SSH offered
   only password auth, so `l40s_public` / `cpu_public` could not be checked.
   Run `sinfo -s` before submitting. GPU scripts request the explicit
   `--gres=gpu:l40s:1`; if that form is rejected, change it to `--gres=gpu:1`.

6. **`--requeue` does not resume on its own.** It only re-runs the script.
   `train_full.sbatch` therefore detects `models/<run>/last.pt` itself and
   adds `--resume`, so a requeued job continues rather than silently
   restarting at epoch 1. Verified end to end: a 12-epoch run resumed at
   epoch 13 with optimizer state and best-val preserved, producing a
   contiguous 1..20 history. Force a clean start with `FRESH=1`.
5. **Structure arm folds B1 in isolation** (`--use-structure`), which is not
   how it folds inside the 1,309-nt transcript. Recorded in `config.json`.
