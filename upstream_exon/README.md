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

| | `dataset_v1` | **`dataset_v2`** (current) |
|---|---|---|
| Sequences | 10,000 | **40,000** |
| Sampling | one Dirichlet vector per sequence | **one vector per group, 4 sequences each** |
| Groups | — | **10,000** |
| α mixture (20/40/40) | 2,000 / 4,000 / 4,000 seqs | 8,000 / 16,000 / 16,000 seqs |
| Splits | 80/10/10 by sequence | **27,000 / 3,000 / 10,000, by group** |
| Labeled | 300 (preserved) | in progress |

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

### 2. Determine valid SLURM resources FIRST

`cpu_public` does not exist on Torch — submission fails with
`Error partition 'cpu_public' is not valid for this job`. Run the probe on a
login node and send the output back so resources can be set from fact:

```bash
bash upstream_exon/slurm/probe_slurm.sh | tee slurm_probe.txt
```

It is read-only (it submits nothing; section 7 uses `sbatch --test-only`) and
reports which partitions exist, which the account may use, whether any accepts
a CPU-only job, and each partition's walltime cap.

**Until that is known**, every script defaults to the one allocation confirmed
to work on this account:

```
--partition=l40s_public --gres=gpu:l40s:1 --cpus-per-task=4 --mem=12G
```

**This means the scoring job holds an L40S it never uses** — it is pure
network I/O to the AlphaGenome API. At ~0.55 s/call that is ~1.5 idle
GPU-hours per 10,000-call chunk, ~5.8 GPU-hours for the remaining 38,200 —
which is why scoring is chunked rather than held as one long reservation. If the probe reveals a CPU-only partition, override it:

```bash
PARTITION=<cpu_part> GRES=NONE bash upstream_exon/slurm/submit_pipeline.sh --go
```

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

### 5. Tests

```bash
python -m pytest upstream_exon/tests/test_upstream.py -q     # 46 tests, ~6 s
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
