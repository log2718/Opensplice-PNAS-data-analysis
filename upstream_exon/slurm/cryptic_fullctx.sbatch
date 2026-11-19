#!/bin/bash
#SBATCH --job-name=cryptic_fullctx
#SBATCH --account=torch_pr_352_courant
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/cryptic_%x_%j.out
#SBATCH --error=logs/cryptic_%x_%j.err
#
# Cryptic-splice-site scan in the full 8,612-nt context, on local weights.
#
# Redoes the v3 check (1,309-nt cassette, API, n = 3,000) in the context the
# v4 labels actually use. Both the cryptic features and logit_average come
# from ONE forward pass per sequence, so they describe identical input, and
# every row joins onto ag_results.csv by row_id.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/cryptic_fullctx.sbatch
#     # or a different size / library:
#     N=10000 LIBRARY=upstream sbatch upstream_exon/slurm/cryptic_fullctx.sbatch
#     N=5000  LIBRARY=middle   sbatch upstream_exon/slurm/cryptic_fullctx.sbatch
#
# NOTE ON QUEUE CONTENTION: this wants the same l40s_public GPU as the main
# scoring array. If the array is still pending, submitting this makes it wait
# longer. The scan is ~10k sequences and the labeling run is 1.5M, so the
# labeling run should normally go first -- or submit this with a dependency:
#
#     sbatch --dependency=afterany:<array_jobid> \
#         upstream_exon/slurm/cryptic_fullctx.sbatch
#
# Resumable: the sampled subset is fixed by --seed, and an existing output
# CSV is read for already-scanned row_ids.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
LIBRARY=${LIBRARY:-upstream}
N=${N:-10000}
SEED=${SEED:-20261117}
BATCH=${BATCH:-16}
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/alphagenome/bin/python}
WEIGHTS=${WEIGHTS:-/scratch/yl12316/models/alphagenome/model_all_folds.safetensors}
CATALOG=${CATALOG:-/scratch/yl12316/models/alphagenome/track_metadata_human.parquet}

cd "$REPO"
mkdir -p logs

for f in "$PYTHON_BIN" "$WEIGHTS" "$CATALOG" ExonSkip7_Lib2_text.txt; do
    if [[ ! -e "$f" ]]; then
        echo "FATAL: missing $f" >&2
        exit 1
    fi
done

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  library=${LIBRARY}  n=${N}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo
echo "--- plan ---"
"$PYTHON_BIN" upstream_exon/cryptic_fullctx.py \
    --library "$LIBRARY" -n "$N" --seed "$SEED" \
    --catalog "$CATALOG" --dry-run

echo
echo "--- scan ---"
srun "$PYTHON_BIN" upstream_exon/cryptic_fullctx.py \
    --library "$LIBRARY" -n "$N" --seed "$SEED" \
    --weights "$WEIGHTS" --catalog "$CATALOG" \
    --batch-size "$BATCH" "$@"
