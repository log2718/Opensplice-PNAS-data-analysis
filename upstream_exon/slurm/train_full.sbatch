#!/bin/bash
#SBATCH --job-name=b1_full
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1_full_%j.out
#SBATCH --error=logs/b1_full_%j.err
#
# THE full training run: PNAS-derived interpretable CNN on the complete
# 40,000-sequence grouped library (27,000 train / 3,000 val / 10,000 test).
#
# Target : AlphaGenome mean-logit for the fixed middle exon syn_a5.0_00482
# Loss   : MSE (on the train-fitted z-scored target; all reported metrics are
#          converted back to logit units)
# Model  : conv(4->20,k=6) x2 + learned position bias -> Softplus -> SumDiff
#          -> Linear(1,1).  5,324 parameters, trained from scratch.
#
# RESOURCES track the allocation confirmed to work on this account:
#   --partition=l40s_public --gres=gpu:l40s:1 --cpus-per-task=4
# Memory is raised to 24G (above the known-good 12G) because the k-mer
# baseline builds a 27,000 x 4,096 design matrix. If that is rejected, drop
# it -- the CNN itself needs well under 2G:
#   sbatch --mem=12G upstream_exon/slurm/train_full.sbatch --skip-baselines
#
# RESUME: --requeue alone does NOT make the trainer continue from a
# checkpoint; it only re-runs the script. This script therefore detects
# models/<run>/last.pt itself and adds --resume, so a requeued or
# resubmitted job always continues instead of silently restarting from
# epoch 1. Force a fresh start with FRESH=1.
#
# Evaluation written at the end of the run:
#   metrics.json                 overall + relative + within-group + residual
#   metrics_by_alpha.csv         alpha = 0.5 / 1 / 2 separately
#   metrics_by_composition.csv   GC quantile bins
#   baseline_comparison.csv      constant / composition-ridge / k-mer6-ridge
#   predictions.csv              per-sequence, with group_id and composition
#   training_summary.png         curves + test scatter
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_full.sbatch
#     FRESH=1 sbatch upstream_exon/slurm/train_full.sbatch   # ignore checkpoints

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna
RUN_NAME=${RUN_NAME:-full_v2}
EXPECTED_ROWS=${EXPECTED_ROWS:-40000}
FRESH=${FRESH:-0}

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

DSDIR=outputs/upstream_exon/dataset_v2
RESULTS=$DSDIR/b1_primary_ag_results.csv
MANIFEST=$DSDIR/b1_primary_manifest.csv

for f in "$RESULTS" "$MANIFEST"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f -- run the scoring job first." >&2
        exit 1
    fi
done

ROWS=$(( $(wc -l < "$RESULTS") - 1 ))
echo "labeled rows: ${ROWS} (expected ${EXPECTED_ROWS})"
if [[ "${ROWS}" -lt "${EXPECTED_ROWS}" ]]; then
    echo "FATAL: dataset is incomplete (${ROWS}/${EXPECTED_ROWS})." >&2
    echo "       Finish scoring, or set EXPECTED_ROWS to train on a subset." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  run=${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# --- make --requeue actually resume -------------------------------------
RESUME_ARG=()
LAST_CKPT="outputs/upstream_exon/models/${RUN_NAME}/last.pt"
if [[ -f "$LAST_CKPT" && "$FRESH" != "1" ]]; then
    echo "found $LAST_CKPT -> resuming"
    RESUME_ARG=(--resume)
elif [[ "$FRESH" == "1" ]]; then
    echo "FRESH=1 -> ignoring any existing checkpoint"
else
    echo "no checkpoint found -> starting from scratch"
fi

echo
echo "=== unit tests ==="
python -m pytest upstream_exon/tests/test_upstream.py -q

echo
echo "=== training ==="
srun python upstream_exon/train_upstream.py \
    --run-name "${RUN_NAME}" \
    --results-csv "$RESULTS" \
    --manifest-csv "$MANIFEST" \
    --epochs 400 \
    --batch-size 256 \
    --lr 1e-3 \
    --patience 40 \
    --l2-smooth 1e-3 \
    --loss mse \
    --kmer-k 6 \
    --num-workers 4 \
    --seed 42 \
    "${RESUME_ARG[@]+"${RESUME_ARG[@]}"}" \
    "$@"

echo
echo "done -> outputs/upstream_exon/models/${RUN_NAME}"
ls -la "outputs/upstream_exon/models/${RUN_NAME}"
