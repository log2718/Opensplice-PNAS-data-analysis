#!/bin/bash
#SBATCH --job-name=b1_full
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
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
# Evaluation written at the end of the run:
#   metrics.json                 overall + relative + within-group + residual
#   metrics_by_alpha.csv         alpha = 0.5 / 1 / 2 separately
#   metrics_by_composition.csv   GC quantile bins
#   baseline_comparison.csv      constant / composition-ridge / k-mer6-ridge
#   predictions.csv              per-sequence, with group_id and composition
#   training_summary.png         curves + test scatter
#
# PREREQUISITE: the labeled dataset must already be on the cluster at
#   outputs/upstream_exon/dataset_v2/b1_primary_ag_results.csv   (40,000 rows)
# The script refuses to start otherwise rather than training on a partial set.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_full.sbatch
#     sbatch upstream_exon/slurm/train_full.sbatch --resume     # after preemption

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna
RUN_NAME=${RUN_NAME:-full_v2}
EXPECTED_ROWS=${EXPECTED_ROWS:-40000}

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

RESULTS=outputs/upstream_exon/dataset_v2/b1_primary_ag_results.csv
MANIFEST=outputs/upstream_exon/dataset_v2/b1_primary_manifest.csv

for f in "$RESULTS" "$MANIFEST"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f -- transfer the labeled dataset first." >&2
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
    --num-workers 8 \
    --seed 42 \
    "$@"

echo
echo "done -> outputs/upstream_exon/models/${RUN_NAME}"
ls -la "outputs/upstream_exon/models/${RUN_NAME}"
