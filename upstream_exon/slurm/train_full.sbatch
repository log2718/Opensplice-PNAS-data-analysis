#!/bin/bash
#SBATCH --job-name=b1_full
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
# ---------------------------------------------------------------------------
# CPU-ONLY TRAINING
# ---------------------------------------------------------------------------
# The environment has a CPU-only PyTorch build (torch 2.14.0+cpu, cuda False),
# which is fine: the model is 5,324 parameters and trains comfortably on CPU.
# Training is therefore forced with --device cpu, so the run can never silently
# half-initialise a CUDA path.
#
# PARTITION: no --partition and no --gres are requested, so the job lands on
# the cluster default partition. That is not a guess: the AlphaGenome scoring
# and analysis jobs in this repo (run_tp53_wide_hybrids.sbatch,
# run_chimera8_score.sbatch, run_synthetic_chimeras.sbatch, ...) all declare
# only --account=torch_pr_352_general with no partition and no gres, and they
# ran to completion on CPU nodes cs655 / cs667 / cs695 / cs787
# (see the committed slurm-177*.out logs). The default partition therefore
# accepts a CPU-only job from this account.
#
# `cpu_public` is NOT a partition on this cluster -- submitting to it fails
# with "Error partition 'cpu_public' is not valid for this job". Note also
# that torch_pr_352_general is a SLURM *account*, not a partition.
#
# Requesting --partition=l40s_public --gres=gpu:l40s:1 also works, but it
# reserves an L40S this job never touches (torch is a CPU-only build) and it
# queues behind GPU demand -- which is why job 21316880 sat pending with
# Reason=None. If the default partition ever rejects the job, that GPU
# allocation is the fallback:
#   sbatch --partition=l40s_public --gres=gpu:l40s:1 \
#          upstream_exon/slurm/train_full.sbatch
#
# RESOURCES: 8 CPUs / 32G / 12h. Memory is sized for the 6-mer ridge baseline,
# which builds a 27,000 x 4,096 design matrix (~885 MB float64 before the
# scaler copies it). The CNN itself needs well under 2G. If 32G is rejected:
#   sbatch --mem=16G upstream_exon/slurm/train_full.sbatch --skip-baselines
#
# TESTS are NOT run here. A missing pytest killed job 20673283 before training
# started. Tests belong in their own job so a tooling gap can never cost a
# training run, and so a real test failure is never silently swallowed:
#   sbatch upstream_exon/slurm/run_tests.sbatch
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
RUN_NAME=${RUN_NAME:-full_v2}
EXPECTED_ROWS=${EXPECTED_ROWS:-40000}
FRESH=${FRESH:-0}

cd "$REPO"
mkdir -p logs

# Use the environment's interpreter directly. `conda activate` is not
# available on compute nodes here (no conda on PATH, no profile.d hook), so
# activation silently failed and the job fell back to a system python without
# pandas. The interpreter path works without any activation.
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

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

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-default}  run=${RUN_NAME}"

# Keep every numeric backend inside the allocation. Without this each library
# spawns one thread per visible core and they fight each other.
NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" \
       OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"
echo "threads: $NT"

"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__,
      '| cuda available:', torch.cuda.is_available(),
      '| threads:', torch.get_num_threads())"

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
echo "=== training (CPU) ==="
srun "$PYTHON_BIN" upstream_exon/train_upstream.py \
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
    --device cpu \
    --threads "$NT" \
    --num-workers 2 \
    --seed 42 \
    "${RESUME_ARG[@]+"${RESUME_ARG[@]}"}" \
    "$@"

echo
echo "done -> outputs/upstream_exon/models/${RUN_NAME}"
ls -la "outputs/upstream_exon/models/${RUN_NAME}"
