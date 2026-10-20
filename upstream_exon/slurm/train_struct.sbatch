#!/bin/bash
#SBATCH --job-name=b1_struct
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/b1_struct_%j.out
#SBATCH --error=logs/b1_struct_%j.err
#
# Structure-aware comparison arm: the original PNAS structure branch is kept
# verbatim (Conv1d(8 -> 8, k=30, 'same') + position bias + the [2:-3] crop).
#
# CAVEAT recorded in the run config: each 113-nt B1 is folded IN ISOLATION.
# It does not fold alone inside the 1,309-nt transcript, and any long-range
# pairing with the middle exon is invisible to this representation. Treat the
# comparison against train_seq.sbatch as "does local B1 structure add
# anything", not as "does RNA structure matter".
#
# Needs ViennaRNA RNAfold on PATH. Check first:
#     conda activate /scratch/yl12316/conda_envs/opensplicerna && which RNAfold
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_struct.sbatch

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna
RUN_NAME=${RUN_NAME:-struct_v1}

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

if ! command -v RNAfold >/dev/null 2>&1; then
    echo "FATAL: RNAfold not on PATH; the structure arm cannot run." >&2
    echo "       Install ViennaRNA into $ENV, or run train_seq.sbatch instead." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  run=${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
RNAfold --version

srun python upstream_exon/train_upstream.py \
    --run-name "${RUN_NAME}" \
    --use-structure \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-3 \
    --patience 40 \
    --l2-smooth 1e-3 \
    --num-workers 8 \
    --seed 42 \
    "$@"

echo "done: outputs/upstream_exon/models/${RUN_NAME}"
