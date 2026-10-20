#!/bin/bash
#SBATCH --job-name=b1_seq
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/b1_seq_%j.out
#SBATCH --error=logs/b1_seq_%j.err
#
# Sequence-only PNAS-derived regressor: B1 -> AlphaGenome mean-logit.
#
# Submit from the repository root:
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_seq.sbatch
#
# Resume an interrupted run (reads models/<run>/last.pt):
#     sbatch upstream_exon/slurm/train_seq.sbatch --resume

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna
RUN_NAME=${RUN_NAME:-seq_v1}

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  run=${RUN_NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun python upstream_exon/train_upstream.py \
    --run-name "${RUN_NAME}" \
    --epochs 400 \
    --batch-size 128 \
    --lr 1e-3 \
    --patience 40 \
    --l2-smooth 1e-3 \
    --l1-lambda 0.0 \
    --num-workers 4 \
    --seed 42 \
    "$@"

echo "done: outputs/upstream_exon/models/${RUN_NAME}"
