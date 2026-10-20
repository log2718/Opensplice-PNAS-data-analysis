#!/bin/bash
#SBATCH --job-name=b1_smoke
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/b1_smoke_%j.out
#SBATCH --error=logs/b1_smoke_%j.err
#
# GPU smoke test. Proves the environment, the GPU, the data path and the
# training loop before any long job is queued. Runs 3 epochs on a subsample
# and also executes the unit tests.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/smoke_gpu.sbatch

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "=== environment ==="
echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "=== unit tests ==="
python -m pytest upstream_exon/tests/test_upstream.py -q

echo
echo "=== training smoke test (3 epochs, subsampled) ==="
srun python upstream_exon/train_upstream.py \
    --smoke \
    --run-name smoke_gpu \
    --batch-size 32

echo
echo "SMOKE TEST COMPLETE"
