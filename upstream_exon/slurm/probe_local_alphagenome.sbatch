#!/bin/bash
#SBATCH --job-name=ag_local_probe
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ag_local_probe_%j.out
#SBATCH --error=logs/ag_local_probe_%j.err
#
# Resolve the local alphagenome_pytorch contract: output keys and shapes,
# whether 16,384 is accepted, which of the 734 splice_site_usage tracks is
# CL:0002518, whether usage is in [0,1] or logits, and the native
# construct's logit-average.
#
# Batch rather than `srun --pty` because an interactive allocation dies with
# the SSH connection; this one survives and leaves its output in logs/.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/probe_local_alphagenome.sbatch
#     # then, once it finishes:
#     cat logs/ag_local_probe_<jobid>.out
#
# Makes NO AlphaGenome API calls and writes only
# outputs/upstream_exon/fullctx_probe/local_probe.json.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
# The env with alphagenome_pytorch + the 920 MB weights. NOT opensplicerna,
# which has the API client but not the local model.
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/alphagenome/bin/python}
WEIGHTS=${WEIGHTS:-/scratch/yl12316/models/alphagenome/model_all_folds.safetensors}

cd "$REPO"
mkdir -p logs

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
    echo "FATAL: weights not found: $WEIGHTS" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "python=$PYTHON_BIN"
echo "weights=$WEIGHTS ($(du -h "$WEIGHTS" | cut -f1))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

"$PYTHON_BIN" -c "import numpy,pandas,torch;print('numpy',numpy.__version__,'pandas',pandas.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo

srun "$PYTHON_BIN" upstream_exon/probe_local_alphagenome.py \
    --weights "$WEIGHTS" "$@"
