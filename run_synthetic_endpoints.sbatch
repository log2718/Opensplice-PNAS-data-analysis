#!/bin/bash
#SBATCH --job-name=syn_endpoints
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# JOB 1 -- synthetic library + PNAS_common + 30,000 endpoint AlphaGenome calls.
# CPU/API only, no GPU. ONE AlphaGenome stream; do not run concurrently with JOB 2.
# Fully resumable: resubmit this exact script and it continues from the last
# checkpoint (keyed on row_id). The library and PNAS stage are cached.

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $(hostname)"; echo "Start: $(date)"

module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate /scratch/yl12316/conda_envs/opensplicerna
cd /scratch/yl12316/Opensplice-PNAS-data-analysis

export PIP_CACHE_DIR=/scratch/yl12316/pip_cache
export TMPDIR=/scratch/yl12316/tmp
export PYTHONPATH="$PWD/alphagenome/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u analysis_script/run_synthetic_endpoint_experiment.py \
  --env-path .env.alphagenome \
  --pnas-threads 8 --flush-every 50 --call-timeout 90 --max-retries 3

python -u analysis_script/analyze_synthetic_context.py --endpoints

echo "Finished: $(date)"
