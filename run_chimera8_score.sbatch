#!/bin/bash
#SBATCH --job-name=chim8_score
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%1
#SBATCH --output=slurm-chim8-score-%A_%a.out
#SBATCH --error=slurm-chim8-score-%A_%a.err

# STEP 2 -- 2^8 chimeras on the same balanced 400-exon subset.
#   195,600 new AlphaGenome calls, split into 12 contiguous disjoint shards
#   of 16,300 calls each.
#
#   %1 in the array spec runs the shards STRICTLY ONE AT A TIME, so there is
#   never more than a single AlphaGenome stream open, exactly as in the
#   6-block run. Do not remove the %1 unless the API quota is known to permit
#   concurrency.
#
#   Per-shard worst case 16,300 x 0.60 s = 2.72 h; the 6 h walltime leaves
#   >2x headroom. A shard that is killed anyway resumes from its own CSV and
#   re-scores nothing.

set -eo pipefail
echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"; echo "Start: $(date)"

module load anaconda3/2025.06
eval "$(conda shell.bash hook)"
conda activate /scratch/yl12316/conda_envs/opensplicerna
cd /scratch/yl12316/Opensplice-PNAS-data-analysis

export PIP_CACHE_DIR=/scratch/yl12316/pip_cache
export TMPDIR=/scratch/yl12316/tmp
export PYTHONPATH="$PWD/alphagenome/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u analysis_script/run_chimera8_experiment.py \
  --shard "$SLURM_ARRAY_TASK_ID" --n-shards 12 \
  --env-path .env.alphagenome \
  --flush-every 50 --call-timeout 90 --max-retries 3

echo "Finished: $(date)"
