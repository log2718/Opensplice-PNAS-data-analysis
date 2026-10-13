#!/bin/bash
#SBATCH --job-name=chim8_merge
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-chim8-merge-%j.out
#SBATCH --error=slurm-chim8-merge-%j.err

# STEP 3 -- merge the 12 shards with the 7 reused 6-block contexts into the
# complete 2 x 256 x 400 table. Makes NO AlphaGenome calls.

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

python -u analysis_script/merge_chimera8_results.py

echo "Finished: $(date)"
