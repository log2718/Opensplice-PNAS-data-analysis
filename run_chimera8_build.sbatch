#!/bin/bash
#SBATCH --job-name=chim8_build
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-chim8-build-%j.out
#SBATCH --error=slurm-chim8-build-%j.err

# STEP 1 of the 8-block chimera run -- manifest + 20-point validation.
# Makes NO AlphaGenome calls. Must succeed before the scoring array is submitted.

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

python -u analysis_script/splice_site_partition.py
python -u analysis_script/build_chimera8_manifest.py --n-shards 12

echo "Finished: $(date)"
