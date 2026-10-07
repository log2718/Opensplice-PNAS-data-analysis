#!/bin/bash
#SBATCH --job-name=syn_chimeras
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# JOB 2 -- 2^6 chimeras on the balanced 400-exon subset: 49,600 AlphaGenome calls.
# Depends on JOB 1 (needs the library + PNAS_common + endpoint results).
# ONE AlphaGenome stream; resumable the same way as JOB 1.

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

python -u analysis_script/run_chimera_experiment.py \
  --env-path .env.alphagenome \
  --flush-every 50 --call-timeout 90 --max-retries 3

python -u analysis_script/analyze_synthetic_context.py --chimeras

echo "Finished: $(date)"
