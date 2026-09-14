#!/bin/bash
#SBATCH --job-name=tp53_wide
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

module load anaconda3/2025.06
eval "$(conda shell.bash hook)"

conda activate /scratch/yl12316/conda_envs/opensplicerna

cd "$SLURM_SUBMIT_DIR"

export PIP_CACHE_DIR=/scratch/yl12316/pip_cache
export TMPDIR=/scratch/yl12316/tmp

# Makes repo-local AlphaGenome source importable.
export PYTHONPATH="$SLURM_SUBMIT_DIR/alphagenome/src:${PYTHONPATH:-}"

echo "Python: $(which python)"
python --version

python analysis_script/run_hybrid_pnas_alphagenome.py \
  --alphagenome \
  --env-path .env.alphagenome \
  --out tp53_hybrid_scores_with_alphagenome.csv

python analysis_script/plot_hybrid_pnas_vs_alphagenome.py

echo "Finished: $(date)"
