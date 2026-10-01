#!/bin/bash
#SBATCH --job-name=os_ctx_sat
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# OpenSplice/FAS-context AlphaGenome saturation experiment.
#
# CPU/API job only -- no GPU. A single AlphaGenome scoring stream (one process)
# to stay within API quota; do NOT run several of these concurrently.
#
# Fully resumable: if Slurm kills the job at the time limit, resubmit this exact
# script and it continues from the last checkpoint (keyed on variant_id). The
# PNAS stage is cached separately in variants_with_pnas.csv and is not redone.

set -eo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

module load anaconda3/2025.06
eval "$(conda shell.bash hook)"

conda activate /scratch/yl12316/conda_envs/opensplicerna

cd /scratch/yl12316/Opensplice-PNAS-data-analysis

export PIP_CACHE_DIR=/scratch/yl12316/pip_cache
export TMPDIR=/scratch/yl12316/tmp

# Makes repo-local AlphaGenome source importable.
export PYTHONPATH="$PWD/alphagenome/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1   # progress lines must appear in slurm-%j.out live

echo "Python: $(which python)"
python --version

# 1. Deterministic variant generation (no API calls). Idempotent: rerunning
#    with the same seed reproduces the identical variant set.
python -u analysis_script/generate_context_saturation_variants.py

# 2. PNAS pre-tuner (local, batched) + AlphaGenome SPLICE_SITES (single stream,
#    checkpointed every 50 calls, 90 s per-call watchdog, 3 retries).
python -u analysis_script/run_open_splice_context_saturation.py \
  --env-path .env.alphagenome \
  --pnas-threads 8 \
  --flush-every 50 \
  --call-timeout 90 \
  --max-retries 3

# 3. Figures (all / low / high views per exon x motif family).
python -u analysis_script/plot_open_splice_context_saturation.py

echo "Finished: $(date)"
