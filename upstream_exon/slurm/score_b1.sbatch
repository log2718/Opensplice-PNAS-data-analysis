#!/bin/bash
#SBATCH --job-name=b1_score
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=cpu_public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/b1_score_%j.out
#SBATCH --error=logs/b1_score_%j.err
#
# AlphaGenome labeling of the B1 dataset.
#
# IMPORTANT -- read before submitting:
#   This job is NOT compute-bound. It is a single-stream sequence of HTTPS/gRPC
#   calls to the AlphaGenome API, so it needs OUTBOUND INTERNET from the
#   compute node. On many HPC clusters compute nodes are firewalled and this
#   will fail with connection errors. Check first:
#       srun --account=torch_pr_352_general --partition=cpu_public --time=00:05:00 \
#            --pty bash -c 'curl -sS -o /dev/null -w "%{http_code}\n" https://www.google.com'
#   Or submit check_connectivity.sbatch, which answers this directly.
#   If there is no outbound access, run the scorer on a machine that has it
#   (a laptop works: measured ~0.55 s/call, so 40,000 calls is ~6 hours)
#   and copy outputs/upstream_exon/${DATASET}/*.csv to the cluster.
#
# It requests no GPU on purpose -- a GPU would sit idle for the whole job.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/score_b1.sbatch
#     MODE=replication sbatch upstream_exon/slurm/score_b1.sbatch
#
# Fully resumable: re-submitting skips every row already in the results CSV.

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna
MODE=${MODE:-primary}
DATASET=${DATASET:-dataset_v2}

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

if [[ ! -f .env.alphagenome ]]; then
    echo "FATAL: .env.alphagenome not found in $REPO" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  mode=${MODE}"

# Always print the planned call count before spending anything.
python upstream_exon/score_b1_dataset.py --mode "${MODE}" --dataset "${DATASET}" --dry-run

srun python upstream_exon/score_b1_dataset.py --mode "${MODE}" --dataset "${DATASET}" --flush-every 200 "$@"

echo "done: outputs/upstream_exon/${DATASET}/b1_${MODE}_ag_results.csv"
