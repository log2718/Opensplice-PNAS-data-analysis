#!/bin/bash
#SBATCH --job-name=b1_score
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=cpu_public
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1_score_%j.out
#SBATCH --error=logs/b1_score_%j.err
#
# AlphaGenome labeling of the 40,000-sequence B1 library, on the cluster.
#
# This job is NOT compute-bound: it is a single stream of gRPC calls to the
# AlphaGenome API, so it needs OUTBOUND INTERNET from the compute node. It
# requests no GPU on purpose -- one would sit idle for the whole job.
#
# Run check_connectivity.sbatch FIRST. If compute nodes cannot reach the API,
# this job fails fast with a clear message instead of burning its walltime,
# and the fallback is to score locally and rsync the CSV across (see README).
#
# RESUMABLE BY CONSTRUCTION: synthetic_context_lib.run_scoring keys on
# row_id and skips every row already present in the results CSV. A requeued,
# resubmitted or previously-interrupted job continues where it stopped and
# never re-spends a call. Partial results scored elsewhere can simply be
# copied in beforehand and will be honoured the same way.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/score_b1.sbatch
#     MODE=replication sbatch upstream_exon/slurm/score_b1.sbatch
#     DATASET=dataset_v1 sbatch upstream_exon/slurm/score_b1.sbatch

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
    echo "       Copy it from the local checkout (it is gitignored)." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  mode=${MODE}  dataset=${DATASET}"

RESULTS="outputs/upstream_exon/${DATASET}/b1_${MODE}_ag_results.csv"
if [[ -f "$RESULTS" ]]; then
    echo "existing labels: $(( $(wc -l < "$RESULTS") - 1 )) rows -- will resume"
else
    echo "no existing labels -- starting from zero"
fi

# Fail fast if the API is unreachable, rather than consuming the walltime.
echo
echo "--- connectivity preflight ---"
python - <<'PY' || { echo "FATAL: AlphaGenome unreachable from this node." >&2; \
                     echo "       Submit check_connectivity.sbatch for detail, or score" >&2; \
                     echo "       locally and rsync the results CSV (see README)." >&2; exit 1; }
import sys
sys.path.insert(0, "analysis_script")
import alphagenome_local_prediction_pipeline_minigene as ag
ag.create_model(env_path=".env.alphagenome")
print("  AlphaGenome client created OK")
PY

echo
echo "--- planned call count ---"
python upstream_exon/score_b1_dataset.py --mode "${MODE}" --dataset "${DATASET}" --dry-run

echo
echo "--- scoring ---"
srun python upstream_exon/score_b1_dataset.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --flush-every 200 \
    "$@"

echo
echo "done: ${RESULTS}"
echo "rows: $(( $(wc -l < "$RESULTS") - 1 ))"
