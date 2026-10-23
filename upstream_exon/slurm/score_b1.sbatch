#!/bin/bash
#SBATCH --job-name=b1_score
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1_score_%j.out
#SBATCH --error=logs/b1_score_%j.err
#
# AlphaGenome labeling of the 40,000-sequence B1 library, on compute nodes.
#
# ---------------------------------------------------------------------------
# WHY THIS ASKS FOR A GPU IT WILL NOT USE
# ---------------------------------------------------------------------------
# This job is pure network I/O: a single stream of gRPC calls to the
# AlphaGenome API. It needs no GPU and barely any CPU. But `cpu_public` does
# not exist on this cluster, and the only allocation confirmed to work on
# account torch_pr_352_general is:
#
#     --partition=l40s_public --gres=gpu:l40s:1 --cpus-per-task=4 --mem=12G
#
# so that is what it requests. The cost is real: an L40S sits idle for the
# duration (~3.2 h of GPU time for a 10,000-call chunk at ~0.55 s/call).
#
# If probe_slurm.sh shows a genuine CPU-only partition, override it --
# command-line flags beat the #SBATCH directives above:
#
#     sbatch --partition=<cpu_part> --gres=NONE upstream_exon/slurm/score_b1.sbatch
#
# To keep that waste bounded, the job scores a CHUNK and exits; chain several
# sequentially rather than holding one GPU for six hours.
# ---------------------------------------------------------------------------
#
# RESUME: synthetic_context_lib.run_scoring keys on row_id and skips rows
# already present in the results CSV, and the scorer only rebuilds the
# constructs it still needs. The 1,800 labels already on disk are preserved
# and will not be re-spent.
#
# CONCURRENCY: the scorer takes an exclusive lock on the results CSV and
# refuses to start if another scorer holds it, so two jobs can never
# interleave writes. submit_pipeline.sh chains chunks with afterok, which
# also serializes them.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/score_b1.sbatch
#     CHUNK=5000 sbatch upstream_exon/slurm/score_b1.sbatch
#     MODE=replication sbatch upstream_exon/slurm/score_b1.sbatch

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
MODE=${MODE:-primary}
DATASET=${DATASET:-dataset_v2}
CHUNK=${CHUNK:-10000}

cd "$REPO"
mkdir -p logs

# Use the environment's interpreter directly. `conda activate` is not
# available on compute nodes here (no conda on PATH, no profile.d hook), so
# activation silently failed and the job fell back to a system python without
# pandas. The interpreter path works without any activation.
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f .env.alphagenome ]]; then
    echo "FATAL: .env.alphagenome not found at $REPO" >&2
    echo "       Copy it to the REPO ROOT (it is gitignored)." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-?}  mode=${MODE}  dataset=${DATASET}  chunk=${CHUNK}"

RESULTS="outputs/upstream_exon/${DATASET}/b1_${MODE}_ag_results.csv"
if [[ -f "$RESULTS" ]]; then
    echo "existing labels: $(( $(wc -l < "$RESULTS") - 1 )) rows -- will resume"
else
    echo "no existing labels -- starting from zero"
fi

# Fail fast if the API is unreachable, rather than consuming the walltime.
echo
echo "--- connectivity preflight (same node type as the scoring run) ---"
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: AlphaGenome unreachable from this node." >&2; \
                     echo "       Run check_connectivity.sbatch for detail, or score" >&2; \
                     echo "       locally and rsync the results CSV (see README)." >&2; exit 1; }
import sys
sys.path.insert(0, "analysis_script")
import alphagenome_local_prediction_pipeline_minigene as ag
ag.create_model(env_path=".env.alphagenome")
print("  AlphaGenome client created OK")
PY

echo
echo "--- planned call count ---"
"$PYTHON_BIN" upstream_exon/score_b1_dataset.py \
    --mode "${MODE}" --dataset "${DATASET}" \
    --max-new-calls "${CHUNK}" --dry-run

echo
echo "--- scoring ---"
srun "$PYTHON_BIN" upstream_exon/score_b1_dataset.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    --max-new-calls "${CHUNK}" \
    --flush-every 200 \
    "$@"

echo
echo "rows now: $(( $(wc -l < "$RESULTS") - 1 ))"
