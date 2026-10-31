#!/bin/bash
#SBATCH --job-name=b1v3_score
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1v3_score_%A_%a.out
#SBATCH --error=logs/b1v3_score_%A_%a.err
#
# AlphaGenome labeling of the v3 fixed-composition B1 library (250,000 seqs).
#
# ---------------------------------------------------------------------------
# NO --array DIRECTIVE HERE, ON PURPOSE
# ---------------------------------------------------------------------------
# The array spec is given at SUBMIT time, because the pilot and the full run
# want different ones and a baked-in array would silently queue nine useless
# tasks behind a 1,000-call pilot:
#
#   pilot  (1,000 calls, ~9 min):
#     PILOT=1 sbatch --array=0-0 --time=01:00:00 \
#             upstream_exon/slurm/score_b1_v3.sbatch
#
#   full   (250,000 calls, ~35-46 h of streaming, 13 shards of 20,000):
#     sbatch --array=0-12%1 upstream_exon/slurm/score_b1_v3.sbatch
#
#   13 x 20,000 = 260,000 >= 250,000; the last shard simply finds fewer rows
#   pending and exits. An extra no-op shard costs nothing; a short array that
#   leaves the library unfinished costs another submission.
#
# %1 runs the shards STRICTLY ONE AT A TIME. Only one AlphaGenome stream may
# be open at once -- the quota is not known to permit concurrency -- and the
# scorer additionally takes an exclusive lock on the results CSV and refuses
# to start if another scorer holds it.
#
# ---------------------------------------------------------------------------
# PARTITION
# ---------------------------------------------------------------------------
# No --partition and no --gres: the job lands on the cluster default
# partition. This is evidenced, not guessed. run_chimera8_score.sbatch made
# 195,600 AlphaGenome calls from this same account with no partition and no
# gres, and the committed logs show those jobs running on CPU nodes
# cs655 / cs667 / cs695 / cs787 and writing AlphaGenome results. The API is
# therefore reachable from the default partition.
#
# The older upstream_exon/slurm/score_b1.sbatch asks for an L40S it never
# touches; that was a workaround from before the default partition was
# confirmed, and it is deliberately NOT repeated here. This job is pure
# network I/O: one gRPC stream, negligible CPU, no GPU.
#
# If the default partition ever rejects the job, the GPU allocation still
# works as a fallback:
#   sbatch --partition=l40s_public --gres=gpu:l40s:1 --array=0-9%1 \
#          upstream_exon/slurm/score_b1_v3.sbatch
#
# ---------------------------------------------------------------------------
# RESUME AND COST
# ---------------------------------------------------------------------------
# run_scoring keys on row_id and skips rows already present in the results
# CSV, flushing every 200 calls. Any task that dies, times out or is requeued
# loses at most 200 calls, and the next shard picks up exactly where the last
# one stopped. The pilot's 1,000 labels are in the SAME results CSV, so the
# full run reuses them instead of re-spending them.
#
# Budget: the rates logged across this repo's scoring runs span 0.49 to
# 0.66 s/call (v2 itself averaged ~0.50, older runs were slower), so size the
# shard for the SLOW end, not the fast one:
#   20,000 calls  ~ 2.8 h at 0.50 s/call,  ~3.7 h at 0.66  (5 h walltime)
#   250,000 calls ~ 35 h at 0.50 s/call,  ~46 h at 0.66, across 13 shards.
# A shard that runs out of walltime is not a loss -- the next one resumes
# where it stopped -- but a shard sized for the best-case rate would hit the
# wall routinely, and each wall-clock kill wastes the queue wait.
#
#     mkdir -p logs
#     PILOT=1 sbatch --array=0-0 --time=01:00:00 upstream_exon/slurm/score_b1_v3.sbatch
#     sbatch --array=0-12%1 upstream_exon/slurm/score_b1_v3.sbatch

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
DATASET=${DATASET:-dataset_v3}
MODE=${MODE:-primary}
CHUNK=${CHUNK:-20000}
PILOT=${PILOT:-0}

cd "$REPO"
mkdir -p logs

# Use the environment's interpreter directly. `conda activate` is not
# available on compute nodes here, and activation silently fell back to a
# system python without pandas. The interpreter path works unactivated.
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

DSDIR="outputs/upstream_exon/${DATASET}"
MANIFEST="$DSDIR/b1_${MODE}_manifest.csv"
RESULTS="$DSDIR/b1_${MODE}_ag_results.csv"

if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: missing $MANIFEST" >&2
    echo "       Build it first (it is not committed -- 150 MB -- but is" >&2
    echo "       reproducible from the seed):" >&2
    echo "         $PYTHON_BIN upstream_exon/build_b1_dataset_v3.py" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-default}  dataset=${DATASET}  mode=${MODE}"

PILOT_ARG=()
if [[ "$PILOT" == "1" ]]; then
    PILOT_ARG=(--pilot-only)
    CHUNK=100000            # no cap inside the pilot; it is 1,000 rows
    echo "PILOT=1 -> scoring only the manifest rows marked pilot=True"
else
    echo "full run -> up to ${CHUNK} new calls in this shard"
fi

if [[ -f "$RESULTS" ]]; then
    echo "existing labels: $(( $(wc -l < "$RESULTS") - 1 )) rows -- will resume"
else
    echo "no existing labels -- starting from zero"
fi

# Fail fast if the API is unreachable rather than burning the walltime.
echo
echo "--- connectivity preflight (same node type as the scoring run) ---"
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: AlphaGenome unreachable from this node." >&2; \
                     echo "       Run check_connectivity.sbatch for detail." >&2; exit 1; }
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
    "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --max-new-calls "${CHUNK}" --dry-run

echo
echo "--- scoring ---"
srun "$PYTHON_BIN" upstream_exon/score_b1_dataset.py \
    --mode "${MODE}" \
    --dataset "${DATASET}" \
    "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --max-new-calls "${CHUNK}" \
    --flush-every 200 \
    "$@"

echo
echo "rows now: $(( $(wc -l < "$RESULTS") - 1 )) / 250000"
if [[ "$PILOT" == "1" ]]; then
    echo
    echo "--- pilot go / no-go ---"
    "$PYTHON_BIN" upstream_exon/analyze_b1_v3_pilot.py || true
fi
