#!/bin/bash
#SBATCH --job-name=fullctx_score
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/fullctx_%x_%A_%a.out
#SBATCH --error=logs/fullctx_%x_%A_%a.err
#
# AlphaGenome SPLICE_SITE_USAGE labeling of a full-context library.
# Target: logit_average = logit((usage_acceptor + usage_donor) / 2).
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE SUBMITTING THE FULL ARRAY
# ---------------------------------------------------------------------------
# 1,000,000 + 500,000 = 1,500,000 calls. At this repo's measured 0.49-0.66
# s/call on a SINGLE stream that is 204-275 hours: 8.5 to 11.5 DAYS of
# continuous wall-clock scoring, and the two libraries cannot overlap unless
# the quota turns out to permit concurrency (it has never been tested here;
# every previous run was deliberately serialized with %1).
#
# For comparison, v3's 250,000 calls took 35-46 h. This is six times that.
#
# So: run the pilot first, look at it, and only then commit the array.
#
# ---------------------------------------------------------------------------
# NO --array DIRECTIVE HERE, ON PURPOSE
# ---------------------------------------------------------------------------
# The array spec is given at SUBMIT time, because the pilot and the two
# libraries all want different ones.
#
#   0. probe the readout (3 calls, interactive, do this FIRST):
#        python upstream_exon/probe_splice_site_usage.py
#
#   1. pilot, upstream (2,000 calls, ~25 min):
#        LIBRARY=upstream PILOT=2000 sbatch --array=0-0 --time=02:00:00 \
#            upstream_exon/slurm/score_fullctx.sbatch
#
#   2. full upstream (1,000,000 calls, 40 shards of 25,000):
#        LIBRARY=upstream sbatch --array=0-39%1 \
#            upstream_exon/slurm/score_fullctx.sbatch
#
#   3. full middle (500,000 calls, 20 shards of 25,000):
#        LIBRARY=middle sbatch --array=0-19%1 \
#            upstream_exon/slurm/score_fullctx.sbatch
#
# %1 runs the shards STRICTLY ONE AT A TIME. Only one AlphaGenome stream may
# be open at once, and score_fullctx.py additionally takes an exclusive lock
# on the results CSV and refuses to start if another scorer holds it. Running
# steps 2 and 3 at the same time is therefore NOT safe by default -- they are
# different CSVs, so the lock will not stop them, but they would share one
# quota. Only overlap them if a concurrency test says the quota allows it.
#
# 25,000 calls x 0.66 s/call = 4.6 h, inside the 6 h walltime with margin. A
# shard that hits the wall is not a loss: the next one resumes where it
# stopped, losing at most --flush-every (200) calls.
#
# ---------------------------------------------------------------------------
# PARTITION
# ---------------------------------------------------------------------------
# No --partition and no --gres: the default partition reaches the API. This
# is evidenced -- score_b1_v3.sbatch made 250,000 calls this way from this
# same account. The job is pure network I/O: one gRPC stream, negligible CPU,
# no GPU.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
LIBRARY=${LIBRARY:?set LIBRARY=upstream or LIBRARY=middle}
CHUNK=${CHUNK:-25000}
PILOT=${PILOT:-0}
FLUSH=${FLUSH:-200}
# api   = the DeepMind gRPC client: quota-limited, one stream, hence %1.
# local = weights on the cluster: no quota and no single-stream rule, so the
#         shard sizes and the %1 serialization below are far more
#         conservative than they need to be.
BACKEND=${BACKEND:-api}

cd "$REPO"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$BACKEND" == "api" && ! -f .env.alphagenome ]]; then
    echo "FATAL: .env.alphagenome not found at $REPO" >&2
    echo "       Copy it to the REPO ROOT (it is gitignored)," >&2
    echo "       or run with BACKEND=local if the weights are on disk." >&2
    exit 1
fi

if [[ ! -f ExonSkip7_Lib2_text.txt ]]; then
    echo "FATAL: ExonSkip7_Lib2_text.txt not found at $REPO" >&2
    echo "       The full-context experiments cannot run without it." >&2
    exit 1
fi

case "$LIBRARY" in
  upstream) DSDIR=outputs/upstream_exon/fullctx_upstream_1m ; N_TOTAL=1000000 ;;
  middle)   DSDIR=outputs/upstream_exon/fullctx_middle_500k ; N_TOTAL=500000  ;;
  *) echo "FATAL: LIBRARY must be upstream or middle, got '$LIBRARY'" >&2; exit 1 ;;
esac

if [[ ! -f "$DSDIR/manifest.csv" ]]; then
    echo "FATAL: missing $DSDIR/manifest.csv" >&2
    echo "       Build it first (not committed -- reproducible from the seed):" >&2
    echo "         $PYTHON_BIN upstream_exon/build_fullctx_library.py --library $LIBRARY" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-default}  library=${LIBRARY}  backend=${BACKEND}"

PILOT_ARG=()
if [[ "$PILOT" != "0" ]]; then
    PILOT_ARG=(--pilot "$PILOT")
    CHUNK="$PILOT"
    echo "PILOT=${PILOT} -> scoring only the first ${PILOT} unlabeled TRAIN rows"
else
    echo "full run -> up to ${CHUNK} new calls in this shard"
fi

# Fail fast if the backend is unreachable rather than burning the walltime.
if [[ "$BACKEND" == "api" ]]; then
echo
echo "--- connectivity preflight ---"
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: AlphaGenome unreachable from this node." >&2; exit 1; }
import sys
sys.path.insert(0, "analysis_script")
import alphagenome_local_prediction_pipeline_minigene as ag
ag.create_model(env_path=".env.alphagenome")
print("  AlphaGenome client created OK")
PY
fi

# The geometry is derived from the context file and asserted against recorded
# constants; if the file ever changes, stop here, not 20,000 calls in.
echo
echo "--- geometry preflight ---"
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: full-context geometry check failed." >&2; exit 1; }
import sys
sys.path.insert(0, "upstream_exon")
import fullctx_lib as F
lp, acc, don = F.padded_positions()
print(f"  context {F.CONTEXT_LEN} nt -> {F.TARGET_LEN} (left_pad {lp})")
print(f"  acceptor {F.GEOMETRY['acceptor']} -> padded {acc}")
print(f"  donor    {F.GEOMETRY['donor']} -> padded {don}")
PY

echo
echo "--- planned call count ---"
"$PYTHON_BIN" upstream_exon/score_fullctx.py \
    --library "$LIBRARY" "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --backend "$BACKEND" --max-new-calls "$CHUNK" --dry-run

echo
echo "--- scoring ---"
srun "$PYTHON_BIN" upstream_exon/score_fullctx.py \
    --library "$LIBRARY" \
    "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --backend "$BACKEND" \
    --max-new-calls "$CHUNK" \
    --flush-every "$FLUSH" \
    "$@"

RES="$DSDIR/ag_results.csv"
if [[ -f "$RES" ]]; then
    echo
    echo "rows now: $(( $(wc -l < "$RES") - 1 )) / ${N_TOTAL}"
fi
