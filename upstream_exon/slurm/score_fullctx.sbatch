#!/bin/bash
#SBATCH --job-name=fullctx_score
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/fullctx_%x_%A_%a.out
#SBATCH --error=logs/fullctx_%x_%A_%a.err
#
# AlphaGenome SPLICE_SITE_USAGE labeling of a full-context library.
# Target: logit_average = logit((usage_acceptor + usage_donor) / 2),
# read from the CL:0002518 PLUS-strand track (index 71 of 734).
#
# ---------------------------------------------------------------------------
# ONE SUBMISSION, SELF-VALIDATING, SELF-CHAINING
# ---------------------------------------------------------------------------
# Default backend is LOCAL: alphagenome_pytorch 0.3.1 against
# model_all_folds.safetensors on an L40S. No quota, no single stream, so the
# API-era design (25,000-row shards serialized with %1) is gone.
#
# The run measures itself. The only local throughput number we have is 20.5
# seq/s on a B200 at 131,072 nt -- a different GPU and an 8x longer input --
# so there is no honest walltime estimate. Instead:
#
#   * the loop prints live seq/s and an ETA against the whole library
#   * every shard is RESUMABLE on row_id, so hitting the wall costs nothing
#   * --array=0-N%1 chains tasks that each resume where the last stopped
#
# So you submit once and it runs to completion, however long that is. A task
# that starts with nothing left to do exits in seconds.
#
#   1. upstream, 1,000,000 rows -- chain of 8 x 12 h, one at a time:
#        LIBRARY=upstream sbatch --array=0-7%1 \
#            upstream_exon/slurm/score_fullctx.sbatch
#
#   2. middle, 500,000 rows (safe to run CONCURRENTLY with 1 if a second GPU
#      is free -- different CSVs, different lock, and no shared quota now):
#        LIBRARY=middle sbatch --array=0-3%1 \
#            upstream_exon/slurm/score_fullctx.sbatch
#
#   3. to sanity-check the plumbing without committing, add PILOT:
#        LIBRARY=upstream PILOT=2000 sbatch --array=0-0 --time=01:00:00 \
#            upstream_exon/slurm/score_fullctx.sbatch
#      This is optional. The preflight below already gates correctness, so a
#      full submission is not a leap of faith.
#
# %1 keeps one task per library running at a time: score_fullctx.py takes an
# exclusive lock on the results CSV and a second task would just block.
#
# ---------------------------------------------------------------------------
# WHAT THE PREFLIGHT GUARANTEES
# ---------------------------------------------------------------------------
# Before a single library row is scored, in-process:
#   * the track catalogue's sha256 matches the recorded one, and CL:0002518
#     (+) still resolves to index 71 -- otherwise every label would silently
#     describe the wrong cell type
#   * the native construct's usage is in [0, 1] (predictions, not logits)
#   * the PLUS track beats its MINUS twin (index 438) at BOTH the acceptor
#     and the donor -- if not, the strand is inverted
#   * both annotated positions rank in the top 200 of 16,384 -- if not, the
#     geometry or the index is wrong
# Any failure aborts before writing. Override only with --no-strict-preflight.
#
# ---------------------------------------------------------------------------
# API FALLBACK
# ---------------------------------------------------------------------------
# BACKEND=api still works and still writes the same columns, so a run can be
# finished on either backend. It needs .env.alphagenome at the repo root, the
# opensplicerna interpreter, and no GPU -- and it is 8.5-11.5 DAYS for 1.5M
# calls at this repo's measured 0.49-0.66 s/call. Use it only for the
# local-vs-API concordance check.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
LIBRARY=${LIBRARY:?set LIBRARY=upstream or LIBRARY=middle}
# 0 = the whole remaining library; the walltime, not a row cap, ends a task.
CHUNK=${CHUNK:-0}
PILOT=${PILOT:-0}
FLUSH=${FLUSH:-2000}
BACKEND=${BACKEND:-local}
BATCH=${BATCH:-32}
WEIGHTS=${WEIGHTS:-/scratch/yl12316/models/alphagenome/model_all_folds.safetensors}
CATALOG=${CATALOG:-/scratch/yl12316/models/alphagenome/track_metadata_human.parquet}

cd "$REPO"
mkdir -p logs

if [[ "$BACKEND" == "local" ]]; then
    PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/alphagenome/bin/python}
else
    PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ "$BACKEND" == "local" ]]; then
    if [[ ! -f "$WEIGHTS" ]]; then
        echo "FATAL: weights not found: $WEIGHTS" >&2
        exit 1
    fi
    if [[ ! -f "$CATALOG" ]]; then
        echo "FATAL: track catalogue not found: $CATALOG" >&2
        echo "       The 0.3.1 wheel omits it; fetch it with:" >&2
        echo "         curl -fL --create-dirs -o $CATALOG \\" >&2
        echo "           https://raw.githubusercontent.com/genomicsxai/alphagenome-pytorch/v0.3.1/src/alphagenome_pytorch/data/track_metadata_human.parquet" >&2
        exit 1
    fi
elif [[ ! -f .env.alphagenome ]]; then
    echo "FATAL: .env.alphagenome not found at $REPO (needed by BACKEND=api)" >&2
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
if [[ "$BACKEND" == "local" ]]; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

PILOT_ARG=()
if [[ "$PILOT" != "0" ]]; then
    PILOT_ARG=(--pilot "$PILOT")
    echo "PILOT=${PILOT} -> scoring only the first ${PILOT} unlabeled TRAIN rows"
else
    if [[ "$CHUNK" == "0" ]]; then
        echo "full run -> the whole remaining library, until walltime"
    else
        echo "full run -> up to ${CHUNK} new calls in this shard"
    fi
fi

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

# Geometry is derived from the context file and asserted against recorded
# constants; if the file ever changes, stop here, not 20,000 rows in. For the
# local backend also re-derive the track index from the catalogue, because
# every label in the run depends on it.
echo
echo "--- geometry / track preflight ---"
BACKEND="$BACKEND" CATALOG="$CATALOG" \
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: geometry/track check failed." >&2; exit 1; }
import hashlib, os, sys
sys.path.insert(0, "upstream_exon")
import fullctx_lib as F
lp, acc, don = F.padded_positions()
print(f"  context {F.CONTEXT_LEN} nt -> {F.TARGET_LEN} (left_pad {lp})")
print(f"  acceptor {F.GEOMETRY['acceptor']} -> padded {acc}")
print(f"  donor    {F.GEOMETRY['donor']} -> padded {don}")
if os.environ.get("BACKEND") == "local":
    cat = os.environ["CATALOG"]
    got = hashlib.sha256(open(cat, "rb").read()).hexdigest()
    if got != F.CATALOG_SHA256:
        raise SystemExit(
            f"  catalogue sha256 mismatch\n    expected {F.CATALOG_SHA256}\n"
            f"    got      {got}")
    idx = F.resolve_usage_track(cat, strand="+")
    print("  catalogue sha256 OK")
    print(f"  {F.ONTOLOGY[0]} (+) -> track {idx} of {F.N_USAGE_TRACKS}")
PY

echo
echo "--- planned row count ---"
"$PYTHON_BIN" upstream_exon/score_fullctx.py \
    --library "$LIBRARY" "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --backend "$BACKEND" --max-new-calls "$CHUNK" --dry-run

EXTRA=()
if [[ "$BACKEND" == "local" ]]; then
    EXTRA=(--weights "$WEIGHTS" --catalog "$CATALOG" --batch-size "$BATCH")
fi

echo
echo "--- scoring ---"
srun "$PYTHON_BIN" upstream_exon/score_fullctx.py \
    --library "$LIBRARY" \
    "${PILOT_ARG[@]+"${PILOT_ARG[@]}"}" \
    --backend "$BACKEND" \
    --max-new-calls "$CHUNK" \
    --flush-every "$FLUSH" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    "$@"

RES="$DSDIR/ag_results.csv"
if [[ -f "$RES" ]]; then
    echo
    echo "rows now: $(( $(wc -l < "$RES") - 1 )) / ${N_TOTAL}"
fi
