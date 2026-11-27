#!/bin/bash
#SBATCH --job-name=infer_fnbm_v4
#SBATCH --account=torch_pr_352_courant
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
# ^ GENERIC, not gpu:l40s:1 -- a typed gres is rejected outright on any
#   partition without that GPU model.
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# ^ THE POINT OF THIS SCRIPT. infer.py one-hots the whole split in one go
#   (160,000 x 113 x 4 for upstream) and was OOM-killed on the login node,
#   which caps at a few GB. The model itself is 258 KB.
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/infer_fnbm_v4_%x_%A_%a.out
#SBATCH --error=logs/infer_fnbm_v4_%x_%A_%a.err
#SBATCH --array=0-1%1
#
# Per-sequence test-split predictions from the two v4 best.pt checkpoints,
# for the predicted-vs-actual figure (F1). Neither train.py nor
# score_fnbm_run.py ever wrote these -- they only produce test_metrics.csv.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/infer_fnbm_v4.sbatch
#
# Task 0 = upstream (fnbm_o1_b1_v4_c5f128), task 1 = middle (fnbm_o1_syn70_v4).
# Resumable: a task whose test_pred.csv already exists exits in seconds.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
FNBM_REPO=${FNBM_REPO:-/scratch/yl12316/factorized-model-iterations}
OUT_DIR=${OUT_DIR:-/scratch/yl12316/results/upstream-exon-fnbm}
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/fnbm/bin/python}
BATCH=${BATCH:-1024}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) RUN_NAME=fnbm_o1_b1_v4_c5f128
     DATA_DIR=/scratch/yl12316/data/fnbm_upstream_1m
     WANT_N=160000 ;;
  1) RUN_NAME=fnbm_o1_syn70_v4
     DATA_DIR=/scratch/yl12316/data/fnbm_middle_500k
     WANT_N=80000 ;;
  *) echo "FATAL: array task must be 0 or 1" >&2; exit 1 ;;
esac

RUN_DIR="$OUT_DIR/$RUN_NAME"
DEST="$REPO/outputs/upstream_exon/models/$RUN_NAME"
RAW="$RUN_DIR/test_pred.csv"

cd "$REPO"
mkdir -p logs "$DEST"

for p in "$FNBM_REPO/infer.py" "$RUN_DIR/best.pt" "$DATA_DIR/test.csv" \
         "$DATA_DIR/export_meta.json"; do
    [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-0}  run=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ -f "$DEST/test_pred.csv" ]]; then
    echo "already done -> $DEST/test_pred.csv"
    exit 0
fi

# --- inference -------------------------------------------------------------
if [[ -f "$RAW" ]]; then
    echo "--- raw predictions already present, skipping infer.py ---"
else
    echo "--- infer.py: $RUN_DIR on $DATA_DIR/test.csv ---"
    cd "$FNBM_REPO"
    srun "$PYTHON_BIN" infer.py "$RUN_DIR" \
        --csv "$DATA_DIR/test.csv" --batch_size "$BATCH" --output "$RAW"
    cd "$REPO"
fi

# --- trim + verify ---------------------------------------------------------
# Keep only the two numeric columns: the sequence column would make this ~18 MB
# of git for upstream, and the figure never needs it. The run's own test_loss
# must come back out of these columns, or this is not the checkpoint, the split
# or the unit convention that produced the reported metric.
WANT_N="$WANT_N" "$PYTHON_BIN" - "$RAW" "$DEST/test_pred.csv" \
        "$RUN_DIR/test_metrics.csv" <<'PY'
import os, sys, csv
import numpy as np, pandas as pd

raw, dest, tmcsv = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(raw)
for c in ("logit_average", "pred_0"):
    if c not in d.columns:
        sys.exit(f"FATAL: column {c!r} missing; got {list(d.columns)}")

y = d["logit_average"].to_numpy(dtype=float)
p = d["pred_0"].to_numpy(dtype=float)
mse = float(((p - y) ** 2).mean())
r2 = 1.0 - mse / y.var()
r = float(np.corrcoef(y, p)[0, 1])
print(f"n={len(d):,}  standardized MSE {mse:.6f}  R2 {r2:+.4f}  r {r:+.4f}")

want = int(os.environ["WANT_N"])
if len(d) != want:
    print(f"WARNING: {len(d)} rows, expected {want}", file=sys.stderr)

with open(tmcsv, newline="") as f:
    tm = next(csv.DictReader(f))
reported = float(tm["test_loss"])
diff = abs(mse - reported)
print(f"run reported test_loss {reported:.6f}  (diff {diff:.2e})")
if diff > 2e-3:
    sys.exit(f"FATAL: recomputed MSE {mse:.6f} != reported {reported:.6f}. "
             f"Refusing to write a predictions file that disagrees with the "
             f"metric the report quotes.")

d[["logit_average", "pred_0"]].to_csv(dest, index=False)
print(f"wrote {dest}")
PY

cp "$DATA_DIR/export_meta.json" "$DEST/export_meta.json"
echo "done -> $DEST  (test_pred.csv + export_meta.json)"
echo
echo "now commit, from $REPO:"
echo "    git add outputs/upstream_exon/models/$RUN_NAME"
