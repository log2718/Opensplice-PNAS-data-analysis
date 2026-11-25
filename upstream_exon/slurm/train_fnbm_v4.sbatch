#!/bin/bash
#SBATCH --job-name=fnbm_v4
#SBATCH --account=torch_pr_36_mren
# ^ ONE-OFF, 2026-11-25. Every other script in this directory stays on
#   torch_pr_352_courant; only this one was moved, to get the v4 chain
#   scheduled. Do not treat mren as the new default.
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/fnbm_v4_%x_%A_%a.out
#SBATCH --error=logs/fnbm_v4_%x_%A_%a.err
#
# Arush's first-order PositionalFactorizedNBM on a v4 FULL-CONTEXT library.
#
# Runs HIS train.py from HIS checkout with our config and our data. The FNBM
# repo is not vendored and is never written to.
#
#   LIBRARY=upstream sbatch --array=0-5%1 upstream_exon/slurm/train_fnbm_v4.sbatch
#   LIBRARY=middle   sbatch --array=0-3%1 upstream_exon/slurm/train_fnbm_v4.sbatch
#
# ---------------------------------------------------------------------------
# WHY AN ARRAY
# ---------------------------------------------------------------------------
# 800,000 training rows is 4x the v3 library, so 100 epochs will not fit in one
# 4 h task. Each task resumes from the newest checkpoint_*.pt, and a task that
# starts after the run already wrote test_metrics.csv exits in seconds. %1
# keeps one task on a GPU at a time, which is also what keeps this run under
# QOSMaxGRESPerUser -- the cap that cancelled 23546677_0 mid-run.
#
# checkpoint_every is 2 in the v4 configs (10 in v3): a 4 h task that dies
# must not throw away more than ~2 epochs.
#
# ---------------------------------------------------------------------------
# THE EXPORT RUNS HERE
# ---------------------------------------------------------------------------
# export_fnbm_csv.py writes train/valid/test.csv + export_meta.json from
# ag_results.csv. It is skipped when export_meta.json already exists, and
# that file is written LAST, so a task killed mid-export simply re-exports.
# Keeping it in-job means the whole pipeline is three submissions, not five.
#
# --center --standardize is NOT optional. Every L2 penalty in the config is an
# absolute weight penalty tuned at var(target) ~ 0.414 (v3). The v4 targets
# have var 0.0021 (upstream) and 8.5 (middle) -- 195x smaller and 20x larger.
# Raw, the upstream run would be regularized into predicting a constant and
# would look like "no learnable signal". R2 is scale-invariant and var_pop is
# recorded after scaling, so score_fnbm_run.py still reports the right R2.

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
FNBM_REPO=${FNBM_REPO:-/scratch/yl12316/factorized-model-iterations}
OUT_DIR=${OUT_DIR:-/scratch/yl12316/results/upstream-exon-fnbm}
LIBRARY=${LIBRARY:?set LIBRARY=upstream or LIBRARY=middle}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-}
SUBSAMPLE=${SUBSAMPLE:-}
NO_TRITON=${NO_TRITON:-1}
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/fnbm/bin/python}

case "$LIBRARY" in
  upstream)
    SRC=outputs/upstream_exon/fullctx_upstream_1m/ag_results.csv
    SEQ_COL=b1
    DATA_DIR=${DATA_DIR:-/scratch/yl12316/data/fnbm_upstream_1m}
    CONFIG=${CONFIG:-$REPO/upstream_exon/fnbm/first_order_b1_v4.yaml}
    RUN_NAME=${RUN_NAME:-fnbm_o1_b1_v4}
    WANT_N="800000 40000 160000" ;;
  middle)
    SRC=outputs/upstream_exon/fullctx_middle_500k/ag_results.csv
    SEQ_COL=syn70
    DATA_DIR=${DATA_DIR:-/scratch/yl12316/data/fnbm_middle_500k}
    CONFIG=${CONFIG:-$REPO/upstream_exon/fnbm/first_order_syn70_v4.yaml}
    RUN_NAME=${RUN_NAME:-fnbm_o1_syn70_v4}
    WANT_N="400000 20000 80000" ;;
  *) echo "FATAL: LIBRARY must be upstream or middle, got '$LIBRARY'" >&2; exit 1 ;;
esac

cd "$REPO"
mkdir -p logs

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}"
echo "library=$LIBRARY  run=$RUN_NAME  seed=$SEED"
echo "config=$CONFIG"
echo "data=$DATA_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for p in "$FNBM_REPO/train.py" "$CONFIG" "$SRC"; do
    [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done

# --- finished already? -----------------------------------------------------
RUN_DIR="$OUT_DIR/$RUN_NAME"
if [[ -f "$RUN_DIR/test_metrics.csv" ]]; then
    echo "already finished -> $RUN_DIR/test_metrics.csv"
    "$PYTHON_BIN" "$REPO/upstream_exon/score_fnbm_run.py" \
        --run-dir "$RUN_DIR" --export-meta "$DATA_DIR/export_meta.json" || true
    exit 0
fi

# --- export (idempotent; export_meta.json is the completion marker) --------
if [[ -f "$DATA_DIR/export_meta.json" ]]; then
    echo "--- export: already present at $DATA_DIR, skipping ---"
else
    echo "--- export: $SRC -> $DATA_DIR ---"
    "$PYTHON_BIN" upstream_exon/export_fnbm_csv.py \
        --from-share "$SRC" --out-dir "$DATA_DIR" \
        --sequence-column "$SEQ_COL" --target-column logit_average \
        --center --standardize
fi

for p in "$DATA_DIR/train.csv" "$DATA_DIR/valid.csv" "$DATA_DIR/test.csv" \
         "$DATA_DIR/export_meta.json"; do
    [[ -e "$p" ]] || { echo "FATAL: export did not produce $p" >&2; exit 1; }
done

# --- preflight: fail in seconds, not after 4 h -----------------------------
"$PYTHON_BIN" - "$FNBM_REPO" <<'PY'
import importlib, sys
sys.path.insert(0, sys.argv[1])
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible to this interpreter.")
print(torch.cuda.get_device_name(0))
try:
    from triton.runtime import driver
    _ = driver.active.utils
    print("triton: usable (NO_TRITON=0 would run his kernels)")
except Exception as exc:
    print(f"triton: unusable ({type(exc).__name__}) -- expected on bare metal; "
          f"NO_TRITON=1 uses the PyTorch reference")
missing = [m for m in ("numpy", "pandas", "yaml", "sklearn", "tqdm")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"FATAL: training needs {missing}")
# hooks.py swallows every hook exception, so a missing plotting dep produces a
# COMPLETED run with a valid R2 and no logos, no dedup, no positional plots.
interp = [m for m in ("matplotlib", "seaborn", "logomaker", "scipy")
          if not importlib.util.find_spec(m)]
if interp:
    print(f"WARNING: interpretation will be SKIPPED SILENTLY -- missing {interp}. "
          f"Install into the venv PYTHON_BIN points at.", file=sys.stderr)
else:
    print("interpretation deps ok")
PY

# --- what the targets look like, recorded in the log -----------------------
WANT_N="$WANT_N" "$PYTHON_BIN" - "$DATA_DIR/export_meta.json" <<'PY'
import json, os, sys
m = json.load(open(sys.argv[1]))
s = m["splits"]
print(f"data: input_length={m['input_length']}  seq={m['sequence_column']}  "
      f"target={m['target_column']}")
print(f"      center_shift={m['center_shift']:.6f}  "
      f"target_scale={m.get('target_scale', 1.0):.6f}  "
      f"(prediction * target_scale + center_shift = logit units)")
print("      " + "  ".join(f"{k} n={v['n']:,}" for k, v in s.items()))
print(f"      test var_pop={s['test']['var_pop']:.6f}  "
      f"-> R2 = 1 - test_loss/{s['test']['var_pop']:.6f}")
want = [int(x) for x in os.environ["WANT_N"].split()]
got = [s["train"]["n"], s["valid"]["n"], s["test"]["n"]]
if got != want:
    print(f"WARNING: split sizes {got} != expected {want}", file=sys.stderr)
if abs(s["train"]["var_pop"] - 1.0) > 1e-6:
    sys.exit(f"FATAL: train var_pop is {s['train']['var_pop']}, not 1.0 -- "
             f"the export ran without --standardize. Delete {sys.argv[1]} "
             f"and its CSVs, then resubmit.")
PY

# --- this run's config: our YAML with name/seed/paths applied --------------
CELL=${SLURM_TMPDIR:-/tmp}/fnbm_cell_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}.yaml
"$PYTHON_BIN" - "$CONFIG" "$CELL" "$RUN_NAME" "$SEED" "$DATA_DIR" "$OUT_DIR" \
               "$FNBM_REPO" "$EPOCHS" "$SUBSAMPLE" <<'PY'
import sys, yaml
_, src, dest, run_name, seed, data_dir, out_dir, fnbm, epochs, subsample = sys.argv
c = yaml.safe_load(open(src))
c["run"]["name"], c["run"]["seed"] = run_name, int(seed)
c["run"]["output_dir"] = out_dir
for split, key in (("train", "train_csv"), ("valid", "valid_csv"), ("test", "test_csv")):
    c["data"][key] = f"{data_dir}/{split}.csv"
c["model"]["file"] = f"{fnbm}/models/fnbm/positional_model.py"
for h in c.get("interpret", {}).get("hooks", []):
    h["file"] = f"{fnbm}/hooks/fnbm_standard_interpret.py"
if epochs:
    c["run"]["num_epochs"] = int(epochs)
    c["run"]["checkpoint_every"] = 1
    c["interpret"]["every_k_epochs"] = max(1, int(epochs) // 2)
if subsample:
    c["data"]["train_subsample_n"] = int(subsample)
    c["data"]["train_subsample_seed"] = 0
yaml.safe_dump(c, open(dest, "w"), sort_keys=False)
print(f"wrote {dest}  (num_epochs={c['run']['num_epochs']}, "
      f"checkpoint_every={c['run']['checkpoint_every']}, "
      f"train_subsample_n={c['data'].get('train_subsample_n')})")
PY

[[ -n "$(ls "$RUN_DIR"/checkpoint_*.pt 2>/dev/null || true)" ]] \
    && echo "resuming from the newest checkpoint in $RUN_DIR"

export PYTHONUNBUFFERED=1
cd "$FNBM_REPO"
if [[ "$NO_TRITON" == "1" ]]; then
    srun "$PYTHON_BIN" "$REPO/upstream_exon/fnbm/train_reference_kernels.py" \
        "$FNBM_REPO" "$CELL"
else
    srun "$PYTHON_BIN" train.py "$CELL"
fi

echo "done -> $RUN_DIR"
"$PYTHON_BIN" "$REPO/upstream_exon/score_fnbm_run.py" \
    --run-dir "$RUN_DIR" --export-meta "$DATA_DIR/export_meta.json" || true
