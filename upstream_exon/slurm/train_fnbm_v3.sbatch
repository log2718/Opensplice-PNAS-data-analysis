#!/bin/bash
#SBATCH --job-name=b1v3_fnbm
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1v3_fnbm_%j.out
#SBATCH --error=logs/b1v3_fnbm_%j.err
#
# Arush's first-order PositionalFactorizedNBM on the v3 upstream-exon library.
#
# This runs HIS train.py from HIS checkout, with our config and our data. The
# FNBM repo is not vendored into this one: his loop, Muon optimizer and
# interpretation hooks stay his, and only the config and the CSVs are ours.
#
# PURPOSE: the interpretable CNN reaches test R2 0.6806 and the 6-mer ridge
# 0.6716, while an unconstrained BiLSTM reaches 0.8199 -- so roughly a third
# of the learnable signal is invisible to a 20-filter additive hexamer model.
# The question here is whether a first-order factorized model with learned
# positional curves recovers that gap while staying readable.
#
# RESUMING: just resubmit. His loop restarts from the newest checkpoint_*.pt,
# and a run that already wrote test_metrics.csv is skipped entirely. The 4 h
# limit is deliberate -- short jobs are scheduled sooner.
#
# MEMORY: SequenceDataset one-hots every row up front, so the 200k training
# split holds 200000*4*113*4 B = 362 MB resident, plus 18 MB val and 72 MB
# test. 48G is generous; 16G would likely do.
#
# PREREQUISITES (both one-off):
#   1. Arush's repo checked out at $FNBM_REPO.
#   2. The CSVs written by upstream_exon/export_fnbm_csv.py at $DATA_DIR.
#      Use --center: see the note in that script.
#
# Usage:
#   sbatch upstream_exon/slurm/train_fnbm_v3.sbatch
#   RUN_NAME=fnbm_o1_b1_v3_seed1 SEED=1 sbatch upstream_exon/slurm/train_fnbm_v3.sbatch

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
FNBM_REPO=${FNBM_REPO:-/scratch/yl12316/factorized-model-iterations}
CONFIG=${CONFIG:-$REPO/upstream_exon/fnbm/first_order_b1_v3.yaml}
DATA_DIR=${DATA_DIR:-/scratch/yl12316/data/b1_v3_fnbm}
OUT_DIR=${OUT_DIR:-/scratch/yl12316/results/upstream-exon-fnbm}
RUN_NAME=${RUN_NAME:-fnbm_o1_b1_v3}
SEED=${SEED:-0}
# Smoke-test knobs. EPOCHS caps num_epochs; SUBSAMPLE thins the training split
# through his own data.train_subsample_n, so a 2-epoch run on 5,000 rows
# exercises the whole path (loaders, Muon, checkpointing, interpretation, the
# test pass) in minutes. Leave both empty for a real run.
EPOCHS=${EPOCHS:-}
SUBSAMPLE=${SUBSAMPLE:-}
# A venv layered over the cluster's CUDA torch: it adds pandas, scikit-learn,
# scipy, matplotlib, seaborn and logomaker without writing into the base env,
# and reaches torch through a .pth pointing at that env's site-packages.
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/envs/fnbm/bin/python}

cd "$REPO"
mkdir -p logs

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-default}"
echo "run=$RUN_NAME  seed=$SEED  config=$CONFIG"

for p in "$FNBM_REPO/train.py" "$CONFIG" "$DATA_DIR/train.csv" \
         "$DATA_DIR/valid.csv" "$DATA_DIR/test.csv"; do
    [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 1; }
done

# Preflight. Both halves fail in seconds rather than after a 4 h run:
#
#   * no CUDA -> the job would silently run at CPU speed.
#   * missing interpretation deps -> hooks.py catches every hook exception
#     ("training never crashes"), so the run would COMPLETE with a valid test
#     R2 and no logos, no dedup and no positional plots. seaborn and logomaker
#     are imported at module scope by utils/interpretation/{dedup,interactions,
#     validation}.py and are NOT in his pyproject.toml, so this is the likely
#     failure, not a hypothetical one.
"$PYTHON_BIN" - "$FNBM_REPO" <<'PY'
import importlib, sys
fnbm = sys.argv[1]
sys.path.insert(0, fnbm)

import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA device visible to this interpreter. Point "
             "PYTHON_BIN at a GPU-enabled env, or run train.py inside the "
             "singularity overlay Arush's jobs/alphagenome_splicing/train.sbatch uses.")
print(torch.cuda.get_device_name(0))

missing = [m for m in ("numpy", "pandas", "yaml", "sklearn", "tqdm", "pyarrow")
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"FATAL: training needs {missing}")

interp = [m for m in ("matplotlib", "seaborn", "logomaker", "scipy", "plotly")
          if not importlib.util.find_spec(m)]
if interp:
    print(f"WARNING: interpretation will be SKIPPED silently -- missing {interp}. "
          f"Install them into the venv PYTHON_BIN points at: "
          f"$PYTHON_BIN -m pip install {' '.join(interp)}", file=sys.stderr)
else:
    print("interpretation deps ok")
PY

# Record which target convention this run used, so the log says whether the
# targets were centered without anyone having to re-read the CSVs.
"$PYTHON_BIN" - "$DATA_DIR/export_meta.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
s = m["splits"]
print(f"data: input_length={m['input_length']}  target={m['target_column']}  "
      f"center_shift={m['center_shift']:.6f}")
print("      " + "  ".join(f"{k} n={v['n']:,}" for k, v in s.items()))
print(f"      test var_pop={s['test']['var_pop']:.6f}  "
      f"-> R2 = 1 - test_loss/{s['test']['var_pop']:.6f}")
if (s["train"]["n"], s["valid"]["n"], s["test"]["n"]) != (200000, 10000, 40000):
    print("WARNING: split sizes are not 200000/10000/40000", file=sys.stderr)
PY

# Write this run's config: our YAML with the run name, seed and paths applied,
# so one checked-in config can drive several seeds without being edited.
CELL=${SLURM_TMPDIR:-/tmp}/fnbm_cell_${SLURM_JOB_ID:-0}.yaml
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
    # Otherwise a 2-epoch smoke run writes no checkpoint and no final report.
    c["run"]["checkpoint_every"] = 1
    c["interpret"]["every_k_epochs"] = max(1, int(epochs) // 2)
if subsample:
    c["data"]["train_subsample_n"] = int(subsample)
    c["data"]["train_subsample_seed"] = 0
yaml.safe_dump(c, open(dest, "w"), sort_keys=False)
print(f"wrote {dest}  (num_epochs={c['run']['num_epochs']}, "
      f"train_subsample_n={c['data'].get('train_subsample_n')})")
PY

RUN_DIR="$OUT_DIR/$RUN_NAME"
if [[ -f "$RUN_DIR/test_metrics.csv" ]]; then
    echo "already finished -> $RUN_DIR/test_metrics.csv"; exit 0
fi
[[ -n "$(ls "$RUN_DIR"/checkpoint_*.pt 2>/dev/null || true)" ]] \
    && echo "resuming from the newest checkpoint in $RUN_DIR"

export PYTHONUNBUFFERED=1
cd "$FNBM_REPO"
srun "$PYTHON_BIN" train.py "$CELL"

echo "done -> $RUN_DIR"
"$PYTHON_BIN" "$REPO/upstream_exon/score_fnbm_run.py" \
    --run-dir "$RUN_DIR" --export-meta "$DATA_DIR/export_meta.json" || true
