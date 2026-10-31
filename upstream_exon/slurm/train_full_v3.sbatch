#!/bin/bash
#SBATCH --job-name=b1v3_train
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1v3_train_%j.out
#SBATCH --error=logs/b1v3_train_%j.err
#
# CNN training on the v3 FIXED-COMPOSITION library
# (200,000 train / 10,000 val / 40,000 test).
#
# Every sequence is a permutation of the same 29 A / 28 C / 28 G / 28 T, so
# nucleotide composition is constant and cannot explain anything. In v2 it
# explained ~70% of the response -- a gradient-boosted model on three base
# frequencies beat both the CNN and the 6-mer ridge. This run asks what is
# left once that axis is gone.
#
# Model, loss and optimization are UNCHANGED from full_v2:
#   conv(4->20,k=6) x2 + learned position bias -> Softplus -> SumDiff
#   -> Linear(1,1).  5,324 parameters, trained from scratch.
#   MSE on the train-fitted z-scored target, Adam lr 1e-3, batch 256,
#   l2-smooth 1e-3 on the position biases, early stopping patience 40.
# Only the data and the baseline set differ.
#
# ---------------------------------------------------------------------------
# BASELINES: --baseline-set fixed_composition
# ---------------------------------------------------------------------------
# The composition ridge is DEGENERATE here -- its features have zero variance
# across the library, so it can only predict the training mean. It is replaced
# by a dinucleotide ridge (16 features), which is the simplest thing that
# still varies once composition is fixed.
#
# The 6-mer ridge uses the streaming Gram solver. A 200,000 x 4,096 design is
# 3.3 GB as float32; StandardScaler copies it and RidgeCV's leave-one-out path
# copies it again, which is >10 GB before any arithmetic. The Gram solver
# never materializes the matrix: it folds chunks into X'X (4,096^2 = 134 MB)
# and solves every penalty against one eigendecomposition. Measured locally:
# ~7 s to featurize, ~30 s for the Gram pass, ~55 s for the eigendecomposition,
# peak ~1.5 GB. Its penalty is chosen on the REAL validation split, and it
# warns if the choice lands on the edge of the grid -- which is what happened
# in full_v2, where RidgeCV picked the largest alpha offered.
#
# ---------------------------------------------------------------------------
# PARTITION / RESOURCES
# ---------------------------------------------------------------------------
# No --partition and no --gres, so the job lands on the cluster default
# partition -- the same allocation full_v2 trained on successfully (CPU nodes
# cs###). torch here is a CPU-only build, and --device cpu makes that explicit
# so the run can never half-initialise a CUDA path.
#
# 8 CPUs / 32G / 12h. Memory: one-hot inputs are 200,000 x 4 x 113 float32
# = 361 MB (plus 90 MB for val+test), the CNN is negligible, and the k-mer
# baseline peaks around 1.5 GB. 32G is generous; 16G would also do.
#
# TIME: measured ~50 s/epoch for 200,000 rows at batch 256 on 8 CPU threads,
# so 12 h fits roughly 700 epochs. The cap is 400, which is 400 x 782 = 313k
# gradient steps -- three times full_v2's 104k. If early stopping has not
# triggered when the walltime ends, just resubmit: the script detects
# last.pt and resumes with the patience counter intact.
#
# PREFLIGHT: the trainer checks every import it needs and that the dataset
# holds exactly EXPECTED_ROWS labeled rows BEFORE training starts.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_full_v3.sbatch
#     FRESH=1 sbatch upstream_exon/slurm/train_full_v3.sbatch    # ignore ckpts
#     EXPECTED_ROWS=100000 sbatch upstream_exon/slurm/train_full_v3.sbatch
#                                                   # train on a partial library

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
RUN_NAME=${RUN_NAME:-full_v3}
EXPECTED_ROWS=${EXPECTED_ROWS:-250000}
EPOCHS=${EPOCHS:-400}
FRESH=${FRESH:-0}

cd "$REPO"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

DSDIR=outputs/upstream_exon/dataset_v3
RESULTS=$DSDIR/b1_primary_ag_results.csv
MANIFEST=$DSDIR/b1_primary_manifest.csv

for f in "$RESULTS" "$MANIFEST"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing $f" >&2
        echo "       Build the manifest with build_b1_dataset_v3.py and run" >&2
        echo "       the scoring job first." >&2
        exit 1
    fi
done

ROWS=$(( $(wc -l < "$RESULTS") - 1 ))
echo "labeled rows: ${ROWS} (expected ${EXPECTED_ROWS})"
if [[ "${ROWS}" -lt "${EXPECTED_ROWS}" ]]; then
    echo "FATAL: dataset is incomplete (${ROWS}/${EXPECTED_ROWS})." >&2
    echo "       Finish scoring, or set EXPECTED_ROWS to train on a subset" >&2
    echo "       -- and say so when reporting results, because a partial" >&2
    echo "       library is a different experiment." >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-default}  run=${RUN_NAME}"

# Keep every numeric backend inside the allocation. Without this each library
# spawns one thread per visible core and they fight each other.
NT=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" \
       OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"
echo "threads: $NT"

"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__,
      '| cuda available:', torch.cuda.is_available(),
      '| threads:', torch.get_num_threads())"

# --- make --requeue actually resume -------------------------------------
RESUME_ARG=()
LAST_CKPT="outputs/upstream_exon/models/${RUN_NAME}/last.pt"
if [[ -f "$LAST_CKPT" && "$FRESH" != "1" ]]; then
    echo "found $LAST_CKPT -> resuming"
    RESUME_ARG=(--resume)
elif [[ "$FRESH" == "1" ]]; then
    echo "FRESH=1 -> ignoring any existing checkpoint"
else
    echo "no checkpoint found -> starting from scratch"
fi

echo
echo "=== training (CPU) ==="
srun "$PYTHON_BIN" upstream_exon/train_upstream.py \
    --run-name "${RUN_NAME}" \
    --results-csv "$RESULTS" \
    --manifest-csv "$MANIFEST" \
    --epochs "${EPOCHS}" \
    --batch-size 256 \
    --lr 1e-3 \
    --patience 40 \
    --l2-smooth 1e-3 \
    --loss mse \
    --kmer-k 6 \
    --baseline-set fixed_composition \
    --device cpu \
    --threads "$NT" \
    --num-workers 2 \
    --seed 42 \
    --expect-rows "${EXPECTED_ROWS}" \
    "${RESUME_ARG[@]+"${RESUME_ARG[@]}"}" \
    "$@"

echo
echo "done -> outputs/upstream_exon/models/${RUN_NAME}"
ls -la "outputs/upstream_exon/models/${RUN_NAME}"
