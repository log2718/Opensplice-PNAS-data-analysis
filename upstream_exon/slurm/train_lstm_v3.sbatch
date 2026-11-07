#!/bin/bash
#SBATCH --job-name=b1v3_lstm
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/b1v3_lstm_%j.out
#SBATCH --error=logs/b1v3_lstm_%j.err
#
# Unconstrained BiLSTM control on the v3 fixed-composition library.
#
# PURPOSE: bound what is learnable from B1 sequence at all. The CNN reaches
# test R2 0.6806 and the 6-mer ridge 0.6716. If a free-form recurrent model
# with ~40x the parameters lands in the same place, that ceiling is the data
# (AlphaGenome's own response to a 113-nt window), not the CNN's inductive
# bias. If it goes well past, the CNN is leaving signal on the table.
#
# Same splits, same target standardizer, same metrics and the same early
# stopping as full_v3 -- it runs through train_upstream.py, so the numbers
# are directly comparable to baseline_comparison.csv rather than being a
# separate experiment.
#
# Position biases and filter L1 do not exist here, so --l2-smooth and
# --l1-lambda are left at 0 (the model stubs them out).
#
# TIME: an LSTM is sequential over 113 steps and cannot use the CNN's
# parallelism -- measured ~115 s/epoch for a 1-layer h=32 BiLSTM at batch
# 256 on 8 Apple-silicon threads, and this is 2 layers at h=128, so expect
# roughly 10-20 min/epoch on a cs node. 12 h therefore buys ~40-70 epochs.
# LSTMs on 200k samples typically converge in tens of epochs, so patience 12
# should stop it before the walltime; if it does not, resubmit and it
# resumes from last.pt with Adam state and the patience counter intact.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/train_lstm_v3.sbatch
#     HIDDEN=192 sbatch upstream_exon/slurm/train_lstm_v3.sbatch
#     FRESH=1 sbatch upstream_exon/slurm/train_lstm_v3.sbatch

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
RUN_NAME=${RUN_NAME:-lstm_v3}
HIDDEN=${HIDDEN:-128}
LAYERS=${LAYERS:-2}
DROPOUT=${DROPOUT:-0.2}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
FRESH=${FRESH:-0}

cd "$REPO"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
DSDIR=outputs/upstream_exon/dataset_v3
RESULTS=$DSDIR/b1_primary_ag_results.csv
MANIFEST=$DSDIR/b1_primary_manifest.csv

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-default}"
echo "run=${RUN_NAME}  hidden=${HIDDEN}  layers=${LAYERS}  batch=${BATCH}"

NT=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" \
       OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"

RESUME_ARG=()
if [[ -f "outputs/upstream_exon/models/${RUN_NAME}/last.pt" && "$FRESH" != "1" ]]; then
    echo "resuming from last.pt"
    RESUME_ARG=(--resume)
fi

srun "$PYTHON_BIN" upstream_exon/train_upstream.py \
    --arch bilstm \
    --lstm-hidden "${HIDDEN}" \
    --lstm-layers "${LAYERS}" \
    --lstm-dropout "${DROPOUT}" \
    --run-name "${RUN_NAME}" \
    --results-csv "$RESULTS" \
    --manifest-csv "$MANIFEST" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH}" \
    --lr 1e-3 \
    --lr-plateau 4 \
    --patience 12 \
    --loss mse \
    --kmer-k 6 \
    --baseline-set fixed_composition \
    --device cpu \
    --threads "$NT" \
    --num-workers 2 \
    --seed 42 \
    --expect-rows 250000 \
    "${RESUME_ARG[@]+"${RESUME_ARG[@]}"}" \
    "$@"

echo
echo "done -> outputs/upstream_exon/models/${RUN_NAME}"
ls -la "outputs/upstream_exon/models/${RUN_NAME}"
