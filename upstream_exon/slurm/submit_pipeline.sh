#!/bin/bash
#
# Submit the B1 pipeline on Torch as an afterok dependency chain:
#
#   connectivity check -> scoring chunk 1 -> ... -> chunk N -> GPU training
#
# Scoring is split into chunks because the job holds a GPU allocation it does
# not use (no CPU-only partition exists on this cluster), and because a single
# job may not fit inside the partition walltime. Chunks are chained with
# afterok, so they run STRICTLY ONE AT A TIME -- never concurrently against
# the same results CSV. The scorer also takes an exclusive lock as a second
# guard.
#
# Every stage is submitted only if the previous submission succeeded; a failed
# sbatch aborts immediately and schedules nothing downstream.
#
# Usage, from the repository root on Torch:
#
#     bash upstream_exon/slurm/submit_pipeline.sh              # dry run: prints the plan
#     bash upstream_exon/slurm/submit_pipeline.sh --go         # actually submit
#     bash upstream_exon/slurm/submit_pipeline.sh --go --score-only
#     bash upstream_exon/slurm/submit_pipeline.sh --go --train-only
#     bash upstream_exon/slurm/submit_pipeline.sh --go --no-check
#
# Resource overrides (passed through to sbatch, beating the #SBATCH lines):
#
#     PARTITION=<name> GRES=NONE bash upstream_exon/slurm/submit_pipeline.sh --go
#     CHUNK=5000 bash upstream_exon/slurm/submit_pipeline.sh --go

set -euo pipefail

cd "$(dirname "$0")/../.."
SLURM_DIR=upstream_exon/slurm
DATASET=${DATASET:-dataset_v2}
MODE=${MODE:-primary}
CHUNK=${CHUNK:-10000}
PARTITION=${PARTITION:-}
GRES=${GRES:-}

DO_CHECK=1 DO_SCORE=1 DO_TRAIN=1 GO=0
for arg in "$@"; do
    case "$arg" in
        --go)         GO=1 ;;
        --no-check)   DO_CHECK=0 ;;
        --score-only) DO_TRAIN=0 ;;
        --train-only) DO_CHECK=0; DO_SCORE=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

MANIFEST="outputs/upstream_exon/${DATASET}/b1_${MODE}_manifest.csv"
RESULTS="outputs/upstream_exon/${DATASET}/b1_${MODE}_ag_results.csv"

[[ -f "$MANIFEST" ]] || { echo "FATAL: missing $MANIFEST" >&2; exit 1; }

TOTAL=$(( $(wc -l < "$MANIFEST") - 1 ))
DONE=0
[[ -f "$RESULTS" ]] && DONE=$(( $(wc -l < "$RESULTS") - 1 ))
REMAINING=$(( TOTAL - DONE ))
(( REMAINING < 0 )) && REMAINING=0
N_CHUNKS=$(( (REMAINING + CHUNK - 1) / CHUNK ))
(( DO_SCORE == 1 )) || N_CHUNKS=0

# sbatch overrides, applied to every stage
EXTRA=()
[[ -n "$PARTITION" ]] && EXTRA+=(--partition="$PARTITION")
[[ -n "$GRES" ]] && EXTRA+=(--gres="$GRES")

echo "=============================================================="
echo " B1 pipeline"
echo "=============================================================="
echo " dataset        : $DATASET / $MODE"
echo " manifest rows  : $TOTAL"
echo " already scored : $DONE"
echo " remaining      : $REMAINING"
echo " chunk size     : $CHUNK"
echo " scoring jobs   : $N_CHUNKS   (chained afterok, one at a time)"
echo " stages         : check=$DO_CHECK score=$DO_SCORE train=$DO_TRAIN"
[[ ${#EXTRA[@]} -gt 0 ]] && echo " overrides      : ${EXTRA[*]}"
echo "=============================================================="

if (( DO_SCORE == 1 && REMAINING == 0 )); then
    echo "Nothing left to score."
    DO_SCORE=0; N_CHUNKS=0
fi

if (( GO == 0 )); then
    echo
    echo "DRY RUN -- nothing submitted. Re-run with --go to submit."
    exit 0
fi

mkdir -p logs

submit() {   # submit <dependency-or-empty> <script> [env assignments...]
    local dep="$1"; shift
    local script="$1"; shift
    local -a depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")
    local id
    if ! id=$(env "$@" sbatch --parsable \
                   "${depflag[@]+"${depflag[@]}"}" \
                   "${EXTRA[@]+"${EXTRA[@]}"}" \
                   "$script" 2>&1); then
        echo "FATAL: sbatch failed for $script" >&2
        echo "$id" >&2
        echo "Nothing further was submitted." >&2
        exit 1
    fi
    echo "$id"
}

DEP=""

if (( DO_CHECK == 1 )); then
    ID=$(submit "$DEP" "$SLURM_DIR/check_connectivity.sbatch")
    echo "connectivity check : $ID"
    DEP="$ID"
fi

for (( i = 1; i <= N_CHUNKS; i++ )); do
    ID=$(submit "$DEP" "$SLURM_DIR/score_b1.sbatch" \
                "DATASET=$DATASET" "MODE=$MODE" "CHUNK=$CHUNK")
    printf 'scoring chunk %d/%d : %s\n' "$i" "$N_CHUNKS" "$ID"
    DEP="$ID"
done

if (( DO_TRAIN == 1 )); then
    ID=$(submit "$DEP" "$SLURM_DIR/train_full.sbatch")
    echo "GPU training       : $ID"
fi

echo
echo "watch:  squeue -u \$USER --format='%.10i %.12P %.14j %.8T %.10M %R'"
echo "logs:   logs/"
echo
echo "If a stage fails, downstream jobs stay in DependencyNeverSatisfied and"
echo "can be cancelled with:  scancel -u \$USER --state=PENDING"
