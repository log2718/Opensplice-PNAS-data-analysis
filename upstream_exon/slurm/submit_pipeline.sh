#!/bin/bash
#
# Submit the whole B1 pipeline on Torch as a dependency chain:
#
#     connectivity check  ->  AlphaGenome scoring  ->  GPU training
#
# Each stage only starts if the previous one succeeded (afterok), so a failed
# connectivity check or an incomplete scoring run never reaches the GPU queue.
#
# Usage, from the repository root on Torch:
#
#     bash upstream_exon/slurm/submit_pipeline.sh            # full chain
#     bash upstream_exon/slurm/submit_pipeline.sh --no-check # skip connectivity
#     bash upstream_exon/slurm/submit_pipeline.sh --score-only
#     bash upstream_exon/slurm/submit_pipeline.sh --train-only
#
# Everything is resumable: re-running after an interruption continues rather
# than restarting, for both scoring (keyed on row_id) and training (last.pt).

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs

SLURM_DIR=upstream_exon/slurm
DO_CHECK=1
DO_SCORE=1
DO_TRAIN=1

for arg in "$@"; do
    case "$arg" in
        --no-check)   DO_CHECK=0 ;;
        --score-only) DO_TRAIN=0 ;;
        --train-only) DO_CHECK=0; DO_SCORE=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

DEP=""

if [[ "$DO_CHECK" == "1" ]]; then
    CHECK_ID=$(sbatch --parsable "$SLURM_DIR/check_connectivity.sbatch")
    echo "connectivity check : $CHECK_ID"
    DEP="--dependency=afterok:$CHECK_ID"
fi

if [[ "$DO_SCORE" == "1" ]]; then
    # shellcheck disable=SC2086
    SCORE_ID=$(sbatch --parsable $DEP "$SLURM_DIR/score_b1.sbatch")
    echo "scoring (40,000)   : $SCORE_ID"
    DEP="--dependency=afterok:$SCORE_ID"
fi

if [[ "$DO_TRAIN" == "1" ]]; then
    # shellcheck disable=SC2086
    TRAIN_ID=$(sbatch --parsable $DEP "$SLURM_DIR/train_full.sbatch")
    echo "GPU training       : $TRAIN_ID"
fi

echo
echo "watch with:  squeue -u \$USER"
echo "logs in:     logs/"
