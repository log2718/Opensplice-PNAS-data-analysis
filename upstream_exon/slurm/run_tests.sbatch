#!/bin/bash
#SBATCH --job-name=b1_tests
#SBATCH --account=torch_pr_352_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/b1_tests_%j.out
#SBATCH --error=logs/b1_tests_%j.err
#
# Unit tests, in their own job.
#
# They used to run inside train_full.sbatch, where a missing pytest killed
# job 20673283 before training began. Splitting them out means:
#
#   * a tooling gap can never cost a training run, and
#   * a REAL test failure is never silently swallowed -- this job exits
#     non-zero and says so.
#
# Nothing here needs a GPU, so no --partition and no --gres are requested and
# the job lands on the cluster default partition -- the same allocation the
# repo's scoring jobs used successfully on CPU nodes cs655/cs667/cs695/cs787.
# See train_full.sbatch for the evidence and the GPU fallback.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/run_tests.sbatch
#     cat logs/b1_tests_*.out

set -euo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
cd "$REPO"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

NT=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS="$NT" MKL_NUM_THREADS="$NT" \
       OPENBLAS_NUM_THREADS="$NT" NUMEXPR_NUM_THREADS="$NT"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  partition=${SLURM_JOB_PARTITION:-default}"
"$PYTHON_BIN" -c "import torch; print('torch', torch.__version__,
      '| cuda available:', torch.cuda.is_available())"
echo

# Distinguish "pytest is not installed" from "the tests failed". Conflating
# the two is exactly how a broken build gets mistaken for a tooling problem.
if ! "$PYTHON_BIN" -c "import pytest" 2>/dev/null; then
    echo "FATAL: pytest is not installed in $PYTHON_BIN" >&2
    echo "       Install it once, on a login node:" >&2
    echo "         $PYTHON_BIN -m pip install --no-input pytest" >&2
    echo "       Training does NOT need pytest; train_full.sbatch runs" >&2
    echo "       without it." >&2
    exit 2
fi

echo "=== unit tests ==="
"$PYTHON_BIN" -m pytest upstream_exon/tests/test_upstream.py -q
rc=$?
echo
if [[ $rc -eq 0 ]]; then
    echo "ALL TESTS PASSED"
else
    echo "TESTS FAILED (exit $rc) -- do not trust a training run from this code" >&2
fi
exit $rc
