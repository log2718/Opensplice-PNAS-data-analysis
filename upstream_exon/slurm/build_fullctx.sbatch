#!/bin/bash
#SBATCH --job-name=fullctx_build
#SBATCH --account=torch_pr_352_courant
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fullctx_build_%j.out
#SBATCH --error=logs/fullctx_build_%j.err
#
# Generate a full-context permutation manifest. Makes NO AlphaGenome calls,
# so it costs nothing but CPU and can be re-run freely -- except that it
# REFUSES to overwrite an existing manifest, because labels are keyed to
# manifest row_ids and a silently regenerated manifest would orphan them.
#
# The manifests are ~120 MB (upstream) and ~45 MB (middle) and are NOT
# committed; manifest_fingerprint.json carries a SHA-256 over the ordered
# sequences so a manifest built here can be proven identical to one built on
# a laptop.
#
#   LIBRARY=upstream sbatch upstream_exon/slurm/build_fullctx.sbatch
#   LIBRARY=middle   sbatch upstream_exon/slurm/build_fullctx.sbatch
#
# Small enough to run on the login node instead, in tmux:
#   python upstream_exon/build_fullctx_library.py --library upstream

set -euo pipefail

REPO=${REPO:-/scratch/yl12316/Opensplice-PNAS-data-analysis}
LIBRARY=${LIBRARY:?set LIBRARY=upstream or LIBRARY=middle}

cd "$REPO"
mkdir -p logs

PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f ExonSkip7_Lib2_text.txt ]]; then
    echo "FATAL: ExonSkip7_Lib2_text.txt not found at $REPO" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}  library=${LIBRARY}"
echo

echo "--- validate-only rehearsal (5,000 sequences, writes nothing) ---"
"$PYTHON_BIN" upstream_exon/build_fullctx_library.py \
    --library "$LIBRARY" --validate-only -n 5000

echo
echo "--- full build ---"
srun "$PYTHON_BIN" upstream_exon/build_fullctx_library.py --library "$LIBRARY"
