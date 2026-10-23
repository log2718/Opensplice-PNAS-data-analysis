#!/bin/bash
#SBATCH --job-name=b1_netcheck
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=logs/b1_netcheck_%j.out
#SBATCH --error=logs/b1_netcheck_%j.err
#
# Answers one question: can a Torch COMPUTE node reach the AlphaGenome API?
#
# RESOURCES: these mirror the only allocation known to work on this account
# (l40s_public + gpu:l40s:1 + 4 CPUs + 12G). `cpu_public` does NOT exist on
# this cluster. If a CPU-only partition turns out to be available, override
# on the command line -- sbatch flags beat the #SBATCH directives above:
#
#     sbatch --partition=<cpu_part> --gres=NONE \
#            upstream_exon/slurm/check_connectivity.sbatch
#
# This runs on the SAME resource type as score_b1.sbatch deliberately, so a
# pass here actually predicts that scoring will connect.
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/check_connectivity.sbatch
#     cat logs/b1_netcheck_*.out
#
# Spends at most ONE AlphaGenome call, and only if the client builds.

set -uo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis

cd "$REPO" || exit 1
mkdir -p logs

# Use the environment's interpreter directly. `conda activate` is not
# available on compute nodes here (no conda on PATH, no profile.d hook), so
# activation silently failed and the job fell back to a system python without
# pandas. The interpreter path works without any activation.
PYTHON_BIN=${PYTHON_BIN:-/scratch/yl12316/conda_envs/opensplicerna/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "FATAL: interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
echo "partition=${SLURM_JOB_PARTITION:-?}  gres=${SLURM_JOB_GRES:-none}"
echo

echo "--- 1. DNS ---"
getent hosts alphagenome.deepmind.google || echo "  (no A record / DNS blocked)"
echo

echo "--- 2. generic outbound HTTPS ---"
curl -sS --max-time 20 -o /dev/null -w "  https://www.google.com -> %{http_code}\n" \
     https://www.google.com || echo "  outbound HTTPS FAILED"
echo

echo "--- 3. proxy environment ---"
env | grep -iE 'proxy' || echo "  (no proxy variables set)"
echo

echo "--- 4. credentials ---"
if [[ -f .env.alphagenome ]]; then
    echo "  .env.alphagenome present at repo root"
else
    echo "  MISSING .env.alphagenome at $REPO -- scoring cannot run"
fi
echo

echo "--- 5. one real AlphaGenome call ---"
"$PYTHON_BIN" - <<'PY'
import sys
sys.path.insert(0, "analysis_script")
sys.path.insert(0, "upstream_exon")
try:
    import pandas as pd
    import b1_lib as B
    import synthetic_context_lib as L
    import alphagenome_local_prediction_pipeline_minigene as ag
    lib = pd.read_csv("outputs/synthetic_context/synthetic_exon_library.csv")
    syn = lib[lib.sequence_id == "syn_a5.0_00482"].iloc[0].synthetic_70mer
    core, start = B.build_construct(B.native_b1(), syn)
    model = ag.create_model(env_path=".env.alphagenome")
    out = L.score_core(model, core, start, 76)
    print(f"  SUCCESS: AG_mean_logit={out['AG_mean_logit']:+.4f}")
    print("  => compute nodes CAN reach AlphaGenome; run score_b1.sbatch here")
    sys.exit(0)
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("  => score locally and rsync the results CSV (see README)")
    sys.exit(1)
PY
rc=$?
echo
echo "connectivity check exit code: $rc"
exit $rc
