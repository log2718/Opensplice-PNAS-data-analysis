#!/bin/bash
#SBATCH --job-name=b1_netcheck
#SBATCH --account=torch_pr_352_general
#SBATCH --partition=cpu_public
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/b1_netcheck_%j.out
#SBATCH --error=logs/b1_netcheck_%j.err
#
# Answers one question: can a Torch COMPUTE node reach the AlphaGenome API?
#
# If yes, scoring can run on the cluster via score_b1.sbatch.
# If no, score locally and rsync the results CSV across (see README).
#
#     mkdir -p logs
#     sbatch upstream_exon/slurm/check_connectivity.sbatch
#     cat logs/b1_netcheck_*.out
#
# Spends at most ONE AlphaGenome call, and only if DNS and TLS both succeed.

set -uo pipefail

REPO=/scratch/yl12316/Opensplice-PNAS-data-analysis
ENV=/scratch/yl12316/conda_envs/opensplicerna

cd "$REPO"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
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

echo "--- 4. one real AlphaGenome call ---"
python - <<'PY'
import sys, pathlib
sys.path.insert(0, "analysis_script")
sys.path.insert(0, "upstream_exon")
try:
    import b1_lib as B
    import synthetic_context_lib as L
    import pandas as pd
    lib = pd.read_csv("outputs/synthetic_context/synthetic_exon_library.csv")
    syn = lib[lib.sequence_id == "syn_a5.0_00482"].iloc[0].synthetic_70mer
    core, start = B.build_construct(B.native_b1(), syn)
    import alphagenome_local_prediction_pipeline_minigene as ag
    model = ag.create_model(env_path=".env.alphagenome")
    out = L.score_core(model, core, start, 76)
    print(f"  SUCCESS: AG_mean_logit={out['AG_mean_logit']:+.4f}")
    print("  => compute nodes CAN reach AlphaGenome; use score_b1.sbatch")
except Exception as exc:
    print(f"  FAILED: {type(exc).__name__}: {exc}")
    print("  => score locally and rsync the results CSV to the cluster")
PY
