#!/bin/bash
#
# Dump everything needed to choose valid SLURM resources on Torch.
# Read-only: queries SLURM and submits nothing.
#
# Run this ON A TORCH LOGIN NODE, then paste the output back:
#
#     bash upstream_exon/slurm/probe_slurm.sh | tee slurm_probe.txt
#
# What matters in the output:
#   * section 2 lists the partitions that exist -- `cpu_public` did not.
#   * section 3 shows which partitions the account may actually use.
#   * section 5 shows, per partition, whether a GPU is mandatory
#     (TRES=gres/gpu with MinTRESPerJob, or a GPU-only node set).
#   * section 6 shows the max walltime per partition, which decides how many
#     scoring chunks are needed.

set -uo pipefail

ACCOUNT=${ACCOUNT:-torch_pr_352_general}

hr() { printf '\n%s\n%s\n' "$1" "$(printf '=%.0s' {1..70})"; }

hr "0. identity"
echo "user    : $(whoami)"
echo "host    : $(hostname)"
echo "account : ${ACCOUNT}"

hr "1. sinfo summary"
sinfo -s 2>&1 | head -40

hr "2. partitions (name / avail / timelimit / nodes / state)"
sinfo -o "%24P %8a %14l %8D %12t %N" 2>&1 | head -60

hr "3. account associations (which partitions + QOS this account may use)"
sacctmgr -n show assoc user="$(whoami)" \
    format=Account%28,Partition%24,QOS%40,MaxWall%12,GrpTRES%30 2>&1 | head -60

hr "4. QOS definitions"
sacctmgr -n show qos format=Name%24,MaxWall%12,MaxTRESPU%40,MaxJobsPU%10,Priority%10 2>&1 | head -40

hr "5. per-partition detail (GPU requirements, limits, allowed accounts)"
for p in $(sinfo -h -o "%P" | tr -d '*' | sort -u); do
    echo
    echo "--- partition: $p ---"
    scontrol show partition "$p" 2>&1 | \
        tr ' ' '\n' | grep -E \
        'PartitionName|MaxTime|DefaultTime|TRES=|AllowAccounts|AllowQos|DenyAccounts|MaxNodes|MaxCPUsPerNode|DefMemPerNode|MaxMemPerNode|State=' \
        | sed 's/^/    /'
done

hr "6. GPU generic resources available"
sinfo -o "%24P %24G %12N" 2>&1 | head -40

hr "7. does any partition accept a CPU-only job? (test-only, submits nothing)"
for p in $(sinfo -h -o "%P" | tr -d '*' | sort -u); do
    out=$(sbatch --test-only --account="${ACCOUNT}" --partition="$p" \
                 --cpus-per-task=2 --mem=8G --time=01:00:00 \
                 --wrap="true" 2>&1)
    if echo "$out" | grep -qi "allocation failure\|error"; then
        echo "  $p : CPU-only REJECTED  -- $(echo "$out" | head -1 | cut -c1-110)"
    else
        echo "  $p : CPU-only ACCEPTED  -- $(echo "$out" | head -1 | cut -c1-110)"
    fi
done

hr "8. does the known-good GPU allocation still validate?"
sbatch --test-only --account="${ACCOUNT}" --partition=l40s_public \
       --gres=gpu:l40s:1 --cpus-per-task=4 --mem=12G --time=00:30:00 \
       --wrap="true" 2>&1 | head -3

hr "done"
echo "Paste this whole output back so the sbatch resources can be set from"
echo "fact rather than assumption."
