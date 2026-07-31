#!/bin/bash

# Gap-fill resubmission -- EASY tier, Relational encoder only (the one
# encoder with zero complete EASY runs; see reports/sweep_report.html for
# the full audit). Narrowed --array to just the 5 gap indices instead of
# 0-29 so this doesn't burn queue slots on instant no-ops for the 25
# already-complete EASY cells -- sweep_cluster.py's `run()` would skip them
# anyway (idempotent: exits immediately if <tag>.json already exists), but
# there's no reason to submit those tasks at all.
#
# Array index -> (encoder, seed), from envs/launch_reduced_sweep.py's
# bijection (index = encoder_idx*5 + seed, ENCODER_NAMES order
# hrep,vrep,mlp,cnn,relational,gnn -> relational is encoder_idx=4):
#   20 -> relational seed 0   (was stuck at iter 2100/3000)
#   21 -> relational seed 1   (was stuck at iter 2100/3000)
#   22 -> relational seed 2   (was stuck at iter 2100/3000)
#   23 -> relational seed 3   (was stuck at iter 2400/3000)
#   24 -> relational seed 4   (was stuck at iter 2400/3000)
#
# All 5 are RESUMES, not fresh starts -- sweep_cluster.py's run() loads
# each task's existing sweep_results/easy_relational_seed<N>_checkpoint.pt
# and continues from its last completed iteration; nothing else to do here.
#
# PREREQUISITE: relational_deepset.py's region-graph caching fix must be
# deployed to WORK_DIR before this runs -- see the fix's own comment in
# that file (RelationalDeepSet._relational_feature_cache). Without it this
# is the same ~3.4x-slower encoder that timed out twice already; measured
# locally (Mac, single instance timing, not Alpine-calibrated) the fix
# brings EASY-tier Relational to ~8s/iter, matching H-Rep -- 3000 iterations
# should finish in single-digit hours, not near the 24h wall that killed it
# twice before. The 72h budget below is insurance on top of that, not a
# real estimate of need.
#
# *** QOS/PARTITION BELOW ARE UNVERIFIED -- CONFIRM AGAINST ALPINE BEFORE
# SUBMITTING. *** Every existing script in this repo uses
# --qos=cpu-normal, which jobs/README.md documents as capped at 24h. `long`
# below is this benchmark's best guess at Alpine/CURC's extended-walltime
# QOS name, not something verified from this machine (no cluster access
# from here) -- check with `sacctmgr show qos -p` (or your site's
# equivalent) on Alpine and correct --qos/--partition if the name differs
# or if the long QOS requires a different partition than acpu.

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=20-24
#SBATCH --job-name=nav-sweep-easy-gapfill
#SBATCH --output=nav-sweep-easy-gapfill.%A_%a.out
#SBATCH --qos=long

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 sweep_cluster.py \
    --tier easy \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results
