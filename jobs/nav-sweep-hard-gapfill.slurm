#!/bin/bash

# Gap-fill resubmission -- HARD tier. Every encoder has at least one gap
# here: H-Rep (0/5 complete), V-Rep (2/5 partial), Relational (0/5
# complete), GNN (1/5 partial). See reports/sweep_report.html for the full
# audit. Narrowed --array to just the 13 gap indices instead of 0-29 -- see
# nav-sweep-easy-gapfill.slurm's header for why.
#
# Array index -> (encoder, seed), same bijection as EASY/MEDIUM
# (ENCODER_NAMES order hrep=0, vrep=1, mlp=2, cnn=3, relational=4, gnn=5;
# index = encoder_idx*5 + seed):
#    0 -> hrep seed 0         (was stuck at iter 2400/3000)
#    1 -> hrep seed 1         (was stuck at iter 2700/3000)
#    2 -> hrep seed 2         (was stuck at iter 2100/3000)
#    3 -> hrep seed 3         (was stuck at iter 2700/3000)
#    4 -> hrep seed 4         (was stuck at iter 2400/3000)
#    6 -> vrep seed 1         (was stuck at iter 2700/3000)
#    8 -> vrep seed 3         (was stuck at iter 2700/3000)
#   20 -> relational seed 0   (was stuck at iter 600/3000)
#   21 -> relational seed 1   (was stuck at iter 300/3000)
#   22 -> relational seed 2   (was stuck at iter 300/3000)
#   23 -> relational seed 3   (was stuck at iter 600/3000)
#   24 -> relational seed 4   (was stuck at iter 300/3000)
#   28 -> gnn seed 3          (was stuck at iter 2700/3000)
#
# All 13 are RESUMES from their existing sweep_results/hard_<encoder>_
# seed<N>_checkpoint.pt. (vrep seeds 0/2/4, hrep's other array slots N/A,
# gnn seeds 0/1/2/4, mlp all 5, cnn all 5 are already complete and
# deliberately NOT in this array.)
#
# Note the batch-1 HARD crash (skipped-cell AssertionError from nav_env.py,
# see reports/sweep_report.html) was already fixed at the harness level
# before batch 2 (sweep_cluster.py catches _is_sim_violation and scores it
# as not-solved) -- that fix is orthogonal to and independent of the
# Relational caching fix below, and is already live in this checkout.
# Nothing to do for that here; it's just why these are resumable at all
# rather than crashed-and-stuck.
#
# PREREQUISITE (matters for the 5 relational indices above; the other 8 are
# unaffected): relational_deepset.py's region-graph caching fix must be
# deployed to WORK_DIR before this runs. Measured locally (Mac, single-
# instance timing, not Alpine-calibrated): HARD-tier Relational was
# ~122s/iter pre-fix (3.8x H-Rep's ~32s/iter) and ~17.5s/iter post-fix (now
# slightly FASTER than H-Rep's own ~20s/iter in the same local
# measurement) -- see nav-sweep-easy-gapfill.slurm's header for the EASY-
# tier equivalent.
#
# *** QOS/PARTITION BELOW ARE UNVERIFIED -- CONFIRM AGAINST ALPINE BEFORE
# SUBMITTING. *** See nav-sweep-easy-gapfill.slurm's header -- same caveat,
# not re-derived per tier.

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-4,6,8,20-24,28
#SBATCH --job-name=nav-sweep-hard-gapfill
#SBATCH --output=nav-sweep-hard-gapfill.%A_%a.out
#SBATCH --qos=long

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 sweep_cluster.py \
    --tier hard \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results
