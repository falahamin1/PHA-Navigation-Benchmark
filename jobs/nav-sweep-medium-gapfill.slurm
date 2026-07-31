#!/bin/bash

# Gap-fill resubmission -- MEDIUM tier, Relational encoder only (the one
# encoder with zero complete MEDIUM runs; see reports/sweep_report.html for
# the full audit). Narrowed --array to just the 5 gap indices instead of
# 0-29 -- see nav-sweep-easy-gapfill.slurm's header for why.
#
# Array index -> (encoder, seed) (relational = encoder_idx 4, same
# bijection as EASY):
#   20 -> relational seed 0   (was stuck at iter 1500/3000)
#   21 -> relational seed 1   (was stuck at iter 1500/3000)
#   22 -> relational seed 2   (was stuck at iter 1500/3000)
#   23 -> relational seed 3   (was stuck at iter 1500/3000)
#   24 -> relational seed 4   (was stuck at iter 900/3000)
#
# All 5 are RESUMES from their existing
# sweep_results/medium_relational_seed<N>_checkpoint.pt.
#
# PREREQUISITE: relational_deepset.py's region-graph caching fix must be
# deployed to WORK_DIR before this runs (see nav-sweep-easy-gapfill.slurm's
# header for details/measured speedup).
#
# *** QOS/PARTITION BELOW ARE UNVERIFIED -- CONFIRM AGAINST ALPINE BEFORE
# SUBMITTING. *** See nav-sweep-easy-gapfill.slurm's header -- same caveat,
# not re-derived per tier.

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --array=20-24
#SBATCH --job-name=nav-sweep-medium-gapfill
#SBATCH --output=nav-sweep-medium-gapfill.%A_%a.out
#SBATCH --qos=long

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 sweep_cluster.py \
    --tier medium \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results
