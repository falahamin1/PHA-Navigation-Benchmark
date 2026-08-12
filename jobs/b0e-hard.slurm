#!/bin/bash

# B0e -- protocol-matched chance baseline, HARD tier. Same contract as
# b0e-easy.slurm, but HARD keeps the full horizon=40 (uncapped) for BOTH
# the untrained baseline and the checkpoint re-eval -- local timing found
# no horizon cap below 40 that was lossless for HARD's untrained baseline
# (even horizon=30 lost 1 point off the measured solve rate: an undirected
# random walk occasionally reaches the goal only via a long path a shorter
# cap would clip). Trained/collapsed HARD checkpoints WERE lossless down to
# horizon=20, but the same horizon must be used on both sides of the
# comparison -- see envs/b0e_array_task.py's module docstring -- so HARD
# pays the full cost on both.
#
# --time is generously padded: local (single-core, non-cluster) timing on
# HARD at horizon=40 measured as low as ~0.7-0.9 episodes/sec for a single
# untrained-model eval slice, well under EASY/MEDIUM's capped throughput.
# Extrapolated per-task cost (baseline + checkpoint, both modes, ~4000
# episodes total) landed near 60-100 minutes in that extrapolation --
# not a cluster-measured number, and cluster CPUs may differ substantially
# from the laptop this was timed on. 4 hours is a deliberately generous
# ceiling for the first run; tighten once real sacct timings are in.
#
# --mem matches nav-sweep-hard.slurm's training-job value (32G) rather than
# EASY/MEDIUM's B0e value (20G) for the same reason that job gave: HARD has
# the largest region graphs/partitions of the three tiers, and eval-only
# still builds and holds them all in partition_cache for the tier's 100
# sampled instances.

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-29
#SBATCH --job-name=b0e-hard
#SBATCH --output=b0e-hard.%A_%a.out
#SBATCH --qos=cpu-normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 b0e_array_task.py \
    --tier hard \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results_b0e
