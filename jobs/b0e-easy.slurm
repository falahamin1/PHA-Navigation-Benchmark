#!/bin/bash

# B0e -- protocol-matched chance baseline, EASY tier, 30-task array (6
# encoders x 5 seeds, same array_index -> (encoder, seed) bijection the
# training sweep already uses: envs/launch_reduced_sweep.py).
#
# EVALUATION ONLY. No training. Each task loads its already-trained final
# checkpoint (sweep_results/easy_<encoder>_seed<seed>_checkpoint.pt -- must
# already exist from the Stage A sweep) and computes both an untrained
# baseline and a checkpoint re-eval under the interval protocol (fixed
# 100-instance subsample, seed=999, 10 resets, both eval modes). See
# envs/b0e_array_task.py's module docstring for the full protocol and the
# horizon-cap policy (EASY/MEDIUM capped at 15, empirically lossless;
# HARD stays at the full 40).
#
# Idempotent (matches sweep_cluster.py's convention): safe to resubmit,
# already-finished tasks are a no-op.
#
# --time is a rough estimate from local (non-cluster) timing extrapolation
# at horizon=15, not a cluster-measured number -- EASY/MEDIUM tasks were
# the fast side of that extrapolation (~10-15 min/task); padded generously
# since this is the first run of this exact script on Alpine. Watch the
# first array's actual wall time and tighten if it's overprovisioned.
#
# Writes sweep_results_b0e/easy_<encoder>_seed<seed>_b0e.json -- a NEW
# directory. sweep_results/*.json (the validated Stage A/B sweep output) is
# read-only.

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=0-29
#SBATCH --job-name=b0e-easy
#SBATCH --output=b0e-easy.%A_%a.out
#SBATCH --qos=cpu-normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 b0e_array_task.py \
    --tier easy \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results_b0e
