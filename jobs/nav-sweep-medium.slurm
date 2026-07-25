#!/bin/bash

# Step 5b reduced sweep -- MEDIUM tier, 30-task array (6 encoders x 5 seeds).
# Identical contract to nav-sweep-easy.slurm (frozen entropy_coef=0.01,
# 3000-iteration budget, fingerprint check, checkpoint/resume) -- see
# jobs/README.md's "Step 5b reduced sweep" section. Only --tier and --mem
# differ from the EASY array.
#
# --mem=24G is a conservative bump over EASY's verified 20G, not a
# cluster-measured number -- MEDIUM's partitions/graphs are larger than
# EASY's but no MEDIUM job has actually been run on Alpine yet to confirm
# peak RSS. Watch the first array's `sacct -j <jobid> --format=MaxRSS` and
# raise this if any task looks close to the limit.

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --array=0-29
#SBATCH --job-name=nav-sweep-medium
#SBATCH --output=nav-sweep-medium.%A_%a.out
#SBATCH --qos=cpu-normal

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
