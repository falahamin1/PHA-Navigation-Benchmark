#!/bin/bash

# Step 5b reduced sweep -- EASY tier, 30-task array (6 encoders x 5 seeds).
# Frozen config: entropy_coef=0.01, 3000-iteration budget, both eval modes
# logged every checkpoint. See jobs/README.md's "Step 5b reduced sweep"
# section for the full contract (fingerprint check, checkpoint/resume,
# output schema).
#
# Resume behaviour (same convention as the calibration jobs and
# tangram-git/jobs/hrep.slurm): if any task times out before reaching 3000
# iterations, just resubmit this exact script -- sweep_cluster.py resumes
# each still-incomplete task from its checkpoint, and is a no-op for any
# task whose result file already exists.
#
# Array index -> (encoder, seed) mapping: envs/launch_reduced_sweep.py,
# `python envs/launch_reduced_sweep.py` prints the full table.

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=0-29
#SBATCH --job-name=nav-sweep-easy
#SBATCH --output=nav-sweep-easy.%A_%a.out
#SBATCH --qos=cpu-normal

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
