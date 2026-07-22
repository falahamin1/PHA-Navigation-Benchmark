#!/bin/bash

# Step 5b-cal, Part 3: entropy re-check arm, entropy_coef=0.01.
# Identical setup to nav-calibrate-easy-hrep-e05.slurm (same tier/encoder/
# seed/budget/eval cadence) -- the ONLY difference is --entropy-coef, so the
# two runs are directly comparable. See that script's header for the time
# estimate and resume behaviour (both apply identically here).

#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=amilan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=nav-cal-easy-hrep-e01
#SBATCH --output=nav-cal-easy-hrep-e01.%j.out
#SBATCH --qos=normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 calibrate_cluster.py \
    --tier easy \
    --encoder hrep \
    --entropy-coef 0.01 \
    --seed 0 \
    --n-iterations 5000 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --n-eval-resets 10 \
    --output-dir calibration_results
