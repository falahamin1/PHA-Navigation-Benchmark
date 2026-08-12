#!/bin/bash

# B0e -- protocol-matched chance baseline, MEDIUM tier. Identical contract
# to b0e-easy.slurm; only --tier differs (MEDIUM's --mem/--time match
# EASY's rather than nav-sweep-medium.slurm's training-job bump -- eval-only
# has no optimizer/gradient buffers, and horizon=15 applies here too, same
# per-task cost profile as EASY). See b0e-easy.slurm's header for the full
# protocol description and envs/b0e_array_task.py's module docstring for
# the horizon-cap policy and rationale.

#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=0-29
#SBATCH --job-name=b0e-medium
#SBATCH --output=b0e-medium.%A_%a.out
#SBATCH --qos=cpu-normal

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 b0e_array_task.py \
    --tier medium \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir sweep_results_b0e
