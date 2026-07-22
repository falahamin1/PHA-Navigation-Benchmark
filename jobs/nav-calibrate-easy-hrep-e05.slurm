#!/bin/bash

# Step 5b-cal, Part 2/3: budget + entropy calibration, entropy_coef=0.05 arm.
# H-Rep on the full clean EASY pool (812 train / 203 test), generous
# iteration ceiling -- CPU jobs are cheap, the point is finding where
# convergence (entropy decay + greedy-solve plateau + greedy/stochastic gap
# closing) actually happens, not economizing on iterations.
#
# Time estimate (from local timing, not cluster-calibrated -- treat as a
# rough floor, not a promise):
#   ~7-8s/iteration training + ~1 eval checkpoint (203 instances x2 modes
#   x10 resets) roughly every 8-10 min of training time at --eval-every=100.
#   5000 iterations => very roughly 12-20h depending on node speed -- may
#   exceed one 24h window.
#
# QoS cap: the 'normal' (soon 'cpu-normal') QoS on this partition caps
# runtime at 24h, so --time is set to the max allowed, not a real estimate
# of total need. If 5000 iterations takes longer than 24h, the job hits the
# wall and stops mid-run -- that's expected, not a failure.
#
# Resume behaviour (same convention as tangram-git/jobs/hrep.slurm):
#   If the job times out or crashes (including hitting the 24h wall above),
#   just resubmit this exact script. calibrate_cluster.py resumes
#   automatically from calibration_results/easy_hrep_entropy0.05_seed0_checkpoint.pt
#   and appends to the existing *_curve.json rather than overwriting it --
#   spanning several 24h submissions to reach 5000 iterations is normal.

#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --job-name=nav-cal-easy-hrep-e05
#SBATCH --output=nav-cal-easy-hrep-e05.%j.out
#SBATCH --qos=cpu-normal

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
    --entropy-coef 0.05 \
    --seed 0 \
    --n-iterations 5000 \
    --eval-every 100 \
    --checkpoint-every 250 \
    --n-eval-resets 10 \
    --output-dir calibration_results
