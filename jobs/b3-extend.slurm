#!/bin/bash

# B3 -- convergence calibration. 6-task array, resuming 3 encoders (H-Rep,
# Relational, CNN) on 2 tiers (HARD, EASY) from their existing Stage A/B
# sweep_results/*_checkpoint.pt (all fixed at iteration 3000), continuing
# training to a 15,000-iteration ceiling. All 6 use seed 0 -- deliberate,
# not an oversight: fairness_harness._episode_assignment makes the training/
# eval episode stream a pure function of (seed, episode_count), independent
# of encoder, so seed0-everywhere keeps every cross-encoder comparison on
# the exact same underlying data stream. EASY is included alongside HARD
# because the original sweep's EASY curves showed the largest final-step
# movement of any tier (V-Rep 10.1, H-Rep 8.5, GNN 8.5 points mean absolute
# over the last 300 iterations) -- HARD is not necessarily the slow-to-
# converge tier.
#
# Array index -> (tier, encoder, seed): see envs/b3_extend_training.py's
# TARGET_RUNS list (the authoritative mapping, tested in
# envs/test_b3_extend_training.py::test_target_runs_are_all_seed_zero_and_
# covers_hard_and_easy):
#   0 -> hard hrep seed0       3 -> easy hrep seed0
#   1 -> hard relational seed0 4 -> easy relational seed0
#   2 -> hard cnn seed0        5 -> easy cnn seed0
#
# EVAL PROTOCOL CHANGE from the original sweep: every checkpoint (not just
# the last 2) evaluates the FULL held-out test set at n_eval_resets=30 --
# no cheap-sample-then-full-set switch. That switch is exactly the protocol
# seam A1 found and B0e confirmed changes MEDIUM's #1-ranked encoder
# depending which side of it you read -- Stage D needs a seamless protocol,
# so this run establishes it. Also logs argmax_dominant_fraction (B0a's
# argmax-invariance measure) at every checkpoint alongside entropy/greedy/
# stochastic solve rate.
#
# BUDGET WARNING, not a cluster-measured number: local (non-Alpine) timing
# from a prior HARD-tier run (see nav-sweep-hard-gapfill.slurm's header:
# ~17.5-20s/iter for H-Rep/Relational post-cache-fix) extrapolates to
# ~67h of TRAINING ALONE for 12,000 additional iterations on HARD, before
# counting eval. A full-protocol HARD eval (168 instances x 30 resets x 2
# modes = 10,080 episodes) was ~70-90 min in local B0e timing. At
# eval_every=1000 (12 checkpoints from 3000->15000) that's another
# ~14-18h, landing the full run close to or slightly past the 72h wall on
# a single submission -- expect HARD tasks in particular to need at least
# one resubmission. That's fine: envs/b3_extend_training.py is idempotent/
# resumable (own checkpoint in sweep_results_b3/, same contract as the
# original sweep) -- just resubmit this exact script and it continues, no
# flag needed. EASY should be faster on both counts (larger cells, fewer
# sim_violations historically) but has not been separately timed here.
# If eval is dominating wall-clock once real numbers are in, --eval-every
# is the knob to raise (see jobs/README.md's existing note on this).
#
# *** QOS/PARTITION BELOW ARE UNVERIFIED -- CONFIRM AGAINST ALPINE BEFORE
# SUBMITTING. *** Same caveat as nav-sweep-*-gapfill.slurm: --qos=long is a
# guess at Alpine's naming (not confirmed from this environment, which has
# no cluster access) -- check with `sacctmgr show qos -p` and correct
# --qos/--partition if the name differs or long requires a different
# partition than acpu.

#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --partition=acpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --job-name=b3-extend
#SBATCH --output=b3-extend.%A_%a.out
#SBATCH --qos=long

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate amfa-custom-env

# Adjust WORK_DIR to wherever Nav-Benchmark actually lives on Alpine.
WORK_DIR="/projects/amfa5003/PHA-Navigation-Benchmark/envs"
cd "$WORK_DIR"

python3 b3_extend_training.py \
    --array-index "$SLURM_ARRAY_TASK_ID" \
    --n-iterations 15000 \
    --eval-every 1000 \
    --checkpoint-every 1000 \
    --output-dir sweep_results_b3
