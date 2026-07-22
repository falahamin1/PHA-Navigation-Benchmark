"""Step 5b, Sub-step 3: parallelization scaffold.

Maps a Slurm array task index -> (tier, encoder, seed) and calls
fairness_harness.train_run. Three per-tier arrays (6 encoders x 10 seeds =
60 tasks each) rather than one flat 180-task array, so HARD's per-task
resource request (time/memory) can be set independently of EASY's on
Alpine -- per the earlier per-tier-resource-request discussion.

Usage (from the shell, or as a Slurm array task):
    python launch_sweep.py <tier> <array_index>       # array_index in [0, 59]
    python launch_sweep.py <tier> <array_index> --iterations N   # calibration/smoke override

Slurm submission (one array per tier), e.g.:
    sbatch --array=0-59 --job-name=navsweep-easy   --wrap="python launch_sweep.py easy   \$SLURM_ARRAY_TASK_ID"
    sbatch --array=0-59 --job-name=navsweep-medium --wrap="python launch_sweep.py medium \$SLURM_ARRAY_TASK_ID"
    sbatch --array=0-59 --job-name=navsweep-hard   --wrap="python launch_sweep.py hard   \$SLURM_ARRAY_TASK_ID"

Index layout within a tier's 60-task array: index = encoder_idx * 10 + seed,
i.e. index 0-9 = ENCODER_NAMES[0] (hrep) seeds 0-9, index 10-19 = ENCODER_NAMES[1]
(vrep) seeds 0-9, etc. -- a simple, gap-free, duplicate-free bijection onto
ENCODER_NAMES x SEEDS, verified by test_fairness_harness.py's array-mapping test.
"""
import argparse
import sys

from fairness_harness import ENCODER_NAMES, SEEDS, TIERS, train_run

N_ENCODERS = len(ENCODER_NAMES)
N_SEEDS = len(SEEDS)
TASKS_PER_TIER = N_ENCODERS * N_SEEDS  # 6 * 10 = 60


def index_to_run(tier: str, array_index: int):
    """(tier, array_index in [0, TASKS_PER_TIER)) -> (tier, encoder, seed)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, must be one of {TIERS}")
    if not (0 <= array_index < TASKS_PER_TIER):
        raise ValueError(f"array_index {array_index} out of range [0, {TASKS_PER_TIER})")
    encoder_idx, seed_idx = divmod(array_index, N_SEEDS)
    return tier, ENCODER_NAMES[encoder_idx], SEEDS[seed_idx]


def full_grid():
    """All 180 (tier, encoder, seed) triples the 3 arrays cover, for the
    no-gaps/no-dupes coverage test."""
    grid = []
    for tier in TIERS:
        for idx in range(TASKS_PER_TIER):
            grid.append(index_to_run(tier, idx))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tier", choices=TIERS)
    parser.add_argument("array_index", type=int)
    parser.add_argument("--iterations", type=int, default=None,
                         help="override the frozen training budget (calibration/smoke runs only)")
    args = parser.parse_args()

    tier, encoder, seed = index_to_run(args.tier, args.array_index)
    print(f"array_index={args.array_index} -> tier={tier}, encoder={encoder}, seed={seed}")
    result = train_run(tier, encoder, seed, n_iterations=args.iterations, verbose=True)
    print(f"done: aggregate_solve_rate={result['aggregate_solve_rate']:.2%}, "
          f"wrote {result['_result_path']}")


if __name__ == "__main__":
    main()
