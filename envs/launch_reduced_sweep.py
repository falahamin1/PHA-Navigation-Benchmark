"""AAAI reduced sweep: array index -> (tier, encoder, seed) for the 90-run
sweep (6 encoders x 3 tiers x 5 seeds) frozen for the submission -- entropy_
coef=0.01, 3000-iteration budget, per the launch prompt's frozen config.

Deliberately separate from launch_sweep.py, which remains the 10-seed/
180-run scaffold for the eventual full sweep (see jobs/README.md's "Step 5b
sweep (next, after calibration gates clean)" section) -- that one is not
being submitted now. This module only changes the seed count/task count;
the index layout convention (index = encoder_idx * n_seeds + seed) is the
same bijection launch_sweep.py already uses and test_fairness_harness.py's
array-mapping test already covers for the 10-seed case.

Usage (from the shell, or as a Slurm array task -- see sweep_cluster.py,
which is what jobs/nav-sweep-*.slurm actually invoke):
    python launch_reduced_sweep.py                    # prints the full 90-run mapping
    python launch_reduced_sweep.py <tier> <array_index>   # prints one row, array_index in [0, 29]
"""
import sys

from fairness_harness import ENCODER_NAMES, TIERS

REDUCED_SEEDS = tuple(range(5))
N_ENCODERS = len(ENCODER_NAMES)  # 6
N_SEEDS = len(REDUCED_SEEDS)  # 5
TASKS_PER_TIER = N_ENCODERS * N_SEEDS  # 30


def index_to_run(tier: str, array_index: int):
    """(tier, array_index in [0, TASKS_PER_TIER)) -> (tier, encoder, seed)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}, must be one of {TIERS}")
    if not (0 <= array_index < TASKS_PER_TIER):
        raise ValueError(f"array_index {array_index} out of range [0, {TASKS_PER_TIER})")
    encoder_idx, seed_idx = divmod(array_index, N_SEEDS)
    return tier, ENCODER_NAMES[encoder_idx], REDUCED_SEEDS[seed_idx]


def full_grid():
    """All 90 (tier, encoder, seed) triples the 3 arrays cover, for the
    no-gaps/no-dupes coverage test."""
    grid = []
    for tier in TIERS:
        for idx in range(TASKS_PER_TIER):
            grid.append(index_to_run(tier, idx))
    return grid


def print_mapping():
    for tier in TIERS:
        print(f"=== {tier} (array=0-{TASKS_PER_TIER - 1}) ===")
        for idx in range(TASKS_PER_TIER):
            _, encoder, seed = index_to_run(tier, idx)
            print(f"  {idx:2d} -> encoder={encoder:<10s} seed={seed}")
    grid = full_grid()
    print(f"\ntotal runs: {len(grid)} (expected {3 * TASKS_PER_TIER}); "
          f"unique: {len(set(grid))} (expected {len(grid)})")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_mapping()
    else:
        tier, array_index = sys.argv[1], int(sys.argv[2])
        print(index_to_run(tier, array_index))
