"""AAAI reduced sweep: array-mapping coverage test for
launch_reduced_sweep.py, mirroring test_fairness_harness.py's
test_array_index_mapping but for the 30-task/tier (90-total) reduced sweep
instead of launch_sweep.py's 60-task/tier (180-total) full sweep.

Run directly with `python test_launch_reduced_sweep.py`.
"""
from fairness_harness import ENCODER_NAMES, TIERS
from launch_reduced_sweep import N_SEEDS, TASKS_PER_TIER, full_grid, index_to_run


def test_reduced_array_index_mapping():
    assert TASKS_PER_TIER == 30, f"expected 30 tasks/tier (6 encoders x 5 seeds), got {TASKS_PER_TIER}"
    grid = full_grid()
    assert len(grid) == 3 * TASKS_PER_TIER == 90, f"expected 90 total runs, got {len(grid)}"
    assert len(set(grid)) == 90, "duplicate (tier, encoder, seed) triples in the array mapping"
    expected = {(tier, enc, seed) for tier in TIERS for enc in ENCODER_NAMES for seed in range(N_SEEDS)}
    assert set(grid) == expected, "array mapping does not cover the full 6x3x5 grid exactly"

    assert index_to_run("easy", 0) == ("easy", ENCODER_NAMES[0], 0)
    assert index_to_run("easy", 4) == ("easy", ENCODER_NAMES[0], 4)
    assert index_to_run("easy", 5) == ("easy", ENCODER_NAMES[1], 0)
    assert index_to_run("hard", TASKS_PER_TIER - 1) == ("hard", ENCODER_NAMES[-1], N_SEEDS - 1)
    print(f"test_reduced_array_index_mapping: PASS (3 arrays x {TASKS_PER_TIER} = 90 runs, "
          f"no gaps/dupes, covers exactly the 6x3x5 grid)")


def test_out_of_range_rejected():
    for bad_index in (-1, TASKS_PER_TIER, TASKS_PER_TIER + 100):
        try:
            index_to_run("easy", bad_index)
            raise AssertionError(f"expected ValueError for array_index={bad_index}")
        except ValueError:
            pass
    try:
        index_to_run("nonexistent", 0)
        raise AssertionError("expected ValueError for unknown tier")
    except ValueError:
        pass
    print("test_out_of_range_rejected: PASS")


if __name__ == "__main__":
    test_reduced_array_index_mapping()
    test_out_of_range_rejected()
    print("\nAll launch_reduced_sweep tests passed.")
