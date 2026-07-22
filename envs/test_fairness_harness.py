"""Step 5b tests: run directly with `python test_fairness_harness.py`.

All tests use a tiny training budget (small n_iterations/steps_per_iter/
n_eval_resets overrides) so the suite runs quickly -- they test the
HARNESS's structural correctness (RNG design, config uniformity, logging
shape, array mapping), not trained-policy quality (that's Sub-step 1's
calibration run, reported separately).

- test_invariant1_identical_instance_stream: two different encoders, same
  seed -> identical (instance_idx, reset_seed) sequence over their shared
  episode-count prefix.
- test_invariant2_frozen_config: every encoder's logged config is identical
  except n_iterations/n_eval_resets overrides applied uniformly -- proving
  train_run never branches on encoder identity for any hyperparameter.
- test_invariant3_shared_test_set: get_final_pool's test-instance list/order
  is invariant across (simulated) different calls -- the same object
  regardless of which encoder/seed is "asking".
- test_determinism: train_run(tier, encoder, seed) with identical args
  reproduces an identical result (learning curve + per-instance outcomes).
- test_per_instance_logging_present: the result file has per-instance
  outcomes, not just an aggregate.
- test_array_index_mapping: launch_sweep's index->(tier,encoder,seed)
  mapping covers all 180 runs with no gaps or duplicates.
- test_smoke_run: one short train_run end-to-end produces a valid,
  parseable result file with every required field.
"""
import json
import os
import shutil
import tempfile

from fairness_harness import ENCODER_NAMES, FROZEN_CONFIG, TIERS, _episode_assignment, get_final_pool, train_run
from launch_sweep import N_ENCODERS, N_SEEDS, TASKS_PER_TIER, full_grid, index_to_run

_TINY_CFG = {"steps_per_iter": 32, "n_eval_resets": 2}
_TMP_DIR = tempfile.mkdtemp(prefix="fairness_harness_test_")


def test_invariant1_identical_instance_stream():
    tier = "easy"
    seed = 0
    r1 = train_run(tier, "hrep", seed, n_iterations=2, config=_TINY_CFG, results_dir=_TMP_DIR)
    r2 = train_run(tier, "gnn", seed, n_iterations=2, config=_TINY_CFG, results_dir=_TMP_DIR)
    log1, log2 = r1["_instance_log"], r2["_instance_log"]
    assert len(log1) > 0 and len(log2) > 0, "expected at least one completed/in-progress episode assignment"
    common_len = min(len(log1), len(log2))
    assert log1[:common_len] == log2[:common_len], (
        f"hrep and gnn saw DIFFERENT instance sequences at the same seed over their shared prefix: "
        f"{log1[:common_len]} vs {log2[:common_len]}"
    )
    # Cross-check directly against the pure assignment function -- the log
    # should match _episode_assignment's own output exactly, episode by episode.
    train_instances, _test = get_final_pool(tier)
    for i in range(common_len):
        expected_idx, _reset_seed = _episode_assignment(seed, i, len(train_instances))
        assert log1[i] == expected_idx == log2[i], f"episode {i}: instance index mismatch"
    print(f"test_invariant1_identical_instance_stream: PASS ({common_len} shared episodes, byte-identical "
          f"instance-index sequence for hrep vs gnn at seed={seed}, cross-checked against the pure "
          "assignment function)")


def test_invariant2_frozen_config():
    tier = "easy"
    seed = 0
    configs = {}
    for encoder in ENCODER_NAMES:
        result = train_run(tier, encoder, seed, n_iterations=1, config=_TINY_CFG, results_dir=_TMP_DIR)
        configs[encoder] = result["config"]

    reference = configs[ENCODER_NAMES[0]]
    for encoder, cfg in configs.items():
        assert cfg == reference, (
            f"{encoder}'s frozen config differs from {ENCODER_NAMES[0]}'s: {cfg} vs {reference} -- "
            "train_run must never vary a hyperparameter by encoder identity"
        )
    # Every hyperparameter in FROZEN_CONFIG not deliberately overridden by
    # _TINY_CFG (for test speed) must be present unchanged.
    for key, value in FROZEN_CONFIG.items():
        if key in ("n_iterations", *_TINY_CFG.keys()):
            continue
        assert reference[key] == value, f"{key} in the logged config ({reference[key]}) != FROZEN_CONFIG ({value})"
    for key, value in _TINY_CFG.items():
        assert reference[key] == value, f"override {key} in the logged config ({reference[key]}) != _TINY_CFG ({value})"
    print(f"test_invariant2_frozen_config: PASS (all {len(ENCODER_NAMES)} encoders logged an identical "
          "config; every FROZEN_CONFIG hyperparameter present unchanged)")


def test_invariant3_shared_test_set():
    for tier in TIERS:
        _train_a, test_a = get_final_pool(tier)
        _train_b, test_b = get_final_pool(tier)
        assert test_a == test_b, f"{tier}: get_final_pool returned different test sets across calls"
        assert list(test_a) == list(test_b), f"{tier}: test set ORDER differs across calls"
    print(f"test_invariant3_shared_test_set: PASS ({len(TIERS)} tiers, identical test-instance list+order "
          "across repeated calls -- the same object every run/encoder/seed will evaluate against)")


def test_determinism():
    r1 = train_run("easy", "mlp", seed=3, n_iterations=2, config=_TINY_CFG, results_dir=_TMP_DIR)
    r2 = train_run("easy", "mlp", seed=3, n_iterations=2, config=_TINY_CFG, results_dir=_TMP_DIR)
    assert r1["learning_curve"] == r2["learning_curve"], "identical (tier,encoder,seed) produced different curves"
    assert r1["per_instance_test_outcomes"] == r2["per_instance_test_outcomes"], (
        "identical (tier,encoder,seed) produced different per-instance test outcomes"
    )
    assert r1["aggregate_solve_rate"] == r2["aggregate_solve_rate"]
    print("test_determinism: PASS (identical (tier,encoder,seed) -> identical learning curve, "
          "per-instance outcomes, and aggregate solve rate)")


def test_per_instance_logging_present():
    result = train_run("easy", "vrep", seed=1, n_iterations=1, config=_TINY_CFG, results_dir=_TMP_DIR)
    with open(result["_result_path"]) as f:
        on_disk = json.load(f)
    # Step 5b-cal, Part 1: dual-mode (greedy + stochastic) logging is now
    # part of the per-run result-file spec, not just an in-memory extra.
    for mode_key in ("per_instance_test_outcomes_greedy", "per_instance_test_outcomes_stochastic"):
        assert mode_key in on_disk and len(on_disk[mode_key]) > 0, f"missing {mode_key} in result file"
        sample_key = next(iter(on_disk[mode_key]))
        sample = on_disk[mode_key][sample_key]
        assert "outcomes" in sample and "steps" in sample and "solve_rate" in sample
    for agg_key in ("aggregate_solve_rate_greedy", "aggregate_solve_rate_stochastic"):
        assert agg_key in on_disk, f"missing {agg_key} in result file"
    # Backward-compatible aliases (greedy is the standard reported protocol).
    assert "per_instance_test_outcomes" in on_disk and "aggregate_solve_rate" in on_disk
    assert on_disk["aggregate_solve_rate"] == on_disk["aggregate_solve_rate_greedy"]
    assert "config" in on_disk and "learning_curve" in on_disk
    assert "entropy" in on_disk["learning_curve"], "entropy curve must be logged alongside solve rates"
    print(f"test_per_instance_logging_present: PASS (result file has dual-mode "
          f"({len(on_disk['per_instance_test_outcomes_greedy'])} greedy + "
          f"{len(on_disk['per_instance_test_outcomes_stochastic'])} stochastic) per-instance entries, "
          "both aggregates, entropy curve, config, and backward-compatible aliases)")


def test_array_index_mapping():
    grid = full_grid()
    assert len(grid) == 3 * TASKS_PER_TIER == 180, f"expected 180 total runs, got {len(grid)}"
    assert len(set(grid)) == 180, "duplicate (tier, encoder, seed) triples in the array mapping"
    expected = {(tier, enc, seed) for tier in TIERS for enc in ENCODER_NAMES for seed in range(N_SEEDS)}
    assert set(grid) == expected, "array mapping does not cover the full 6x3x10 grid exactly"

    # Spot-check the index layout convention documented in launch_sweep.py.
    assert index_to_run("easy", 0) == ("easy", ENCODER_NAMES[0], 0)
    assert index_to_run("easy", 9) == ("easy", ENCODER_NAMES[0], 9)
    assert index_to_run("easy", 10) == ("easy", ENCODER_NAMES[1], 0)
    assert index_to_run("hard", TASKS_PER_TIER - 1) == ("hard", ENCODER_NAMES[-1], N_SEEDS - 1)
    print(f"test_array_index_mapping: PASS (3 arrays x {TASKS_PER_TIER} = 180 runs, no gaps/dupes, "
          "covers the full 6-encoder x 3-tier x 10-seed grid exactly once)")


def test_smoke_run():
    result = train_run("medium", "relational", seed=7, n_iterations=2, config=_TINY_CFG, results_dir=_TMP_DIR)
    for field in ("tier", "encoder", "seed", "learning_curve", "per_instance_test_outcomes",
                  "per_instance_test_outcomes_greedy", "per_instance_test_outcomes_stochastic",
                  "aggregate_solve_rate", "aggregate_solve_rate_greedy", "aggregate_solve_rate_stochastic",
                  "config", "n_train_instances", "n_test_instances"):
        assert field in result, f"missing required field {field!r} in result"
    assert os.path.exists(result["_result_path"])
    with open(result["_result_path"]) as f:
        json.load(f)  # must be valid, parseable JSON
    print(f"test_smoke_run: PASS (tier=medium, encoder=relational, seed=7, tiny budget, valid result file "
          f"at {result['_result_path']})")


if __name__ == "__main__":
    try:
        test_invariant1_identical_instance_stream()
        test_invariant2_frozen_config()
        test_invariant3_shared_test_set()
        test_determinism()
        test_per_instance_logging_present()
        test_array_index_mapping()
        test_smoke_run()
        print("Step 5b (fairness harness) tests: ALL PASS")
    finally:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
