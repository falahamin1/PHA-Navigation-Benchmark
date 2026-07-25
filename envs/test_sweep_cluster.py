"""AAAI reduced sweep: structural tests for sweep_cluster.py (the checkpoint/
resume/fingerprint-check/result-schema machinery), mirroring
test_fairness_harness.py's style -- tiny budgets so the suite runs quickly,
testing the HARNESS's correctness, not trained-policy quality.

Run directly with `python test_sweep_cluster.py`.

- test_fingerprint_mismatch_aborts: a pool that doesn't match
  REFERENCE_POOL_FINGERPRINTS raises before any training.
- test_smoke_run: one short run end-to-end produces a valid, parseable
  result file with every schema field the statistics module needs.
- test_frozen_entropy_and_budget: the CLI/run() path always trains at
  entropy_coef=0.01 regardless of what n_iterations/tier/encoder are --
  i.e. the frozen config really is frozen.
- test_resume_continues_not_restarts: a checkpoint produced by one run() call
  is picked up by a second call with a larger target, continuing the curve
  and episode/RNG state rather than starting over.
- test_resubmit_after_completion_is_noop: calling run() again once the
  tagged result file exists does not retrain or overwrite it.
- test_sim_violation_training_recovers / test_sim_violation_eval_recovers:
  a monkeypatched NavEnv.step that raises nav_env's real skipped-cell
  AssertionError periodically does not crash training or eval -- confirms
  the reduced sweep's HARD/MEDIUM harness-level fix (see
  fairness_harness._is_sim_violation) actually works, not just "looks
  right by inspection".
- test_unrelated_assertion_still_propagates: an AssertionError that is NOT
  the skipped-cell message is not swallowed -- the catch is narrow on
  purpose.
"""
import json
import os
import shutil
import tempfile

from fairness_harness import ENCODER_NAMES, verify_pool_fingerprint
from launch_reduced_sweep import N_SEEDS, TASKS_PER_TIER, index_to_run
import nav_env
import sweep_cluster as sc

_SIM_VIOLATION_ASSERTION = AssertionError(
    "skipped-cell violation: 1 -> 2 not adjacent "
    "(substep displacement too large for this cell size; retune drift_coupling/dt)"
)

_TINY_CFG = {"steps_per_iter": 16, "n_eval_resets": 2, "horizon": 10}
_TINY_KW = dict(config_overrides=_TINY_CFG, eval_sample_size=5, intermediate_eval_resets=2, verbose=False)
_TMP_DIR = tempfile.mkdtemp(prefix="sweep_cluster_test_")


class _FakeInstance:
    tier = "easy"
    partition_seed = 0
    start_cell = 0
    goal_cell = 1
    hazard_cells = ()
    initial_velocity_sign = (1, 1)


def test_fingerprint_mismatch_aborts():
    try:
        verify_pool_fingerprint("easy", [_FakeInstance()], [_FakeInstance()])
        raise AssertionError("expected RuntimeError on fingerprint mismatch")
    except RuntimeError as e:
        assert "MISMATCH" in str(e)
    try:
        verify_pool_fingerprint("nonexistent", [], [])
        raise AssertionError("expected RuntimeError for a tier with no recorded reference")
    except RuntimeError as e:
        assert "no reference fingerprint" in str(e)
    print("test_fingerprint_mismatch_aborts: PASS")


def test_smoke_run():
    out_dir = os.path.join(_TMP_DIR, "smoke")
    result = sc.run("easy", "mlp", 0, out_dir, n_iterations=4, eval_every=2, checkpoint_every=2, **_TINY_KW)

    final_path = os.path.join(out_dir, "easy_mlp_seed0.json")
    assert os.path.exists(final_path), "final result file was not written"
    with open(final_path) as f:
        on_disk = json.load(f)

    required = {
        "tier", "encoder", "seed", "curve", "per_instance_test_outcomes_greedy",
        "per_instance_test_outcomes_stochastic", "aggregate_solve_rate_greedy",
        "aggregate_solve_rate_stochastic", "config", "pool_provenance",
        "n_train_instances", "n_test_instances",
    }
    missing = required - set(on_disk)
    assert not missing, f"result file missing fields: {missing}"
    assert on_disk["curve"]["iteration"] == [2, 4]
    assert len(on_disk["curve"]["entropy"]) == 2
    assert len(on_disk["per_instance_test_outcomes_greedy"]) == 203  # full eval, EASY test set
    assert on_disk["config"]["commit_hash"], "commit_hash was not recorded"
    assert result["aggregate_solve_rate_greedy"] == on_disk["aggregate_solve_rate_greedy"]
    print("test_smoke_run: PASS")


def test_frozen_entropy_and_budget():
    out_dir = os.path.join(_TMP_DIR, "frozen")
    for tier, encoder, seed in [("easy", "mlp", 0), ("easy", "cnn", 1)]:
        result = sc.run(tier, encoder, seed, out_dir, n_iterations=2, eval_every=2, checkpoint_every=2, **_TINY_KW)
        assert result["config"]["entropy_coef"] == sc.SWEEP_ENTROPY_COEF == 0.01, \
            f"{tier}/{encoder}/seed{seed} did not train at the frozen entropy_coef"
    print("test_frozen_entropy_and_budget: PASS")


def test_resume_continues_not_restarts():
    out_dir = os.path.join(_TMP_DIR, "resume")
    sc.run("easy", "mlp", 2, out_dir, n_iterations=2, eval_every=2, checkpoint_every=2, **_TINY_KW)
    final_path = os.path.join(out_dir, "easy_mlp_seed2.json")
    os.remove(final_path)  # stand-in for "job hit the wall before its real target"

    resumed = sc.run("easy", "mlp", 2, out_dir, n_iterations=6, eval_every=2, checkpoint_every=2, **_TINY_KW)
    assert resumed["curve"]["iteration"] == [2, 4, 6], \
        f"resume did not continue the existing curve: {resumed['curve']['iteration']}"
    print("test_resume_continues_not_restarts: PASS")


def test_resubmit_after_completion_is_noop():
    out_dir = os.path.join(_TMP_DIR, "noop")
    r1 = sc.run("easy", "mlp", 3, out_dir, n_iterations=2, eval_every=2, checkpoint_every=2, **_TINY_KW)
    final_path = os.path.join(out_dir, "easy_mlp_seed3.json")
    mtime_before = os.path.getmtime(final_path)

    r2 = sc.run("easy", "mlp", 3, out_dir, n_iterations=2, eval_every=2, checkpoint_every=2, **_TINY_KW)
    assert os.path.getmtime(final_path) == mtime_before, "resubmit after completion rewrote the result file"
    assert r1["aggregate_solve_rate_greedy"] == r2["aggregate_solve_rate_greedy"]
    print("test_resubmit_after_completion_is_noop: PASS")


def test_sim_violation_training_recovers():
    original_step = nav_env.NavEnv.step
    call_count = {"n": 0}

    def flaky_step(self, action):
        call_count["n"] += 1
        if call_count["n"] % 5 == 0:
            raise _SIM_VIOLATION_ASSERTION
        return original_step(self, action)

    nav_env.NavEnv.step = flaky_step
    try:
        out_dir = os.path.join(_TMP_DIR, "sim_violation_train")
        result = sc.run("easy", "mlp", 10, out_dir, n_iterations=4, eval_every=2, checkpoint_every=2, **_TINY_KW)
    finally:
        nav_env.NavEnv.step = original_step

    assert call_count["n"] > 0, "flaky_step was never called -- test didn't exercise the training loop"
    assert result["train_sim_violations"] > 0, \
        "expected at least one caught training-time sim_violation given the 1-in-5 flaky step"
    print(f"test_sim_violation_training_recovers: PASS (train_sim_violations={result['train_sim_violations']}, "
          f"final_eval_sim_violations={result['final_eval_sim_violations']}, run still completed normally)")


def test_sim_violation_eval_recovers():
    from fairness_harness import _evaluate, get_final_pool
    from ppo_train import MaskedEncoderActorCritic
    from baseline_encoders import MLPEncoder

    train_instances, test_instances = get_final_pool("easy")
    model = MaskedEncoderActorCritic(MLPEncoder())
    partition_cache = {}

    original_step = nav_env.NavEnv.step
    call_count = {"n": 0}

    def flaky_step(self, action):
        call_count["n"] += 1
        if call_count["n"] % 3 == 0:
            raise _SIM_VIOLATION_ASSERTION
        return original_step(self, action)

    nav_env.NavEnv.step = flaky_step
    try:
        results = _evaluate(model, "easy", test_instances[:5], partition_cache, n_resets=4, horizon=10,
                             mode="greedy")
    finally:
        nav_env.NavEnv.step = original_step

    outcomes = [o for v in results.values() for o in v["outcomes"]]
    assert "sim_violation" in outcomes, f"expected a 'sim_violation' outcome among {outcomes}"
    print(f"test_sim_violation_eval_recovers: PASS ({outcomes.count('sim_violation')}/{len(outcomes)} "
          "resets hit the injected violation, none crashed)")


def test_unrelated_assertion_still_propagates():
    original_step = nav_env.NavEnv.step

    def broken_step(self, action):
        raise AssertionError("some completely different invariant failed")

    nav_env.NavEnv.step = broken_step
    try:
        out_dir = os.path.join(_TMP_DIR, "unrelated_assertion")
        raised = False
        try:
            sc.run("easy", "mlp", 11, out_dir, n_iterations=2, eval_every=2, checkpoint_every=2, **_TINY_KW)
        except AssertionError as e:
            raised = True
            assert "some completely different invariant failed" in str(e), str(e)
        assert raised, "an unrelated AssertionError was swallowed instead of propagating"
    finally:
        nav_env.NavEnv.step = original_step
    print("test_unrelated_assertion_still_propagates: PASS")


def test_array_index_matches_direct_call():
    # sweep_cluster.main()'s --array-index path just calls index_to_run then
    # run() with the resolved (tier, encoder, seed) -- confirm that mapping
    # is exactly what jobs/nav-sweep-*.slurm's $SLURM_ARRAY_TASK_ID drives.
    array_index = 7
    encoder_idx, seed_idx = divmod(array_index, N_SEEDS)
    expected = ("medium", ENCODER_NAMES[encoder_idx], seed_idx)
    assert index_to_run("medium", array_index) == expected, index_to_run("medium", array_index)
    print(f"test_array_index_matches_direct_call: PASS (index {array_index} -> {expected}, "
          f"{TASKS_PER_TIER} tasks/tier)")


if __name__ == "__main__":
    test_fingerprint_mismatch_aborts()
    test_smoke_run()
    test_frozen_entropy_and_budget()
    test_resume_continues_not_restarts()
    test_resubmit_after_completion_is_noop()
    test_sim_violation_training_recovers()
    test_sim_violation_eval_recovers()
    test_unrelated_assertion_still_propagates()
    test_array_index_matches_direct_call()
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    print("\nAll sweep_cluster tests passed.")
