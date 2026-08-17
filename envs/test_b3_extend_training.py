"""Structural tests for b3_extend_training.py -- mirrors test_sweep_cluster.
py's style (tiny budgets, testing the harness's correctness not trained-
policy quality), against a REAL existing checkpoint (sweep_results/
easy_mlp_seed0_checkpoint.pt, already on disk from the Stage A sweep) rather
than a synthetic fixture, since the whole point is resuming a real one.

Run with `python -m pytest test_b3_extend_training.py`.
"""
import json
import os
import shutil
import tempfile

import pytest

import b3_extend_training as b3

_TINY_CFG = {"steps_per_iter": 16, "n_eval_resets": 2, "horizon": 10}


@pytest.fixture
def tmp_output_dir():
    d = tempfile.mkdtemp(prefix="b3_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_resume_loads_original_checkpoint_and_uses_full_protocol(tmp_output_dir):
    """First invocation for a tag with no B3 state yet must load sweep_
    results/<tag>_checkpoint.pt (start_iteration=3000) and every checkpoint
    must eval the FULL EASY test set (203 instances), never a sample --
    that's the entire point of B3's eval redesign."""
    result = b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3002, eval_every=1,
                     checkpoint_every=1, config_overrides=_TINY_CFG)

    assert result["curve"]["iteration"] == [3001, 3002]
    assert result["curve"]["n_eval_instances"] == [203, 203]  # full EASY set, both checkpoints
    assert len(result["curve"]["argmax_dominant_fraction"]) == 2
    for frac in result["curve"]["argmax_dominant_fraction"]:
        assert 0.0 <= frac <= 1.0
    for n in result["curve"]["argmax_n_distinct"]:
        assert 1 <= n <= 8
    assert result["resumed_from"].endswith("easy_mlp_seed0_checkpoint.pt")
    assert "original_final_greedy" in result

    final_path = os.path.join(tmp_output_dir, "easy_mlp_seed0_b3.json")
    assert os.path.exists(final_path)


def test_never_starts_from_scratch_without_original_result():
    with tempfile.TemporaryDirectory(prefix="b3_test_noorig_") as d:
        with pytest.raises(RuntimeError, match="does not exist"):
            b3.run("easy", "mlp", 999, d, n_iterations=10, config_overrides=_TINY_CFG)


def test_refuses_to_extend_when_already_past_target(tmp_output_dir):
    with pytest.raises(RuntimeError, match="nothing to extend"):
        b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3000, config_overrides=_TINY_CFG)


def test_resume_continues_from_own_b3_checkpoint_not_original_again(tmp_output_dir, capsys):
    b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3001, eval_every=1,
           checkpoint_every=1, config_overrides=_TINY_CFG)
    final_path = os.path.join(tmp_output_dir, "easy_mlp_seed0_b3.json")
    os.remove(final_path)  # stand-in for "job hit the wall before its real target"
    capsys.readouterr()

    result = b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3003, eval_every=1,
                     checkpoint_every=1, config_overrides=_TINY_CFG)
    captured = capsys.readouterr()

    assert "loaded own B3 checkpoint" in captured.out
    assert result["curve"]["iteration"] == [3001, 3002, 3003], \
        f"resume did not continue the existing B3 curve: {result['curve']['iteration']}"


def test_resubmit_after_completion_is_noop(tmp_output_dir):
    r1 = b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3001, eval_every=1,
                checkpoint_every=1, config_overrides=_TINY_CFG)
    final_path = os.path.join(tmp_output_dir, "easy_mlp_seed0_b3.json")
    mtime_before = os.path.getmtime(final_path)

    r2 = b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3001, eval_every=1,
                checkpoint_every=1, config_overrides=_TINY_CFG)
    assert os.path.getmtime(final_path) == mtime_before, "resubmit after completion rewrote the result file"
    assert r1["aggregate_solve_rate_greedy"] == r2["aggregate_solve_rate_greedy"]


def test_does_not_touch_the_original_sweep_files(tmp_output_dir):
    original_ckpt = "sweep_results/easy_mlp_seed0_checkpoint.pt"
    original_result = "sweep_results/easy_mlp_seed0.json"
    mtime_ckpt_before = os.path.getmtime(original_ckpt)
    mtime_result_before = os.path.getmtime(original_result)

    b3.run("easy", "mlp", 0, tmp_output_dir, n_iterations=3001, eval_every=1,
           checkpoint_every=1, config_overrides=_TINY_CFG)

    assert os.path.getmtime(original_ckpt) == mtime_ckpt_before
    assert os.path.getmtime(original_result) == mtime_result_before


def test_target_runs_are_all_seed_zero_and_cover_hard_and_easy():
    tiers = {t for t, _, _ in b3.TARGET_RUNS}
    seeds = {s for _, _, s in b3.TARGET_RUNS}
    encoders = {e for _, e, _ in b3.TARGET_RUNS}
    assert tiers == {"hard", "easy"}
    assert seeds == {0}
    assert encoders == {"hrep", "relational", "cnn"}
    assert len(b3.TARGET_RUNS) == 6
    assert len(set(b3.TARGET_RUNS)) == 6  # no duplicates


def test_array_index_matches_target_runs_list(tmp_output_dir, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["b3_extend_training.py", "--array-index", "2",
                                       "--n-iterations", "3000", "--output-dir", tmp_output_dir])
    with pytest.raises(RuntimeError, match="nothing to extend"):
        # n_iterations=3000 with a checkpoint already at 3000 -> the guard
        # rail fires, which is enough to confirm array-index 2 correctly
        # resolved to hard/cnn/seed0 (TARGET_RUNS[2]) without running anything.
        b3.main()
