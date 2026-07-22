"""Step 1 tests: run directly with `python test_nav_env.py`.

- test_determinism: same seed + same action sequence -> identical trajectory.
- test_boundary_crossing_and_no_skip: hand-constructed crossing lands in the
  right cell; the no-skipped-cells invariant holds under normal (tuned)
  parameters across many random episodes, AND actually fires when parameters
  are deliberately mistuned to force a skip (proving the check has teeth).
- test_reward_accounting: hand-computed reward for a plain step, a goal step,
  and a hazard step, compared bit-exactly to the env's output.
- test_termination: episodes end exactly on goal/hazard entry, and exactly at
  the horizon (never before, never after) when neither is reached.
"""

import numpy as np

from nav_env import (
    DECISION_TIMEOUT_SUBSTEPS,
    GAMMA,
    GOAL_REWARD,
    HAZARD_REWARD,
    STEP_PENALTY,
    NavEnv,
    NavInstanceConfig,
    default_easy_config,
)
from partitions import BoxGridPartition


def make_env(**kwargs):
    partition = BoxGridPartition(num_rows=5, num_cols=5)
    config = default_easy_config(partition)
    return NavEnv(partition, config, **kwargs), partition, config


def test_determinism():
    actions = [0, 3, 6, 1, 4, 7, 2, 5, 0, 1, 2, 3, 4, 5, 6]

    def run():
        env, _, _ = make_env()
        obs, _ = env.reset(seed=42)
        trace = [(obs["state"].copy(), obs["cell"])]
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            trace.append((obs["state"].copy(), obs["cell"], reward, terminated, truncated))
            if terminated or truncated:
                break
        return trace

    trace1 = run()
    trace2 = run()
    assert len(trace1) == len(trace2), "trajectory length differs across identical runs"
    for step1, step2 in zip(trace1, trace2):
        for a, b in zip(step1, step2):
            if isinstance(a, np.ndarray):
                assert np.array_equal(a, b), f"state mismatch: {a} vs {b}"
            else:
                assert a == b, f"value mismatch: {a} vs {b}"
    print(f"test_determinism: PASS ({len(trace1) - 1} steps, bit-identical across two runs)")


def test_boundary_crossing_and_no_skip():
    env, partition, config = make_env()
    env.reset(seed=0, options={"start_state": np.array([0.9, 0.5, 0.0, 0.0])})
    assert env.cell == partition.cell_index(0, 0), f"expected start cell (0,0), got {env.cell}"

    obs, reward, terminated, truncated, info = env.step(0)  # action 0 == "E"
    assert env.cell == partition.cell_index(0, 1), f"expected crossing into (row=0,col=1), got cell {env.cell}"
    assert not terminated and not truncated
    print(f"test_boundary_crossing_and_no_skip: crossing (0,0) -> (0,1) confirmed "
          f"in {info['substeps']} substeps")

    # No-skip invariant holds under tuned parameters across many random episodes.
    rng = np.random.default_rng(1)
    for _ in range(20):
        env, _, _ = make_env()
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        for _ in range(30):
            obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, 8)))
            if terminated or truncated:
                break
    print("test_boundary_crossing_and_no_skip: no-skip invariant held over 20 random episodes (tuned params)")

    # Prove the assertion has teeth: deliberately mistuned k forces a multi-cell
    # jump in a single Euler substep, and the invariant must catch it.
    # STEP A (dt recalibration, 0.05 -> 0.008): k=2000 reliably skipped at the
    # old dt but no longer does at the new, finer dt (finer integration
    # legitimately tolerates more violent k before skipping -- verified
    # empirically to require k on the order of 1e4-1e5 now). Rescaled to
    # 100000, confirmed to jump 6-7 cells (not just barely 2) from several
    # starting positions, so this remains a clear, unambiguous "mistuned"
    # case rather than a borderline one.
    bad_env, _, _ = make_env(drift_coupling=100000.0)
    bad_env.reset(seed=0, options={"start_state": np.array([0.5, 0.5, 0.0, 0.0])})
    try:
        bad_env.step(0)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "expected the no-skipped-cells assertion to fire with a deliberately mistuned drift_coupling"
    print("test_boundary_crossing_and_no_skip: PASS (mistuned params correctly trigger the skip assertion)")


def test_wall_stuck_timeout():
    """Picking a wall-blocked direction from a boundary cell must not hang --
    it should time out, leave the cell unchanged, and return control."""
    env, partition, config = make_env()
    top_right_below_goal = partition.cell_index(partition.num_rows - 1, partition.num_cols - 2)
    assert top_right_below_goal != config.goal_cell and top_right_below_goal not in config.hazard_cells
    centroid = partition.cell_centroid(top_right_below_goal)
    start_state = np.array([centroid[0], centroid[1], 0.0, 0.0])
    env.reset(seed=0, options={"start_state": start_state})
    start_cell = env.cell
    assert env.t == 0

    goal_centroid = partition.cell_centroid(config.goal_cell)
    phi_s = -np.linalg.norm(start_state[:2] - goal_centroid)

    obs, reward, terminated, truncated, info = env.step(2)  # "N" -> straight into the top wall
    assert info["outcome"] == "timeout", f"expected a wall-stuck timeout, got {info}"
    assert env.cell == start_cell, "cell should be unchanged after a timed-out decision"
    assert not terminated and not truncated
    assert info["substeps"] == DECISION_TIMEOUT_SUBSTEPS  # STEP A: was a hardcoded 500, now references the constant
    # directly so this test tracks any future dt/timeout recalibration automatically.

    # Carried forward from Step 1: the DECISION_TIMEOUT_SUBSTEPS physics
    # substeps inside this ONE decision epoch must not multiply into that many
    # step-penalties or consume that many units of the decision horizon --
    # both are applied exactly once, at the env.step() (decision-epoch)
    # granularity, never per substep.
    assert env.t == 1, f"one decision epoch must advance the horizon counter by exactly 1, got t={env.t}"
    phi_next = -np.linalg.norm(obs["state"][:2] - goal_centroid)
    expected_reward = STEP_PENALTY + (GAMMA * phi_next - phi_s)  # single STEP_PENALTY, not DECISION_TIMEOUT_SUBSTEPS x
    assert abs(reward - expected_reward) < 1e-9, f"reward {reward} != single-step formula {expected_reward}"
    assert reward > DECISION_TIMEOUT_SUBSTEPS * STEP_PENALTY, (
        "reward looks like it accumulated a per-substep penalty, not a single one"
    )

    print(f"test_wall_stuck_timeout: PASS (N into top wall timed out after {info['substeps']} substeps, "
          f"cell unchanged, control returned, exactly one step-penalty applied, horizon advanced by exactly 1)")


def test_reward_accounting():
    env, partition, config = make_env()

    # Plain non-terminal step: hand-compute Phi(s), Phi(s'), shaping.
    start_state = np.array([0.9, 0.5, 0.0, 0.0])
    env.reset(seed=0, options={"start_state": start_state})
    goal_centroid = partition.cell_centroid(config.goal_cell)
    phi_s = -np.linalg.norm(start_state[:2] - goal_centroid)

    obs, reward, terminated, truncated, info = env.step(0)  # "E"
    assert not terminated
    phi_next = -np.linalg.norm(obs["state"][:2] - goal_centroid)
    expected = STEP_PENALTY + (GAMMA * phi_next - phi_s)
    assert abs(reward - expected) < 1e-12, f"plain step reward {reward} != hand-computed {expected}"
    print(f"test_reward_accounting: plain step reward={reward:.6f} matches hand computation exactly")

    # Goal entry: place the agent one cell below/left of goal, heading straight in.
    goal_row, goal_col = partition.cell_row_col(config.goal_cell)
    entry_state = np.array([partition.cell_width * (goal_col - 0.1), partition.cell_centroid(config.goal_cell)[1], 0.0, 0.0])
    env2, partition2, config2 = make_env()
    env2.reset(seed=0, options={"start_state": entry_state})
    assert env2.cell != config2.goal_cell
    phi_s2 = -np.linalg.norm(entry_state[:2] - goal_centroid)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(0)  # "E" toward goal
    assert terminated2 and info2["outcome"] == "goal", f"expected goal termination, got {info2}"
    expected2 = STEP_PENALTY + (GAMMA * 0.0 - phi_s2) + GOAL_REWARD
    assert abs(reward2 - expected2) < 1e-12, f"goal reward {reward2} != hand-computed {expected2}"
    print(f"test_reward_accounting: goal-entry reward={reward2:.6f} matches hand computation exactly, terminated=True")

    # Hazard entry: same construction, aimed at the hazard cell instead.
    hz_cell = next(iter(config.hazard_cells))
    hz_row, hz_col = partition.cell_row_col(hz_cell)
    hz_centroid = partition.cell_centroid(hz_cell)
    entry_state3 = np.array([partition.cell_width * (hz_col - 0.1), hz_centroid[1], 0.0, 0.0])
    env3, partition3, config3 = make_env()
    env3.reset(seed=0, options={"start_state": entry_state3})
    assert env3.cell != hz_cell
    goal_centroid3 = partition3.cell_centroid(config3.goal_cell)
    phi_s3 = -np.linalg.norm(entry_state3[:2] - goal_centroid3)
    obs3, reward3, terminated3, truncated3, info3 = env3.step(0)  # "E" toward hazard
    assert terminated3 and info3["outcome"] == "hazard", f"expected hazard termination, got {info3}"
    expected3 = STEP_PENALTY + (GAMMA * 0.0 - phi_s3) + HAZARD_REWARD
    assert abs(reward3 - expected3) < 1e-12, f"hazard reward {reward3} != hand-computed {expected3}"
    print(f"test_reward_accounting: hazard-entry reward={reward3:.6f} matches hand computation exactly, terminated=True")
    print("test_reward_accounting: PASS")


def test_termination():
    partition = BoxGridPartition(num_rows=5, num_cols=5)
    config = default_easy_config(partition)

    # Goal/hazard already exercised in test_reward_accounting; here confirm
    # truncation happens exactly at the horizon and never later, using a
    # short horizon and an action sequence that bounces between two safe
    # cells forever (E, W, E, W, ...) starting far from goal/hazard.
    horizon = 5
    env = NavEnv(partition, config, horizon=horizon)
    start_cell = partition.cell_index(0, 2)  # far from goal (4,4) and hazard (1,3)
    assert start_cell != config.goal_cell and start_cell not in config.hazard_cells
    start_state = np.array([2.5, 0.5, 0.0, 0.0])
    env.reset(seed=0, options={"start_state": start_state})

    for t in range(1, horizon + 1):
        action = 0 if t % 2 == 1 else 4  # alternate E / W
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated, f"unexpectedly reached goal/hazard while bouncing at step {t}: {info}"
        assert obs["cell"] not in config.hazard_cells and obs["cell"] != config.goal_cell
        if t < horizon:
            assert not truncated, f"truncated early at step {t} (horizon={horizon})"
        else:
            assert truncated, f"expected truncated=True at step {t} == horizon"
    print(f"test_termination: PASS (truncated exactly at horizon={horizon}, never before/after; "
          "goal/hazard termination re-confirmed via test_reward_accounting)")


if __name__ == "__main__":
    test_determinism()
    test_boundary_crossing_and_no_skip()
    test_wall_stuck_timeout()
    test_reward_accounting()
    test_termination()
    print("Step 1 tests: ALL PASS")
