"""Step 6 Gate 0-CL tests: run directly with `python test_closed_loop_oracle.py`.

- test_oracle_replans_from_actual_state: on the known-fragile EASY instance
  (start=1, goal=23, hazard=(20,), path_len=7 -- Gate 0's own reported worst
  open-loop instance, 0% open-loop reach), dynamically finds a seed where the
  fixed open-loop action sequence diverges from the intended path (the exact
  seed/cell/hop is a property of dt/discretization, not hardcoded -- see
  Step A's dt recalibration), then confirms a fresh replan from that drifted
  cell recommends a DIFFERENT action than the stale, pre-committed open-loop
  plan would have issued next -- i.e. the oracle is genuinely closed-loop,
  not open-loop in disguise.
- test_oracle_solves_open_loop_failed_instance: the same instance (0%
  open-loop reach across all 10 v0 seeds per Gate 0) is solved by the
  closed-loop oracle at 100% (10/10 seeds) -- the direct proof that
  replanning recovers what a fixed action sequence couldn't.
- test_oracle_respects_hazards: the oracle's per-step BFS plan never routes
  through a hazard cell, even mid-trajectory after drift.
- test_determinism: same (instance, seed) -> identical oracle trajectory.
"""
import numpy as np

from pool import Instance, build_partition, solve_path
from closed_loop_oracle import closed_loop_oracle, direction_for_hop, instance_to_config
from integrator import DIRECTION_NAMES
from nav_env import HORIZON, NavEnv

# Gate 0's own reported worst-instance case: EASY, start=1, goal=23,
# hazard=(20,), path_len=7, 0% open-loop reach (all "drift_into_wrong_cell").
_FRAGILE_INSTANCE = Instance(tier="easy", partition_seed=0, start_cell=1, goal_cell=23,
                              hazard_cells=(20,), initial_velocity_sign=(1, 1))


def test_oracle_replans_from_actual_state():
    # NOTE: which exact cell the object drifts into (and at which hop) is a
    # property of the specific dt/discretization, not a fixed fact about this
    # instance -- so this test DISCOVERS the divergence dynamically (searching
    # seeds if needed) rather than hardcoding a specific (seed, cell, action),
    # which would silently go stale on any future dt recalibration.
    partition = build_partition("easy", 0)
    intended_path = solve_path(partition, 1, 23, (20,))
    assert intended_path == [1, 2, 3, 8, 13, 18, 23], f"precondition changed: {intended_path}"
    intended_actions = [direction_for_hop(partition, intended_path[i], intended_path[i + 1])
                         for i in range(len(intended_path) - 1)]

    drifted_cell = stale_next_action = None
    for seed in range(30):
        cfg = instance_to_config(partition, _FRAGILE_INSTANCE)
        env = NavEnv(partition, cfg, horizon=HORIZON)
        obs, _ = env.reset(seed=seed)
        for i, name in enumerate(intended_actions):
            obs, reward, terminated, truncated, info = env.step(DIRECTION_NAMES.index(name))
            if terminated or truncated:
                break
            if obs["cell"] != intended_path[i + 1]:
                drifted_cell = obs["cell"]
                stale_next_action = intended_actions[i + 1] if i + 1 < len(intended_actions) else None
                break
        if drifted_cell is not None:
            break
    assert drifted_cell is not None, (
        "expected at least one of 30 seeds to show open-loop drift on this known-fragile instance -- "
        "if this fires, the instance/dt combination no longer drifts open-loop and a different fragile "
        "instance should be used for this test"
    )
    assert stale_next_action is not None, "drift happened on the last hop, no 'stale next action' to compare against"

    # A fresh replan FROM THE DRIFTED CELL must recommend a different action
    # than blindly continuing the stale, pre-committed plan.
    fresh_path = solve_path(partition, drifted_cell, 23, (20,))
    fresh_action = direction_for_hop(partition, drifted_cell, fresh_path[1])
    assert fresh_action != stale_next_action, (
        f"replanning from the drifted cell recommended the SAME action ({fresh_action}) as the stale "
        "open-loop plan -- this would not distinguish closed-loop from open-loop behavior"
    )
    print(f"test_oracle_replans_from_actual_state: PASS (seed={seed}, drifted to cell {drifted_cell}, stale plan "
          f"would have issued '{stale_next_action}', fresh replan correctly issues '{fresh_action}' instead)")


def test_oracle_solves_open_loop_failed_instance():
    partition = build_partition("easy", 0)
    n_reached = 0
    for seed in range(10):
        reached, steps, trajectory, outcome = closed_loop_oracle(partition, _FRAGILE_INSTANCE, seed)
        assert outcome != "no_strict_path", f"seed={seed}: oracle found no strict path mid-trajectory"
        n_reached += int(reached)
    assert n_reached == 10, (
        f"expected the closed-loop oracle to recover this 0%-open-loop instance at 10/10 seeds, got {n_reached}/10"
    )
    print("test_oracle_solves_open_loop_failed_instance: PASS (10/10 seeds reach goal, "
          "vs 0/10 for the open-loop scripted path on the same instance)")


def test_oracle_respects_hazards():
    for tier, seed, start, goal, hazards in [
        ("easy", 0, 1, 23, (20,)),
        ("easy", 0, 5, 7, (1, 0)),  # the corner-adjacent-to-hazard instance from earlier diagnostics
    ]:
        partition = build_partition(tier, 0)
        instance = Instance(tier=tier, partition_seed=0, start_cell=start, goal_cell=goal,
                             hazard_cells=hazards, initial_velocity_sign=(1, 1))
        cfg = instance_to_config(partition, instance)
        env = NavEnv(partition, cfg, horizon=HORIZON)
        obs, _ = env.reset(seed=0)
        hazard_set = set(hazards)
        for _ in range(HORIZON):
            current_cell = obs["cell"]
            plan = solve_path(partition, current_cell, goal, hazards)
            if plan is None:
                break
            assert hazard_set.isdisjoint(plan), (
                f"oracle's own plan {plan} routes through a hazard cell {hazard_set & set(plan)} -- "
                "solve_path must exclude hazards from the planning graph entirely"
            )
            action_id = DIRECTION_NAMES.index(direction_for_hop(partition, current_cell, plan[1]))
            obs, reward, terminated, truncated, info = env.step(action_id)
            if terminated or truncated:
                break
    print("test_oracle_respects_hazards: PASS (every replanned path excludes hazard cells, "
          "checked at every step across 2 instances)")


def test_determinism():
    partition = build_partition("easy", 0)
    for seed in (0, 3, 7):
        r1 = closed_loop_oracle(partition, _FRAGILE_INSTANCE, seed)
        r2 = closed_loop_oracle(partition, _FRAGILE_INSTANCE, seed)
        assert r1 == r2, f"seed={seed}: two oracle runs with identical (instance, seed) produced different results"
    print("test_determinism: PASS (identical (instance, seed) -> identical oracle trajectory, 3 seeds checked)")


if __name__ == "__main__":
    test_oracle_replans_from_actual_state()
    test_oracle_solves_open_loop_failed_instance()
    test_oracle_respects_hazards()
    test_determinism()
    print("Step 6 Gate 0-CL tests: ALL PASS")
