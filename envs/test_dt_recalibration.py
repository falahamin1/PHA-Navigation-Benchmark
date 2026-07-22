"""Step A (Step 6 Gate 0-CL fix) tests: run directly with `python test_dt_recalibration.py`.

Context: the closed-loop oracle sweep hit NavEnv's own no-skipped-cells
assertion on HARD (a single Euler substep at the old dt=0.05 crossed an
entire cell). Root cause: dt was tuned only against EASY's unit cells.
nav_env.py's DT is now recalibrated (0.05 -> 0.008), uniform across all
tiers, sized to the empirical global-minimum cell width across all 3 tiers'
pools -- which this file's own measurement confirms is on MEDIUM (0.1779),
not HARD, correcting the initial assumption that HARD's cells alone drove
the requirement. DECISION_TIMEOUT_SUBSTEPS/MAX_SUBSTEPS_HARD_CAP are scaled
alongside it to preserve the original safety-margin ratios.

- test_cell_scale_distribution: measures and prints the per-tier cell-scale
  distribution; asserts the global minimum is what drove the chosen dt, and
  that dt gives >=8 substeps to cross it at steady-state speed.
- test_specific_crashing_trial_fixed: the exact (instance, seed) that crashed
  pre-fix now runs to completion with zero skipped-cell assertions.
- test_no_skipped_cells_hard / _medium / _easy: large-sample (50 instances x
  10 v0 = 500 trials each) zero-skip confirmation via the closed-loop oracle,
  which is what actually stresses the dynamics under real, varied
  trajectories (not just the open-loop script). MEDIUM is included
  deliberately, not just as a regression -- its cells are thinner than
  HARD's, so it's the tier most likely to have LATENT (non-crashing but
  silently corrupting) skip risk under the old dt.
- test_decision_epoch_accounting_unchanged: structural invariants (one
  reward per env.step(), self.t advances by exactly 1 regardless of substep
  count, wall-timeout is non-terminal/same-cell, horizon truncates in
  decision epochs) hold under the new dt -- proving only integration
  granularity changed, not the MDP.
- test_step0_physics_convergence_at_new_dt: re-runs the Gate-1 convergence
  check (test_integrator.py's test_convergence, same tolerances) at the new
  dt with n_steps rescaled to cover the same real-world duration --
  confirming finer integration still converges to the identical analytic
  steady states, i.e. this is a refinement, not a physics change.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dynamics"))

import numpy as np

from integrator import (
    DIRECTIONS,
    DRIFT_COUPLING,
    A_V_C2E2,
    integrate,
    steady_state_velocity,
)
from nav_env import DECISION_TIMEOUT_SUBSTEPS, DT, MAX_SUBSTEPS_HARD_CAP, NavEnv
from pool import Instance, build_partition, generate_pool_with_stats, train_test_split
from closed_loop_oracle import closed_loop_oracle, instance_to_config

POOL_SIZES = {"easy": 800, "medium": 700, "hard": 700}
POOL_SEED = 42
SPLIT_SEED = 7
SAMPLE_SEED = 999
N_SAMPLE = 50
N_V0 = 10

# The exact pre-fix crashing trial (Gate 0-CL sweep report).
_CRASHING_INSTANCE = Instance(tier="hard", partition_seed=865065681, start_cell=19, goal_cell=8,
                               hazard_cells=(2,), initial_velocity_sign=(1, 1))
_CRASHING_SEED = 8


def _cell_min_width(partition, idx):
    """Minimum width of a convex cell: exact for the min-over-ALL-directions
    width of a convex polygon, since that minimum is always attained aligned
    with one of the polygon's own edge normals."""
    if hasattr(partition, "cell_vertices"):
        verts = partition.cell_vertices(idx)
    else:
        x_lo, x_hi, y_lo, y_hi = partition.cell_bounds(idx)
        verts = np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])
    n = len(verts)
    min_width = np.inf
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        edge = b - a
        length = np.linalg.norm(edge)
        if length < 1e-12:
            continue
        normal = np.array([edge[1], -edge[0]]) / length
        projections = verts @ normal
        min_width = min(min_width, projections.max() - projections.min())
    return min_width


def _tier_cell_scales(tier):
    pool, _stats = generate_pool_with_stats(tier, POOL_SIZES[tier], rng_seed=POOL_SEED)
    seeds = sorted({inst.partition_seed for inst in pool})
    scales = []
    for seed in seeds:
        partition = build_partition(tier, seed)
        for idx in range(partition.num_cells):
            scales.append(_cell_min_width(partition, idx))
    return np.array(scales)


def _sample_test_instances(tier):
    pool, _stats = generate_pool_with_stats(tier, POOL_SIZES[tier], rng_seed=POOL_SEED)
    _train, test = train_test_split(pool, test_fraction=0.2, rng_seed=SPLIT_SEED)
    rng = np.random.default_rng(SAMPLE_SEED)
    n = min(N_SAMPLE, len(test))
    idx = rng.choice(len(test), size=n, replace=False)
    return [test[i] for i in idx]


def test_cell_scale_distribution():
    max_speed = max(float(np.linalg.norm(steady_state_velocity(u))) for u in DIRECTIONS.values())
    tier_scales = {tier: _tier_cell_scales(tier) for tier in ("easy", "medium", "hard")}
    for tier, scales in tier_scales.items():
        print(f"  {tier}: min={scales.min():.4f} median={np.median(scales):.4f} max={scales.max():.4f} "
              f"(n={len(scales)} cells)")

    global_min = min(arr.min() for arr in tier_scales.values())
    global_min_tier = min(tier_scales, key=lambda t: tier_scales[t].min())
    assert global_min_tier == "medium", (
        f"expected MEDIUM to hold the global-minimum cell width (correcting the initial HARD assumption), "
        f"got {global_min_tier}"
    )
    assert abs(global_min - 0.1779) < 1e-3, f"expected global min ~0.1779, got {global_min:.4f}"

    margin = global_min / (DT * max_speed)
    assert margin >= 8.0, f"chosen dt={DT} gives only {margin:.2f} substeps across the smallest cell, need >=8"
    print(f"  global min cell width = {global_min:.4f} (on {global_min_tier}), max steady-state speed = "
          f"{max_speed:.4f}, dt={DT} -> {margin:.2f} substeps to cross the smallest cell at steady state")
    print("test_cell_scale_distribution: PASS (global minimum confirmed on MEDIUM, dt gives >=8x margin)")


def test_specific_crashing_trial_fixed():
    partition = build_partition("hard", _CRASHING_INSTANCE.partition_seed)
    reached, steps, trajectory, outcome = closed_loop_oracle(partition, _CRASHING_INSTANCE, _CRASHING_SEED)
    print(f"test_specific_crashing_trial_fixed: PASS (no skipped-cell assertion; reached={reached}, "
          f"steps={steps}, outcome={outcome}, trajectory={trajectory})")


def _large_sample_zero_skips(tier):
    sample = _sample_test_instances(tier)
    n_skips = 0
    skip_details = []
    for inst in sample:
        partition = build_partition(tier, inst.partition_seed)
        for seed in range(N_V0):
            try:
                closed_loop_oracle(partition, inst, seed)
            except AssertionError as e:
                n_skips += 1
                skip_details.append((inst, seed, str(e)))
    return n_skips, skip_details, len(sample) * N_V0


def test_no_skipped_cells_hard():
    n_skips, details, n_trials = _large_sample_zero_skips("hard")
    assert n_skips == 0, f"HARD: {n_skips}/{n_trials} trials skipped a cell: {details[:3]}"
    print(f"test_no_skipped_cells_hard: PASS (0/{n_trials} skips, 50 instances x {N_V0} v0 samples)")


def test_no_skipped_cells_medium():
    n_skips, details, n_trials = _large_sample_zero_skips("medium")
    assert n_skips == 0, f"MEDIUM: {n_skips}/{n_trials} trials skipped a cell: {details[:3]}"
    print(f"test_no_skipped_cells_medium: PASS (0/{n_trials} skips, 50 instances x {N_V0} v0 samples -- "
          "MEDIUM checked deliberately, not just as a regression, since it holds the true global-minimum cells)")


def test_no_skipped_cells_easy():
    n_skips, details, n_trials = _large_sample_zero_skips("easy")
    assert n_skips == 0, f"EASY: {n_skips}/{n_trials} trials skipped a cell: {details[:3]}"
    print(f"test_no_skipped_cells_easy: PASS (0/{n_trials} skips, 50 instances x {N_V0} v0 samples, regression)")


def test_decision_epoch_accounting_unchanged():
    partition = build_partition("easy", 0)
    instance = Instance(tier="easy", partition_seed=0, start_cell=3, goal_cell=0,
                         hazard_cells=(24, 13), initial_velocity_sign=(1, 1))
    cfg = instance_to_config(partition, instance)
    env = NavEnv(partition, cfg, horizon=5)
    obs, _ = env.reset(seed=0)

    assert env.t == 0
    obs, reward, terminated, truncated, info = env.step(0)
    assert env.t == 1, f"expected self.t to advance by exactly 1 per env.step() call, got {env.t}"
    assert isinstance(reward, float) or np.isscalar(reward), "expected exactly one scalar reward per env.step()"
    assert "decision_step" in info and info["decision_step"] == 1

    # Force a wall-timeout: point straight into a boundary wall from a corner
    # cell -- non-terminal, same cell, decision counter still consumed by
    # exactly 1 (design decision #3 in nav_env.py's docstring).
    env2 = NavEnv(partition, instance_to_config(partition, instance), horizon=5)
    obs2, _ = env2.reset(seed=0, options={"start_state": np.array([0.05, 0.05, 0.0, 0.0])})
    cell_before = obs2["cell"]
    t_before = env2.t
    obs2, reward2, terminated2, truncated2, info2 = env2.step(5)  # "SW", straight into the (0,0) corner wall
    assert info2["outcome"] == "timeout", f"expected a wall-timeout outcome, got {info2['outcome']}"
    assert not terminated2, "wall-timeout must be non-terminal"
    assert obs2["cell"] == cell_before, "wall-timeout must leave the cell unchanged"
    assert env2.t == t_before + 1, "wall-timeout must still consume exactly 1 decision epoch"

    # Horizon truncates in decision epochs, not substeps.
    env3 = NavEnv(partition, instance_to_config(partition, instance), horizon=2)
    env3.reset(seed=0)
    _, _, term3a, trunc3a, _ = env3.step(0)
    assert not trunc3a and env3.t == 1
    _, _, term3b, trunc3b, _ = env3.step(0)
    assert (term3b or trunc3b) and env3.t == 2, "expected termination/truncation at exactly horizon=2 epochs"

    print("test_decision_epoch_accounting_unchanged: PASS (self.t advances by 1/step, wall-timeout non-terminal "
          "and same-cell, horizon truncates in decision epochs -- all independent of substep count)")


def test_step0_physics_convergence_at_new_dt():
    # Mirrors test_integrator.py's test_convergence(), at the NEW dt, with
    # n_steps rescaled to cover the same ~20s real-world duration (400*0.05).
    n_steps = int(round(400 * 0.05 / DT))
    for name, u in DIRECTIONS.items():
        state0 = np.zeros(4)
        traj = integrate(state0, u, n_steps=n_steps, dt=DT)
        v_final = traj[-1, 2:4]

        cos_sim = np.dot(v_final, u) / np.linalg.norm(v_final)
        assert cos_sim > 0.99, f"{name}: final velocity not aligned with u under new dt (cos_sim={cos_sim:.6f})"

        v_ss_expected = steady_state_velocity(u)
        assert np.allclose(v_final, v_ss_expected, atol=1e-2), (
            f"{name}: final velocity {v_final} != analytic steady state {v_ss_expected} under new dt"
        )
    print(f"test_step0_physics_convergence_at_new_dt: PASS (8 directions, dt={DT}, n_steps={n_steps} "
          f"[{n_steps * DT:.1f}s simulated], same analytic steady states as dt=0.05 -- refinement, not a "
          "physics change)")


if __name__ == "__main__":
    test_cell_scale_distribution()
    test_specific_crashing_trial_fixed()
    test_no_skipped_cells_hard()
    test_no_skipped_cells_medium()
    test_no_skipped_cells_easy()
    test_decision_epoch_accounting_unchanged()
    test_step0_physics_convergence_at_new_dt()
    print("Step A (dt recalibration) tests: ALL PASS")
