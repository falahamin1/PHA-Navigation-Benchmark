"""Gate 1 -- 4D Euler integrator for the Nav-Benchmark navigation dynamics.

State layout: state = [x1, x2, v1, v2] (position, velocity) in R^4.
Dynamics: xdot = [v1, v2, A_v @ v + k*u(direction)], fixed-step Euler.
See ../SPEC.md Section 2.1 for the numeric source of truth.
"""

import numpy as np

# --- Velocity-coupling block (2x2), config-selectable per SPEC.md 2.1 ---

A_V_C2E2 = np.array([
    [-1.2, 0.1],
    [0.1, -1.2],
])  # verified default (C2E2 instance)

A_V_FEHNKER_IVANCIC = np.array([
    [-0.8, -0.2],
    [-0.2, -0.8],
])  # original Fehnker-Ivancic paper alternative

# --- Verified per-mode drift constants (C2E2, SPEC.md 2.1) ---
# Validation fixtures only: they anchor A_V_C2E2 against ground truth. Not
# assumed to correspond to any of the 8 compass directions below -- that
# mapping doesn't factor cleanly out of these four numbers (unequal
# magnitudes, no exact opposite pairs). See ROADMAP.md Gate 1 note.
C2E2_VERIFIED_MODE_DRIFT = {
    0: (-0.1, 1.2),
    1: (-4.8, 0.4),
    2: (2.4, -0.2),
    3: (3.9, -3.9),
}

# --- 8-direction action set (fresh design choice for this benchmark) ---

_SQRT_HALF = 2 ** -0.5

DIRECTIONS = {
    "E": np.array([1.0, 0.0]),
    "NE": np.array([_SQRT_HALF, _SQRT_HALF]),
    "N": np.array([0.0, 1.0]),
    "NW": np.array([-_SQRT_HALF, _SQRT_HALF]),
    "W": np.array([-1.0, 0.0]),
    "SW": np.array([-_SQRT_HALF, -_SQRT_HALF]),
    "S": np.array([0.0, -1.0]),
    "SE": np.array([_SQRT_HALF, -_SQRT_HALF]),
}
DIRECTION_NAMES = list(DIRECTIONS.keys())  # index <-> name, |A| = 8

# Drift coupling for the 8-direction action set: B = DRIFT_COUPLING * I.
# Chosen independently of the C2E2 validation constants (see note above), and
# tuned (Step 1, Nav-Benchmark/envs/) against the steps-per-cell diagnostic
# for the EASY tier's unit-width cells: at k=5.0 the worst case (diagonal
# direction, entering a cell already at steady-state speed) crossed a cell in
# only 5 Euler substeps (dt=0.05) -- too little margin against skipping a
# cell entirely in one step, especially for smaller MEDIUM/HARD cells later.
# k=3.0 raises that worst case to 8 substeps/cell (23 from rest) while
# keeping episodes reasonably short. Revisit if HARD-tier cells are much
# smaller than 1.0 in linear scale.
#
# STEP A (Step 6 Gate 0-CL fix, Nav-Benchmark/envs/nav_env.py): this WAS
# revisited -- not by changing drift_coupling (the physics), but by
# recalibrating NavEnv's dt to the empirical global-minimum cell width across
# all 3 tiers (it turned out to be on MEDIUM, not HARD). This constant is
# unchanged; see nav_env.py's DT for the fix and test_dt_recalibration.py for
# the verification.
DRIFT_COUPLING = 3.0

# Standalone default for direct integrate()/euler_step() use (e.g. this
# module's own tests, visualize_gate1.py) -- decoupled from NavEnv's
# operational dt (NavEnv always passes its own dt explicitly, see nav_env.py's
# DT). Left unchanged by Step A on purpose: nothing here depends on cell
# scale.
DT_DEFAULT = 0.05


def velocity_rhs(v, drift, A_v=A_V_C2E2):
    """2D velocity-block derivative: A_v @ v + drift."""
    return A_v @ v + drift


def rhs(state, u_dir, A_v=A_V_C2E2, k=DRIFT_COUPLING):
    """Full 4D state derivative for desired-direction unit vector u_dir."""
    v = state[2:4]
    xdot = v
    vdot = velocity_rhs(v, k * u_dir, A_v=A_v)
    return np.concatenate([xdot, vdot])


def euler_step(state, u_dir, dt=DT_DEFAULT, A_v=A_V_C2E2, k=DRIFT_COUPLING):
    """One fixed-step Euler update: state -> state + dt * rhs(state, u_dir)."""
    return state + dt * rhs(state, u_dir, A_v=A_v, k=k)


def integrate(state0, u_dir, n_steps, dt=DT_DEFAULT, A_v=A_V_C2E2, k=DRIFT_COUPLING):
    """Open-loop rollout under a fixed direction, no boundary handling.

    Returns array of shape (n_steps + 1, 4); states[0] == state0.
    """
    states = np.empty((n_steps + 1, 4))
    states[0] = state0
    for i in range(n_steps):
        states[i + 1] = euler_step(states[i], u_dir, dt=dt, A_v=A_v, k=k)
    return states


def steady_state_velocity(u_dir, A_v=A_V_C2E2, k=DRIFT_COUPLING):
    """Analytic steady-state velocity: solves 0 = A_v @ v_ss + k*u_dir."""
    return -np.linalg.solve(A_v, k * u_dir)
