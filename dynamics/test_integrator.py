"""Gate 1 tests: run directly with `python test_integrator.py`.

Test 1 -- exact reproduction of the four verified C2E2 mode vector fields.
Test 2 -- convergence: from rest, each of the 8 directions drives velocity
toward its own steady-state direction (analytic and empirical checks).
"""

import numpy as np

from integrator import (
    A_V_C2E2,
    C2E2_VERIFIED_MODE_DRIFT,
    DIRECTIONS,
    DRIFT_COUPLING,
    integrate,
    steady_state_velocity,
    velocity_rhs,
)

# Literal transcription of SPEC.md 2.1's four mode equations, independent of
# velocity_rhs(), so a transcription bug in one can't hide behind the other.
_LITERAL_MODE_RHS = {
    0: lambda v1, v2: (-1.2 * v1 + 0.1 * v2 - 0.1, 0.1 * v1 - 1.2 * v2 + 1.2),
    1: lambda v1, v2: (-1.2 * v1 + 0.1 * v2 - 4.8, 0.1 * v1 - 1.2 * v2 + 0.4),
    2: lambda v1, v2: (-1.2 * v1 + 0.1 * v2 + 2.4, 0.1 * v1 - 1.2 * v2 - 0.2),
    3: lambda v1, v2: (-1.2 * v1 + 0.1 * v2 + 3.9, 0.1 * v1 - 1.2 * v2 - 3.9),
}


def test_c2e2_modes_exact():
    rng = np.random.default_rng(0)
    for mode, (c1, c2) in C2E2_VERIFIED_MODE_DRIFT.items():
        drift = np.array([c1, c2])
        for _ in range(20):
            v = rng.uniform(-2.0, 2.0, size=2)
            got1, got2 = velocity_rhs(v, drift, A_v=A_V_C2E2)
            exp1, exp2 = _LITERAL_MODE_RHS[mode](v[0], v[1])
            assert abs(got1 - exp1) < 1e-12, f"mode {mode} v1dot mismatch: {got1} vs {exp1}"
            assert abs(got2 - exp2) < 1e-12, f"mode {mode} v2dot mismatch: {got2} vs {exp2}"
    print("test_c2e2_modes_exact: PASS (4 modes x 20 random v samples, machine precision)")


def test_convergence():
    # A_V_C2E2 has a small off-diagonal coupling (0.1), so the steady state
    # for a pure cardinal direction (E/N/W/S) is NOT exactly aligned with u --
    # it's rotated by ~4.7 degrees (cos_sim ~0.9965). Diagonal directions
    # (NE/NW/SW/SE) sit on the coupling's symmetry axis and ARE exact. This
    # is correct physics from the verified A matrix, not a bug -- confirmed
    # by cross-checking against the closed-form analytic steady state below.
    n_steps = 400  # dt=0.05 -> 20s, eigenvalues of A_V_C2E2 ~ -1.1/-1.3 -> settles in ~a few seconds
    for name, u in DIRECTIONS.items():
        state0 = np.zeros(4)
        traj = integrate(state0, u, n_steps=n_steps)
        v_final = traj[-1, 2:4]

        cos_sim = np.dot(v_final, u) / np.linalg.norm(v_final)
        assert cos_sim > 0.99, f"{name}: final velocity not roughly aligned with u (cos_sim={cos_sim:.6f})"

        v_ss_expected = steady_state_velocity(u)
        assert np.allclose(v_final, v_ss_expected, atol=1e-2), (
            f"{name}: final velocity {v_final} != analytic steady state {v_ss_expected}"
        )
    print(f"test_convergence: PASS (8 directions, cos_sim>0.99, matches analytic steady state, "
          f"k={DRIFT_COUPLING})")


if __name__ == "__main__":
    test_c2e2_modes_exact()
    test_convergence()
    print("Gate 1 tests: ALL PASS")
