import pytest

from collapse_detector import analyze_run, find_plateaus, is_collapsed, longest_plateau


def _result(values, iterations=None, tier="medium", encoder="hrep", seed=0):
    if iterations is None:
        iterations = [300 * (i + 1) for i in range(len(values))]
    return {
        "tier": tier, "encoder": encoder, "seed": seed,
        "curve": {"iteration": iterations, "greedy_solve": values},
    }


def test_flat_early_plateau_is_flagged():
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    assert is_collapsed(values)
    plateau = longest_plateau(values)
    assert plateau == (0, 4)


def test_monotonic_curve_not_flagged():
    values = [0.10, 0.15, 0.20, 0.27, 0.33, 0.40, 0.48, 0.55, 0.60, 0.65]
    assert not is_collapsed(values)


def test_noisy_but_effectively_flat_plateau_is_flagged():
    # Real curves carry float noise from single-instance flips (see module
    # docstring); a plateau shouldn't require bit-exact equality.
    values = [0.154, 0.1541, 0.1539, 0.1542, 0.1538, 0.30, 0.35, 0.40, 0.42, 0.45]
    assert is_collapsed(values, tol=0.01)
    plateau = longest_plateau(values, tol=0.01)
    assert plateau == (0, 4)


def test_plateau_shorter_than_min_len_not_flagged():
    values = [0.1, 0.1, 0.1, 0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    assert not is_collapsed(values, min_len=5)


def test_plateau_exactly_min_len_is_flagged():
    values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.35, 0.4, 0.45, 0.5]
    assert is_collapsed(values, min_len=5)
    assert not is_collapsed(values, min_len=6)


def test_recovering_run_flagged_with_recovery_true():
    values = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.30, 0.35, 0.40, 0.45]
    result = _result(values)
    rec = analyze_run(result)
    assert rec is not None
    assert rec["plateau_first_iter"] == 300
    assert rec["plateau_last_iter"] == 1800
    assert rec["plateau_len"] == 6
    assert rec["recovered_by_3000"] is True


def test_never_recovers_flagged_with_recovery_false():
    values = [0.078] * 10
    result = _result(values)
    rec = analyze_run(result)
    assert rec is not None
    assert rec["plateau_first_iter"] == 300
    assert rec["plateau_last_iter"] == 3000
    assert rec["recovered_by_3000"] is False
    assert rec["final_greedy"] == pytest.approx(0.078)


def test_late_plateau_touching_final_two_checkpoints_flagged():
    # Plateau spans the A1 protocol boundary (index 7/8) -- still a genuine
    # >=5-length flat run on the logged curve, so it should be flagged, but
    # the record should note it touches the final two so a reader knows
    # the last two points were measured under a different eval protocol.
    values = [0.05, 0.20, 0.30, 0.40, 0.078, 0.078, 0.078, 0.078, 0.08, 0.079]
    result = _result(values)
    rec = analyze_run(result)
    assert rec is not None
    assert rec["plateau_touches_final_two"] is True


def test_healthy_run_from_real_data_not_flagged():
    # medium_hrep_seed4 -- the report's designated "healthy" comparison run.
    values = [0.219, 0.117, 0.157, 0.136, 0.226, 0.179, 0.266, 0.227,
              0.3579047619047619, 0.304952380952381]
    assert not is_collapsed(values)


def test_collapsed_run_from_real_data_flagged():
    # medium_vrep_seed2 -- flat at 0.078 for all 8 interval checkpoints.
    values = [0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078,
              0.08857142857142856, 0.08914285714285713]
    rec = analyze_run(_result(values, encoder="vrep", seed=2))
    assert rec is not None
    assert rec["plateau_first_iter"] == 300
    assert rec["plateau_last_iter"] == 2400
    assert rec["plateau_len"] == 8


@pytest.mark.parametrize("tol,expected", [(0.001, False), (0.01, True), (0.05, True)])
def test_tolerance_sensitivity_on_borderline_curve(tol, expected):
    # Adjacent-checkpoint deltas of ~0.006-0.008 -- collapsed only once tol
    # is loose enough to bridge them. Exercises that flagged-count is
    # tolerance-dependent, not a fixed artifact of the algorithm.
    values = [0.10, 0.106, 0.098, 0.104, 0.097, 0.30, 0.35, 0.40, 0.45, 0.50]
    assert is_collapsed(values, tol=tol) == expected


def test_find_plateaus_returns_maximal_windows_only():
    values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    plateaus = find_plateaus(values, min_len=5)
    # The length-6 window subsumes the two length-5 windows within it.
    assert plateaus == [(0, 5)]


def test_two_disjoint_plateaus_both_reported():
    values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]
    plateaus = find_plateaus(values, min_len=5)
    assert (0, 4) in plateaus
    assert (6, 10) in plateaus
