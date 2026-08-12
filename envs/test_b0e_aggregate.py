import json
import math

from b0e_aggregate import (
    A1_KNOWN_GREEDY,
    question_chance_rate_by_encoder,
    question_cnn_on_hard,
    question_ranking_agreement,
    step1_baseline_table,
    step2_consistency_check,
    step3_matched_comparison,
)


def _cell(tier, encoder, seed, baseline_greedy, baseline_stochastic,
          checkpoint_greedy, checkpoint_stochastic, reported_full_greedy=None,
          reported_full_stochastic=None, horizon_used=15):
    return dict(
        tier=tier, encoder=encoder, seed=seed, horizon_used=horizon_used,
        baseline=dict(greedy_rate=baseline_greedy, stochastic_rate=baseline_stochastic,
                      self_check=dict(capped_rate=baseline_greedy, uncapped_rate=baseline_greedy, matched=True)),
        checkpoint=dict(greedy_rate=checkpoint_greedy, stochastic_rate=checkpoint_stochastic,
                        self_check=dict(capped_rate=checkpoint_greedy, uncapped_rate=checkpoint_greedy, matched=True),
                        reported_full_protocol_greedy=reported_full_greedy if reported_full_greedy is not None else checkpoint_greedy,
                        reported_full_protocol_stochastic=reported_full_stochastic if reported_full_stochastic is not None else checkpoint_stochastic),
    )


def test_step1_baseline_table_reports_every_seed_not_just_mean():
    cells = {
        ("medium", "hrep", s): _cell("medium", "hrep", s, 0.10 + 0.01 * s, 0.20, 0.15, 0.25)
        for s in range(5)
    }
    rows = step1_baseline_table(cells)
    assert len(rows) == 5
    greedy_vals = sorted(r["greedy"] for r in rows)
    expected = sorted(0.10 + 0.01 * s for s in range(5))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(greedy_vals, expected))


def test_step2_consistency_check_flags_a_real_mismatch():
    built = {}
    for tag, expected in A1_KNOWN_GREEDY.items():
        tier, rest = tag.split("_", 1)
        encoder, seed_str = rest.rsplit("_seed", 1)
        seed = int(seed_str)
        built[(tier, encoder, seed)] = _cell(tier, encoder, seed, 0.1, 0.2, expected, 0.2)
    # Corrupt exactly one to a value that does NOT match A1.
    tier, encoder, seed = "medium", "hrep", 0
    built[(tier, encoder, seed)]["checkpoint"]["greedy_rate"] = 0.5

    checks, mismatches = step2_consistency_check(built)
    assert len(checks) == len(A1_KNOWN_GREEDY)
    assert len(mismatches) == 1
    assert mismatches[0]["tag"] == "medium_hrep_seed0"
    assert math.isclose(mismatches[0]["delta"], 0.5 - A1_KNOWN_GREEDY["medium_hrep_seed0"])


def test_step2_consistency_check_all_match_when_values_are_exact():
    built = {}
    for tag, expected in A1_KNOWN_GREEDY.items():
        tier, rest = tag.split("_", 1)
        encoder, seed_str = rest.rsplit("_seed", 1)
        seed = int(seed_str)
        built[(tier, encoder, seed)] = _cell(tier, encoder, seed, 0.1, 0.2, expected, 0.2)
    checks, mismatches = step2_consistency_check(built)
    assert len(mismatches) == 0
    assert all(c["match"] for c in checks)


def test_step2_consistency_check_reports_missing_cell():
    checks, mismatches = step2_consistency_check({})
    assert len(mismatches) == len(A1_KNOWN_GREEDY)
    assert all(m["reason"] == "missing from sweep_results_b0e" for m in mismatches)


def test_step3_diff_points_and_std_units_hand_computable():
    # Baseline greedy across 5 seeds: 0.08, 0.10, 0.12, 0.10, 0.10 -> mean=0.10, std(ddof=1)=0.01414...
    baseline_vals = [0.08, 0.10, 0.12, 0.10, 0.10]
    cells = {}
    for s, bv in enumerate(baseline_vals):
        # checkpoint greedy fixed at 0.15 for every seed -> checkpoint mean = 0.15
        cells[("medium", "hrep", s)] = _cell("medium", "hrep", s, bv, 0.2, 0.15, 0.25)
    rows = step3_matched_comparison(cells)
    greedy_row = [r for r in rows if r["mode"] == "greedy"][0]

    import numpy as np
    expected_mean = float(np.mean(baseline_vals))
    expected_std = float(np.std(baseline_vals, ddof=1))
    expected_diff = 0.15 - expected_mean
    expected_diff_std_units = expected_diff / expected_std

    assert math.isclose(greedy_row["baseline_mean"], round(expected_mean, 4), abs_tol=1e-9)
    assert math.isclose(greedy_row["baseline_std"], round(expected_std, 4), abs_tol=1e-9)
    assert math.isclose(greedy_row["diff_points"], round(expected_diff, 4), abs_tol=1e-9)
    assert math.isclose(greedy_row["diff_std_units"], round(expected_diff_std_units, 3), abs_tol=1e-2)


def test_step3_flags_checkpoint_within_baseline_ci():
    # checkpoint mean equals baseline mean exactly -> must be flagged (inside CI)
    cells = {("medium", "hrep", s): _cell("medium", "hrep", s, 0.10, 0.2, 0.10, 0.2) for s in range(5)}
    rows = step3_matched_comparison(cells)
    greedy_row = [r for r in rows if r["mode"] == "greedy"][0]
    assert greedy_row["flag_within_baseline_ci"] is True
    assert greedy_row["diff_points"] == 0.0


def test_step3_does_not_flag_checkpoint_far_outside_baseline_ci():
    cells = {("medium", "hrep", s): _cell("medium", "hrep", s, 0.10, 0.2, 0.60, 0.2) for s in range(5)}
    rows = step3_matched_comparison(cells)
    greedy_row = [r for r in rows if r["mode"] == "greedy"][0]
    assert greedy_row["flag_within_baseline_ci"] is False
    assert greedy_row["diff_points"] > 0.4


def test_cnn_on_hard_extracted_correctly():
    cells = {("hard", "cnn", s): _cell("hard", "cnn", s, 0.15, 0.2, 0.15, 0.2) for s in range(5)}
    rows = step3_matched_comparison(cells)
    result = question_cnn_on_hard(rows)
    assert result is not None
    assert result["tier"] == "hard" and result["encoder"] == "cnn" and result["mode"] == "greedy"
    assert result["flag_within_baseline_ci"] is True  # checkpoint == baseline here -> "does not clear chance"


def test_cnn_on_hard_returns_none_when_absent():
    cells = {("hard", "gnn", 0): _cell("hard", "gnn", 0, 0.1, 0.2, 0.1, 0.2)}
    rows = step3_matched_comparison(cells)
    assert question_cnn_on_hard(rows) is None


def test_chance_rate_by_encoder_separates_encoders():
    cells = {}
    cells[("easy", "cnn", 0)] = _cell("easy", "cnn", 0, 0.30, 0.4, 0.3, 0.4)
    cells[("easy", "mlp", 0)] = _cell("easy", "mlp", 0, 0.05, 0.1, 0.3, 0.4)
    result = question_chance_rate_by_encoder(cells)
    assert result["cnn"]["mean"] == 0.3
    assert result["mlp"]["mean"] == 0.05
    assert result["cnn"]["mean"] != result["mlp"]["mean"]


def test_ranking_agreement_detects_a_disagreement():
    # interval protocol says gnn > cnn on hard; full protocol (as recorded
    # in reported_full_protocol_greedy) says the opposite -- must flag agree=False.
    cells = {
        ("hard", "gnn", 0): _cell("hard", "gnn", 0, 0.1, 0.2, 0.50, 0.2, reported_full_greedy=0.10),
        ("hard", "cnn", 0): _cell("hard", "cnn", 0, 0.1, 0.2, 0.10, 0.2, reported_full_greedy=0.50),
    }
    out = question_ranking_agreement(cells)
    assert out["hard"]["interval_ranking"][0] == "gnn"
    assert out["hard"]["full_protocol_ranking"][0] == "cnn"
    assert out["hard"]["agree"] is False
    assert out["hard"]["top1_agrees"] is False


def test_step3_rows_are_json_serializable():
    # Regression test: flag_within_baseline_ci and diff_points were numpy
    # bool_/float64 before an explicit cast -- numpy bool_ fails `is True`
    # identity checks and (depending on numpy/json versions) can raise on
    # json.dumps. Every row must survive a real round-trip.
    cells = {("medium", "hrep", s): _cell("medium", "hrep", s, 0.08 + 0.01 * s, 0.2, 0.15, 0.25)
             for s in range(5)}
    rows = step3_matched_comparison(cells)
    serialized = json.dumps(rows)
    reloaded = json.loads(serialized)
    assert reloaded == rows
    assert all(isinstance(r["flag_within_baseline_ci"], bool) for r in rows)


def test_ranking_agreement_confirms_when_rankings_match():
    cells = {
        ("hard", "gnn", 0): _cell("hard", "gnn", 0, 0.1, 0.2, 0.50, 0.2, reported_full_greedy=0.50),
        ("hard", "cnn", 0): _cell("hard", "cnn", 0, 0.1, 0.2, 0.10, 0.2, reported_full_greedy=0.10),
    }
    out = question_ranking_agreement(cells)
    assert out["hard"]["top1_agrees"] is True
