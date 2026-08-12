"""B0e step 3 + report: aggregates sweep_results_b0e/*_b0e.json (written by
b0e_array_task.py, one file per (tier, encoder, seed)) into:

  - step 1: per-seed untrained-baseline table
  - step 2: 90-checkpoint interval-protocol re-eval, cross-checked against
    the 6 values A1 already computed (same protocol, greedy mode)
  - step 3: the matched comparison -- 18 (tier, encoder) cells against their
    OWN untrained baseline, both modes, points difference and difference in
    units of the baseline's standard deviation, 95%-CI-overlap flag

plus the three questions the task asks to be answered explicitly: CNN-on-
HARD vs its own chance rate, whether chance differs meaningfully across
encoders, and whether the interval-protocol ranking agrees with the
full-protocol ranking already on record in sweep_results/*.json.

Read-only. Does not touch sweep_results/*.json or sweep_results_b0e/*.json.
"""
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

TIERS = ("easy", "medium", "hard")
ENCODERS = ("cnn", "gnn", "hrep", "mlp", "relational", "vrep")

# A1's original 6 (Stage A), same protocol (interval, greedy, seed=999,
# n_resets=10) -- used as the consistency check for step 2.
A1_KNOWN_GREEDY = {
    "medium_hrep_seed0": 0.078000,
    "medium_vrep_seed2": 0.078000,
    "medium_hrep_seed4": 0.249000,
    "medium_mlp_seed0": 0.168000,
    "easy_cnn_seed0": 0.352000,
    "hard_gnn_seed0": 0.146000,
}


def load_cells(b0e_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(b0e_dir, "*_b0e.json"))):
        with open(path) as f:
            d = json.load(f)
        cells[(d["tier"], d["encoder"], d["seed"])] = d
    return cells


def step1_baseline_table(cells):
    rows = []
    for (tier, encoder, seed), d in sorted(cells.items()):
        rows.append(dict(tier=tier, encoder=encoder, seed=seed,
                          greedy=d["baseline"]["greedy_rate"], stochastic=d["baseline"]["stochastic_rate"]))
    return rows


def step2_consistency_check(cells):
    mismatches = []
    checks = []
    for tag, expected in A1_KNOWN_GREEDY.items():
        tier, rest = tag.split("_", 1)
        encoder, seed_str = rest.rsplit("_seed", 1)
        seed = int(seed_str)
        d = cells.get((tier, encoder, seed))
        if d is None:
            mismatches.append(dict(tag=tag, reason="missing from sweep_results_b0e"))
            continue
        got = d["checkpoint"]["greedy_rate"]
        match = math.isclose(got, expected, abs_tol=1e-9)
        checks.append(dict(tag=tag, expected=expected, got=got, match=match,
                            horizon_used=d["horizon_used"]))
        if not match:
            mismatches.append(dict(tag=tag, expected=expected, got=got, delta=got - expected,
                                    horizon_used=d["horizon_used"]))
    return checks, mismatches


def _ci95(mean, std, n):
    se = std / math.sqrt(n) if n > 0 else float("nan")
    return mean - 1.96 * se, mean + 1.96 * se


def step3_matched_comparison(cells):
    baseline_by_tier_encoder = defaultdict(lambda: defaultdict(list))
    checkpoint_by_tier_encoder = defaultdict(dict)  # (tier,encoder) -> {seed: rates}
    for (tier, encoder, seed), d in cells.items():
        baseline_by_tier_encoder[(tier, encoder)]["greedy"].append(d["baseline"]["greedy_rate"])
        baseline_by_tier_encoder[(tier, encoder)]["stochastic"].append(d["baseline"]["stochastic_rate"])
        checkpoint_by_tier_encoder[(tier, encoder)][seed] = d["checkpoint"]

    rows = []
    for tier in TIERS:
        for encoder in ENCODERS:
            key = (tier, encoder)
            if key not in baseline_by_tier_encoder or key not in checkpoint_by_tier_encoder:
                continue
            for mode in ("greedy", "stochastic"):
                base_vals = np.array(baseline_by_tier_encoder[key][mode])
                base_mean, base_std, n = base_vals.mean(), base_vals.std(ddof=1) if len(base_vals) > 1 else 0.0, len(base_vals)
                ci_lo, ci_hi = _ci95(base_mean, base_std, n)

                seed_checkpoint_rates = [checkpoint_by_tier_encoder[key][s][f"{mode}_rate"]
                                          for s in sorted(checkpoint_by_tier_encoder[key])]
                cp_vals = np.array(seed_checkpoint_rates)
                cp_mean = cp_vals.mean()

                diff_points = float(cp_mean - base_mean)
                diff_std_units = diff_points / base_std if base_std > 0 else float("inf") if diff_points != 0 else 0.0
                overlaps_baseline_ci = bool(ci_lo <= cp_mean <= ci_hi)

                rows.append(dict(
                    tier=tier, encoder=encoder, mode=mode,
                    baseline_mean=round(float(base_mean), 4), baseline_std=round(float(base_std), 4),
                    baseline_ci95=[round(float(ci_lo), 4), round(float(ci_hi), 4)],
                    checkpoint_mean=round(float(cp_mean), 4),
                    diff_points=round(float(diff_points), 4),
                    diff_std_units=round(float(diff_std_units), 3) if math.isfinite(diff_std_units) else diff_std_units,
                    flag_within_baseline_ci=overlaps_baseline_ci,
                ))
    return rows


def question_cnn_on_hard(comparison_rows):
    hits = [r for r in comparison_rows if r["tier"] == "hard" and r["encoder"] == "cnn" and r["mode"] == "greedy"]
    return hits[0] if hits else None


def question_chance_rate_by_encoder(cells):
    by_encoder = defaultdict(list)
    for (tier, encoder, seed), d in cells.items():
        by_encoder[encoder].append(d["baseline"]["greedy_rate"])
    return {enc: dict(mean=round(float(np.mean(v)), 4),
                      std=round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0, n=len(v))
            for enc, v in sorted(by_encoder.items())}


def question_ranking_agreement(cells):
    """Per tier: rank encoders by interval-protocol checkpoint greedy mean
    (mean over the 5 seeds) vs. by the full-protocol aggregate_solve_rate_
    greedy already recorded in each b0e cell's `reported_full_protocol_
    greedy` (copied verbatim from sweep_results/*.json by the array task)."""
    interval_by_tier_encoder = defaultdict(list)
    full_by_tier_encoder = defaultdict(list)
    for (tier, encoder, seed), d in cells.items():
        interval_by_tier_encoder[(tier, encoder)].append(d["checkpoint"]["greedy_rate"])
        full_by_tier_encoder[(tier, encoder)].append(d["checkpoint"]["reported_full_protocol_greedy"])

    out = {}
    for tier in TIERS:
        interval_ranking = sorted(
            ENCODERS, key=lambda e: -np.mean(interval_by_tier_encoder.get((tier, e), [float("-inf")])))
        full_ranking = sorted(
            ENCODERS, key=lambda e: -np.mean(full_by_tier_encoder.get((tier, e), [float("-inf")])))
        out[tier] = dict(interval_ranking=interval_ranking, full_protocol_ranking=full_ranking,
                          agree=(interval_ranking == full_ranking),
                          top1_agrees=(interval_ranking[0] == full_ranking[0]))
    return out


def main(b0e_dir="sweep_results_b0e"):
    cells = load_cells(b0e_dir)
    print(f"loaded {len(cells)} cells from {b0e_dir}")

    print("\n=== step 1: baseline table (first 10 rows) ===")
    for row in step1_baseline_table(cells)[:10]:
        print(row)

    print("\n=== step 2: consistency check against A1's 6 known values ===")
    checks, mismatches = step2_consistency_check(cells)
    for c in checks:
        print(c)
    if mismatches:
        print(f"MISMATCHES ({len(mismatches)}):")
        for m in mismatches:
            print(" ", m)
    else:
        print("all matched.")

    print("\n=== step 3: matched comparison (all rows) ===")
    comparison_rows = step3_matched_comparison(cells)
    for row in comparison_rows:
        print(row)

    print("\n=== CNN on HARD ===")
    print(question_cnn_on_hard(comparison_rows))

    print("\n=== chance rate by encoder ===")
    print(question_chance_rate_by_encoder(cells))

    print("\n=== ranking agreement, interval vs full protocol ===")
    print(json.dumps(question_ranking_agreement(cells), indent=2))


if __name__ == "__main__":
    main()
