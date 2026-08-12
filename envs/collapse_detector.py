"""Stage A / A2: formal collapse detector for sweep_results/*.json curves.

A run's curve.greedy_solve is the reported metric (see sweep_cluster.py's
checkpoint schedule and A1's finding that the first 8 checkpoints share one
eval protocol -- 100-instance fixed subsample x 10 resets -- and the last 2
share another -- full test set x 30 resets). This module flags a run as
"collapsed" when greedy_solve sits within a small tolerance band for 5 or
more CONSECUTIVE checkpoints, on the curve exactly as logged (10 points,
protocol switch included) -- that's what every figure actually plots, so
that's what the detector has to answer for.

Tolerance choice: greedy-mode eval outcomes are deterministic given the
policy (no stochastic action sampling; the only randomness is which of the
fixed reset seeds 0..n_resets-1 is used, and those are fixed across
checkpoints). So a genuinely unchanging policy on a genuinely unchanging
instance set reproduces the EXACT same solve rate -- range 0.0, not just
"close". A nonzero range within a plateau reflects real optimizer motion
(gradient noise nudging a few borderline instances) that isn't a behavioral
change. TOL=0.01 (one percentage point of solve rate) is chosen as the
default: small enough to not paper over a genuine trend (typical healthy
runs move by 0.02-0.15 between adjacent checkpoints in this sweep -- see
sweep_results), large enough to absorb single-instance flips (1/100 = 0.01,
1/175 = 0.0057) without under- or over-counting plateaus by one instance's
worth of noise. Sensitivity at 0.005 and 0.02 is reported alongside 0.01
so the flagged count isn't a threshold artifact -- see report_all_runs().
"""
import glob
import json
import os

DEFAULT_TOL = 0.01
MIN_PLATEAU_LEN = 5


def find_plateaus(values, tol=DEFAULT_TOL, min_len=MIN_PLATEAU_LEN):
    """Returns a list of (start_idx, end_idx) inclusive index pairs, each a
    MAXIMAL run of >= min_len consecutive values whose max-min <= tol (i.e.
    every value in the run sits within a band of width tol -- "within a
    small tolerance of a single value"). Runs shorter than min_len are not
    reported. Maximal windows only (not every sub-window), found by
    brute-force max-min over all O(n^2) windows -- n is 10 per curve in this
    sweep, so this is not a performance-sensitive path.
    """
    n = len(values)
    windows = []
    for i in range(n):
        for j in range(i + min_len - 1, n):
            window = values[i:j + 1]
            if max(window) - min(window) <= tol:
                windows.append((i, j))
    if not windows:
        return []
    # Keep only maximal windows: drop any window fully contained in a
    # longer one that starts no later and ends no earlier.
    maximal = []
    for w in windows:
        if not any(w != w2 and w2[0] <= w[0] and w[1] <= w2[1] for w2 in windows):
            maximal.append(w)
    # Deduplicate and sort by start index.
    return sorted(set(maximal))


def is_collapsed(values, tol=DEFAULT_TOL, min_len=MIN_PLATEAU_LEN):
    return len(find_plateaus(values, tol=tol, min_len=min_len)) > 0


def longest_plateau(values, tol=DEFAULT_TOL, min_len=MIN_PLATEAU_LEN):
    """The single longest plateau (ties broken by earliest start), or None."""
    plateaus = find_plateaus(values, tol=tol, min_len=min_len)
    if not plateaus:
        return None
    return max(plateaus, key=lambda w: (w[1] - w[0], -w[0]))


def analyze_run(result, tol=DEFAULT_TOL, min_len=MIN_PLATEAU_LEN):
    """result: a parsed sweep_results/<tag>.json dict. Returns a dict record
    (None if not flagged) with tier/encoder/seed/plateau bounds/value/
    recovery, using the curve exactly as logged."""
    curve = result["curve"]
    values = curve["greedy_solve"]
    iters = curve["iteration"]
    plateau = longest_plateau(values, tol=tol, min_len=min_len)
    if plateau is None:
        return None
    i, j = plateau
    plateau_values = values[i:j + 1]
    plateau_mean = sum(plateau_values) / len(plateau_values)
    final_greedy = values[-1]
    recovered = (j < len(values) - 1) and (final_greedy - plateau_mean > tol)
    return {
        "tag": f"{result['tier']}_{result['encoder']}_seed{result['seed']}",
        "tier": result["tier"],
        "encoder": result["encoder"],
        "seed": result["seed"],
        "plateau_value": round(plateau_mean, 6),
        "plateau_first_iter": iters[i],
        "plateau_last_iter": iters[j],
        "plateau_len": j - i + 1,
        "final_greedy": final_greedy,
        "recovered_by_3000": recovered,
        "plateau_touches_final_two": j >= len(values) - 2,
    }


def load_all_results(results_dir):
    tags = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name.endswith("_progress.json") or "calibration" in name or name.startswith("diag_"):
            continue
        tags.append(path)
    results = []
    for path in tags:
        with open(path) as f:
            results.append(json.load(f))
    return results


def report_all_runs(results_dir, tol=DEFAULT_TOL, min_len=MIN_PLATEAU_LEN):
    results = load_all_results(results_dir)
    flagged = []
    for r in results:
        rec = analyze_run(r, tol=tol, min_len=min_len)
        if rec is not None:
            flagged.append(rec)
    return flagged, len(results)


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "sweep_results")
    for tol in (0.005, 0.01, 0.02):
        flagged, total = report_all_runs(results_dir, tol=tol)
        print(f"tol={tol}: {len(flagged)}/{total} flagged")
