"""Step 5b-diag, Part 2: entropy mini-sweep.

Triggered because Part 1 confirmed the optimization-failure fork (train
22.36% ~ test 14.42%, both far below the random baseline of 31.05%) --
though the gradient-norm ratio (0.09x, entropy term SMALLER than the policy
term into the shared encoder) contradicted the specific entropy-domination
mechanism hypothesized. Run anyway per the user's explicit "run if and only
if Part 1 shows the optimization-failure fork" instruction -- if lower
entropy doesn't help, that further confirms entropy isn't the true
bottleneck.

Same seed=0, same budget=200, same frozen config otherwise. entropy_coef in
{0.05 (control, reused from Part 1 -- not retrained), 0.01, 0.005}.
"""
import json
import os

import numpy as np

from diagnose_5b import BUDGET, gradient_norm_ratio, train_to_checkpoint
from fairness_harness import RESULTS_DIR, _greedy_evaluate

ENTROPY_VALUES = [0.05, 0.01, 0.005]


def run_variant(entropy_coef):
    print(f"\n=== entropy_coef={entropy_coef} ===", flush=True)
    model, buffer, train_solve_history, partition_cache, train_instances, test_instances, entropy_history = (
        train_to_checkpoint(entropy_coef=entropy_coef, n_iterations=BUDGET, verbose=True)
    )

    print(f"Evaluating FULL held-out test set (203 instances x30 resets)...", flush=True)
    test_eval = _greedy_evaluate(model, "easy", test_instances, partition_cache, n_resets=30, horizon=40)
    rates = [v["solve_rate"] for v in test_eval.values()]
    solve_rate = float(np.mean(rates))
    dist_100 = sum(1 for r in rates if r == 1.0)
    dist_0 = sum(1 for r in rates if r == 0.0)
    dist_partial = len(rates) - dist_100 - dist_0

    eg, pg, ratio = gradient_norm_ratio(model, buffer)

    entropy_start = float(np.mean(entropy_history[:10]))
    entropy_end = float(np.mean(entropy_history[-10:]))

    print(f"entropy_coef={entropy_coef}: test_solve={solve_rate:.2%} (100%={dist_100}, partial={dist_partial}, "
          f"0%={dist_0}), entropy start->end = {entropy_start:.4f} -> {entropy_end:.4f}, "
          f"grad_ratio(entropy/policy)={ratio:.3f}x", flush=True)

    return {
        "entropy_coef": entropy_coef, "test_solve_rate": solve_rate,
        "dist_100": dist_100, "dist_partial": dist_partial, "dist_0": dist_0,
        "entropy_history": entropy_history, "entropy_start": entropy_start, "entropy_end": entropy_end,
        "entropy_grad_norm": eg, "policy_grad_norm": pg, "grad_ratio": ratio,
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    # Re-run 0.05 fresh too (not reused from Part 1) so all three variants are
    # measured with IDENTICAL methodology (full entropy_history, same-protocol
    # test eval) -- Part 1's run didn't persist a comparable entropy
    # trajectory, and this decision is too consequential to patch together
    # from partially-inconsistent data.
    for e in ENTROPY_VALUES:
        results[str(e)] = run_variant(e)

    with open(os.path.join(RESULTS_DIR, "diag_part2.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== PART 2 SUMMARY TABLE ===", flush=True)
    print(f"{'entropy':>8} | {'test_solve':>10} | {'100%/partial/0%':>18} | {'entropy end':>11} | {'grad_ratio':>10}",
          flush=True)
    for e in ENTROPY_VALUES:
        r = results[str(e)]
        dist = f"{r['dist_100']}/{r['dist_partial']}/{r['dist_0']}"
        print(f"{e:>8} | {r['test_solve_rate']:>9.2%} | {dist:>18} | {r['entropy_end']:>11.4f} | "
              f"{r['grad_ratio']:>9.3f}x", flush=True)

    print("\nDONE PART 2")
