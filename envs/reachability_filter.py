"""Step 6 Gate 0-CL, Part 2: closed-loop reachability-acceptance filter.

Layered ON TOP OF Step 2c-fix's strict-solvability pool generation
(pool.py), not merged into it: `closed_loop_oracle.py` imports from
`pool.py` (`solve_path`), so `pool.py` cannot import back from
`closed_loop_oracle.py` without a circular import. This module is the
natural home for anything that needs BOTH the pool machinery AND the
dynamics-level reachability oracle.

Step 2c-fix's strict facet-sharing BFS certifies a path is dynamics-
EXECUTABLE in principle (no corner-only hops), but Step 6 Gate 0-CL found a
further, real residual: even under sound integration (Step A), a handful of
instances are deterministically hazard-dead under EVERY v0 sample despite
full replanning -- drift, not planning, defeats them. Those are not
"legitimate difficulty" (a policy that always fails isn't a hard task, it's
an unreachable one) and are excluded here, at the pool-acceptance level, so
the eventual multi-encoder sweep never spends compute on them and no
encoder's solve rate is silently capped by instances nothing could ever
solve.

THRESHOLD: reject only strictly-0%-reach instances (REACHABILITY_THRESHOLD
= 0.0, instances kept iff reach_rate > 0.0, i.e. at least 1/N_V0 v0 samples
reaches goal). Deliberately not a higher bar: anything with ANY signal of
solvability is legitimate difficulty, and trimming it would inflate the
eventual sweep's solve rates rather than fixing a real corruption. See
ROADMAP.md's Step B note for the empirical reach-rate distribution that
justified this exact cut point.

DEFENSIVE EXCLUSION (separate from the reach-rate threshold): scanning the
FULL pool (not just a 50-instance sample) surfaced one HARD trial that still
hit NavEnv's no-skipped-cells assertion after Step A's dt fix. Root-caused:
NOT a cell-width/dt-margin issue (the single-substep displacement there was
0.0155, unremarkable) -- the straight-line path between two ordinary substep
positions clipped through a razor-thin sliver of a THIRD cell at a
near-degenerate Voronoi vertex (three cells meeting at an extremely acute
angle). This is a geometric property of the Voronoi generator itself (Step
2b, out of scope here) that no finite, globally-safe dt can fully rule out --
tightening dt only reduces, never eliminates, the odds of clipping an
arbitrarily thin wedge. Any instance whose reachability check raises this
assertion (on ANY v0 sample) is therefore excluded from the pool outright,
regardless of its other samples' outcomes -- tracked as a distinct
`excluded_unsafe` category, never silently folded into "dead" or "kept".
"""
from typing import List, Tuple

import numpy as np

from pool import Instance, build_partition, generate_pool_with_stats, train_test_split
from closed_loop_oracle import closed_loop_oracle

N_V0_DEFAULT = 10
REACHABILITY_THRESHOLD = 0.0


def closed_loop_reach_rate(partition, instance: Instance, n_v0: int = N_V0_DEFAULT):
    """Returns the reach rate in [0,1], or None if any v0 sample raised a
    NavEnv dynamics assertion (see module docstring's DEFENSIVE EXCLUSION
    note) -- None is the caller's signal to exclude the instance outright,
    not to average the assertion in as a failure."""
    n_reached = 0
    for seed in range(n_v0):
        try:
            reached, _steps, _trajectory, _outcome = closed_loop_oracle(partition, instance, seed)
        except AssertionError:
            return None
        n_reached += int(reached)
    return n_reached / n_v0


def filter_pool_by_reachability(
    pool: List[Instance], tier: str, threshold: float = REACHABILITY_THRESHOLD, n_v0: int = N_V0_DEFAULT,
) -> Tuple[List[Instance], List[Tuple[Instance, float]], List[Instance]]:
    """Returns (kept, removed, excluded_unsafe):
    - removed: [(instance, reach_rate)] for every instance at or below `threshold`.
    - excluded_unsafe: instances where a dynamics assertion fired (see module
      docstring) -- excluded regardless of `threshold`, tracked separately.
    """
    partition_cache = {}

    def get_partition(seed):
        if seed not in partition_cache:
            partition_cache[seed] = build_partition(tier, seed)
        return partition_cache[seed]

    kept, removed, excluded_unsafe = [], [], []
    for inst in pool:
        partition = get_partition(inst.partition_seed)
        rate = closed_loop_reach_rate(partition, inst, n_v0=n_v0)
        if rate is None:
            excluded_unsafe.append(inst)
        elif rate > threshold:
            kept.append(inst)
        else:
            removed.append((inst, rate))
    return kept, removed, excluded_unsafe


def generate_reachable_pool(
    tier: str, target_n: int, rng_seed: int, threshold: float = REACHABILITY_THRESHOLD,
    n_v0: int = N_V0_DEFAULT, growth_factor: float = 1.3, max_rounds: int = 6,
):
    """Strict-solvable pool generation (pool.py, unchanged), then closed-loop
    reachability filtering. If filtering drops the pool below `target_n`,
    regenerates with a larger raw target (same rng_seed -- generate_pool's
    RNG stream is a deterministic superset as n_instances grows, so this
    isn't restarting from scratch each round) and re-filters, until the
    reachable pool meets target_n or max_rounds is exhausted.

    Returns (kept_pool, removed, excluded_unsafe, gen_stats, raw_n_used).
    """
    n = target_n
    for _round in range(max_rounds):
        pool, stats = generate_pool_with_stats(tier, n, rng_seed=rng_seed)
        kept, removed, excluded_unsafe = filter_pool_by_reachability(pool, tier, threshold=threshold, n_v0=n_v0)
        if len(kept) >= target_n:
            return kept, removed, excluded_unsafe, stats, n
        n = int(n * growth_factor) + 1
    raise RuntimeError(
        f"{tier}: only reached {len(kept)}/{target_n} reachable instances after {max_rounds} regeneration "
        f"rounds (raw n={n})"
    )
