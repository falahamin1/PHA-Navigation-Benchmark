"""Step 6 Gate 0-CL, Part 2 tests: run directly with `python test_reachability_filter.py`.

Note on style: rather than hardcoding a specific (start, goal, hazards)
tuple as "the dead instance" or "the partial instance", these tests
DISCOVER a live example of each from the actual generated pool at test
time. Two earlier tests this benchmark build (test_closed_loop_oracle.py,
Step 6 Gate 0-CL) had to be fixed for exactly this reason -- a hardcoded
instance/cell/seed baked in an assumption about a specific dt/discretization
or pool-generation snapshot that later changed. Discovering examples
dynamically keeps these tests correct regardless of future pool/dt changes.

- test_filter_removes_dead_instance: a genuine 0%-closed-loop-reach instance
  (discovered from the EASY pool, where the earlier Gate 0-CL sweep already
  found one) is rejected by filter_pool_by_reachability.
- test_filter_keeps_partial_reach_instance: a genuine partial-reach (0% <
  rate < 100%) instance is admitted, not over-trimmed.
- test_filter_excludes_dynamics_unsafe_instance: the one real instance found
  (full-pool HARD scan) where a razor-thin Voronoi-vertex wedge triggers
  NavEnv's no-skipped-cells assertion is excluded outright, tracked
  separately from "dead" (0% reach) -- not silently averaged in.
- test_pool_guarantees_preserved: >=120 test instances/tier after filtering,
  train pool adequate, partition-seed train/test disjointness intact,
  determinism across repeated generate_reachable_pool calls.
- test_every_admitted_instance_clears_threshold: every instance in a
  filtered pool has closed-loop reach rate strictly above the threshold.
"""
import numpy as np

from pool import Instance, build_partition, generate_pool_with_stats, train_test_split
from reachability_filter import (
    REACHABILITY_THRESHOLD,
    closed_loop_reach_rate,
    filter_pool_by_reachability,
    generate_reachable_pool,
)

TIERS = ("easy", "medium", "hard")
_SMALL_POOL_SIZES = {"easy": 120, "medium": 120, "hard": 120}


def _find_instance_with_rate(tier, predicate, max_scan=60, n_v0=10, pool_seed=42):
    """Scan up to max_scan instances from `tier`'s generated pool for one
    whose closed_loop_reach_rate satisfies `predicate`. Returns (instance,
    partition, rate) or (None, None, None) if none found in the scan budget."""
    pool, _stats = generate_pool_with_stats(tier, _SMALL_POOL_SIZES[tier], rng_seed=pool_seed)
    partition_cache = {}
    for inst in pool[:max_scan]:
        if inst.partition_seed not in partition_cache:
            partition_cache[inst.partition_seed] = build_partition(tier, inst.partition_seed)
        partition = partition_cache[inst.partition_seed]
        rate = closed_loop_reach_rate(partition, inst, n_v0=n_v0)
        if predicate(rate):
            return inst, partition, rate
    return None, None, None


def test_filter_removes_dead_instance():
    # EASY's small pool reliably contains a 0%-reach instance (the exact
    # dead instance Gate 0-CL flagged, start=10/goal=24/hazard=(19,), shows
    # up deterministically for this seed) -- but we search rather than
    # hardcode it.
    inst, partition, rate = _find_instance_with_rate("easy", lambda r: r == 0.0, max_scan=120)
    assert inst is not None, "expected to find at least one 0%-reach instance in the EASY pool scan"
    kept, removed, excluded_unsafe = filter_pool_by_reachability([inst], "easy", threshold=REACHABILITY_THRESHOLD)
    assert len(kept) == 0 and len(removed) == 1 and len(excluded_unsafe) == 0, (
        f"expected the 0%-reach instance to be removed (not excluded-unsafe), got kept={kept} removed={removed} "
        f"excluded_unsafe={excluded_unsafe}"
    )
    assert removed[0][1] == 0.0
    print(f"test_filter_removes_dead_instance: PASS (found start={inst.start_cell} goal={inst.goal_cell} "
          f"hazards={inst.hazard_cells}, reach_rate=0.0, correctly rejected)")


def test_filter_keeps_partial_reach_instance():
    inst, partition, rate = _find_instance_with_rate(
        "hard", lambda r: 0.0 < r < 1.0, max_scan=120
    )
    assert inst is not None, "expected to find at least one partial-reach (0<rate<1) instance in the HARD pool scan"
    kept, removed, excluded_unsafe = filter_pool_by_reachability([inst], "hard", threshold=REACHABILITY_THRESHOLD)
    assert len(kept) == 1 and len(removed) == 0 and len(excluded_unsafe) == 0, (
        f"expected the partial-reach instance (rate={rate}) to be KEPT, not trimmed -- over-trimming would "
        "rig the sweep toward easy instances"
    )
    print(f"test_filter_keeps_partial_reach_instance: PASS (found start={inst.start_cell} goal={inst.goal_cell} "
          f"hazards={inst.hazard_cells}, reach_rate={rate:.2f}, correctly kept)")


def test_filter_excludes_dynamics_unsafe_instance():
    # The exact instance found by the Step B full-pool scan: a razor-thin
    # Voronoi-vertex wedge triggers NavEnv's no-skipped-cells assertion on
    # one v0 sample, unrelated to cell width/dt margin (see module
    # docstring's DEFENSIVE EXCLUSION note). Must be excluded outright, not
    # folded into "removed" (dead) or averaged into a reach rate.
    inst = Instance(tier="hard", partition_seed=1067354876, start_cell=3, goal_cell=17,
                     hazard_cells=(29,), initial_velocity_sign=(1, 1))
    kept, removed, excluded_unsafe = filter_pool_by_reachability([inst], "hard", threshold=REACHABILITY_THRESHOLD)
    assert len(kept) == 0 and len(removed) == 0 and len(excluded_unsafe) == 1, (
        f"expected the dynamics-unsafe instance to be excluded (not kept/removed), got kept={kept} "
        f"removed={removed} excluded_unsafe={excluded_unsafe}"
    )
    print("test_filter_excludes_dynamics_unsafe_instance: PASS (known razor-thin-wedge instance correctly "
          "excluded as dynamics-unsafe, not silently averaged into a reach rate)")


def test_every_admitted_instance_clears_threshold():
    for tier in TIERS:
        pool, _stats = generate_pool_with_stats(tier, _SMALL_POOL_SIZES[tier], rng_seed=42)
        kept, removed, excluded_unsafe = filter_pool_by_reachability(pool, tier, threshold=REACHABILITY_THRESHOLD)
        partition_cache = {}
        for inst in kept:
            if inst.partition_seed not in partition_cache:
                partition_cache[inst.partition_seed] = build_partition(tier, inst.partition_seed)
            rate = closed_loop_reach_rate(partition_cache[inst.partition_seed], inst)
            assert rate is not None and rate > REACHABILITY_THRESHOLD, (
                f"{tier}: admitted instance {inst} has reach_rate={rate}, expected > threshold "
                f"{REACHABILITY_THRESHOLD}"
            )
        print(f"test_every_admitted_instance_clears_threshold[{tier}]: PASS ({len(kept)}/{len(pool)} admitted, "
              f"all strictly above threshold, {len(excluded_unsafe)} excluded_unsafe)")


def test_pool_guarantees_preserved():
    for tier in TIERS:
        kept, removed, excluded_unsafe, stats, raw_n = generate_reachable_pool(tier, 120, rng_seed=42)
        assert len(kept) >= 120, f"{tier}: generate_reachable_pool returned {len(kept)} < requested 120"
        train, test = train_test_split(kept, test_fraction=0.2, rng_seed=7)
        assert len(test) >= 20, f"{tier}: test split implausibly small ({len(test)}) for a pool of {len(kept)}"
        assert len(train) >= 80, f"{tier}: train split implausibly small ({len(train)}) for a pool of {len(kept)}"
        assert set(train).isdisjoint(set(test))
        if tier != "easy":
            train_seeds = {i.partition_seed for i in train}
            test_seeds = {i.partition_seed for i in test}
            assert train_seeds.isdisjoint(test_seeds), f"{tier}: train/test partition seeds overlap"

        kept2, removed2, excluded_unsafe2, stats2, raw_n2 = generate_reachable_pool(tier, 120, rng_seed=42)
        assert kept == kept2 and raw_n == raw_n2, f"{tier}: generate_reachable_pool not deterministic"

        print(f"test_pool_guarantees_preserved[{tier}]: PASS (raw_n={raw_n}, kept={len(kept)}, "
              f"removed={len(removed)}, excluded_unsafe={len(excluded_unsafe)}, train={len(train)}, "
              f"test={len(test)}, deterministic across 2 calls)")


if __name__ == "__main__":
    test_filter_removes_dead_instance()
    test_filter_keeps_partial_reach_instance()
    test_filter_excludes_dynamics_unsafe_instance()
    test_every_admitted_instance_clears_threshold()
    test_pool_guarantees_preserved()
    print("Step 6 Gate 0-CL Part 2 (reachability filter) tests: ALL PASS")
