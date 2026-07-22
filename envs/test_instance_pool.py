"""Step 2c tests: run directly with `python test_instance_pool.py`.

- test_solvable_positive: a hand-constructed clear route is solvable.
- test_solvable_negative: hazards walling off the goal, start/goal being a
  hazard, and start == goal are all correctly unsolvable.
- test_corner_touch_not_facet_share: STEP 2C-FIX UPDATE -- solvability now
  uses STRICT facet-sharing adjacency, not permissive transition-adjacency
  (see pool.py's Step 2c-fix docstring / ROADMAP.md Gate 0 finding). This test
  used to lock in the OLD (buggy) behavior -- a corner-hop-only instance being
  certified solvable -- and has been inverted to lock in the FIX: the same
  corner-only instance is now correctly unsolvable, while
  `solve_path_permissive` (kept only for this comparison) still finds the old
  corner-hop route, proving the permissive/strict asymmetry precisely.
- test_all_pooled_instances_solvable: every instance in a >=200-instance pool
  (per tier) is solvable (now under the strict filter).
- test_determinism: generate_pool(tier, n, seed) is deterministic.
- test_train_test_disjoint: no shared instances; for MEDIUM/HARD, no shared
  partition seeds either.
- test_pool_size: >=120 test instances per tier after the split.
- test_no_duplicates: no duplicate instances within a generated pool.
"""

import numpy as np

from geometry import segment_overlap_length
from partitions import BoxGridPartition, MixedConvexPartition
from pool import (
    Instance,
    build_partition,
    generate_pool,
    generate_pool_with_stats,
    is_solvable,
    is_solvable_on_partition,
    make_instance,
    solve_path,
    solve_path_permissive,
    train_test_split,
)

TIERS = ("easy", "medium", "hard")

# Production-scale pools, generated once and reused across the tests that
# need real scale (solvability-over-all, disjointness, pool-size,
# duplicate-check) -- see ROADMAP.md Step 2c note on how these sizes were
# tuned (140 distinct partition seeds for MEDIUM/HARD -> 28 held out at
# test_fraction=0.2, comfortably clearing the >=120-test-instance target
# with a healthy spread of distinct test geometries, not just 1-2 seeds).
_POOL_SIZES = {"easy": 800, "medium": 700, "hard": 700}
_POOL_CACHE = {}
_STATS_CACHE = {}


def _get_pool(tier):
    if tier not in _POOL_CACHE:
        pool, stats = generate_pool_with_stats(tier, _POOL_SIZES[tier], rng_seed=42)
        _POOL_CACHE[tier] = pool
        _STATS_CACHE[tier] = stats
    return _POOL_CACHE[tier]


def test_solvable_positive():
    partition = BoxGridPartition(num_rows=5, num_cols=5)
    path = solve_path(partition, start_cell=0, goal_cell=6, hazard_cells=())
    assert path is not None, "expected a clear route from cell 0 to cell 6 to be solvable"
    assert path[0] == 0 and path[-1] == 6
    assert is_solvable_on_partition(partition, 0, 6, ())
    print(f"test_solvable_positive: PASS (path {path})")


def test_solvable_negative():
    partition = BoxGridPartition(num_rows=5, num_cols=5)

    # Hazards wall off the goal: all of row 2 blocks rows {0,1} from {3,4}
    # under BoxGridPartition's Chebyshev-<=1 adjacency (max reachable row
    # jump is 1, so row 2 fully intact is required to cross it).
    wall = tuple(partition.cell_index(2, c) for c in range(5))
    start, goal = partition.cell_index(0, 0), partition.cell_index(4, 4)
    assert not is_solvable_on_partition(partition, start, goal, wall), "wall should block all routes"

    # Start itself is a hazard.
    assert not is_solvable_on_partition(partition, start, goal, (start,))
    # Goal itself is a hazard.
    assert not is_solvable_on_partition(partition, start, goal, (goal,))
    # start == goal.
    assert not is_solvable_on_partition(partition, start, start, ())

    print("test_solvable_negative: PASS (walled-off goal, start-is-hazard, goal-is-hazard, "
          "start==goal all correctly unsolvable)")


def test_corner_touch_not_facet_share():
    # MixedConvexPartition(seed=0): cells 0 (base grid square (0,0)) and 9
    # (base grid square (1,1)) touch only at the shared grid corner (1,1) --
    # verified below as zero positive-length shared edge, i.e. genuinely
    # corner-only, not a facet-sharing pair.
    partition = MixedConvexPartition(grid_seed=0)
    v0, v9 = partition.cell_vertices(0), partition.cell_vertices(9)
    max_overlap = 0.0
    for i in range(len(v0)):
        a1, a2 = v0[i], v0[(i + 1) % len(v0)]
        for j in range(len(v9)):
            b1, b2 = v9[j], v9[(j + 1) % len(v9)]
            max_overlap = max(max_overlap, segment_overlap_length(a1, a2, b1, b2))
    assert max_overlap < 1e-9, f"expected cells 0,9 to touch only at a point, got shared-edge length {max_overlap}"

    # They ARE still transition-adjacent (the permissive predicate NavEnv's
    # mode-switching uses, unchanged by the Step 2c-fix)...
    assert partition.neighbors_adjacent(0, 9), "expected cells 0,9 to be transition-adjacent via the corner"

    # ...but solvability (Step 2c-fix) now runs over STRICT facet-sharing
    # adjacency, so the direct corner hop is no longer, by itself, a valid
    # planned route: with every other cell hazarded out (isolating the corner
    # touch as the *only* remaining connection), the old permissive BFS still
    # finds it, but the fixed strict BFS correctly reports unsolvable.
    all_other_cells = tuple(c for c in range(partition.num_cells) if c not in (0, 9))
    permissive_path = solve_path_permissive(partition, start_cell=0, goal_cell=9, hazard_cells=all_other_cells)
    assert permissive_path == [0, 9], (
        f"expected solve_path_permissive to still find the direct corner-hop path [0, 9], got {permissive_path}"
    )
    strict_path = solve_path(partition, start_cell=0, goal_cell=9, hazard_cells=all_other_cells)
    assert strict_path is None, (
        f"FAIL: solve_path (strict, Step 2c-fix) certified a corner-hop-only route as solvable: {strict_path} "
        "-- this is precisely the bug Gate 0 found"
    )

    # With the direct facet-adjacent neighbors available (no hazards), a
    # genuine facet-only route still exists -- the fix rejects the corner
    # SHORTCUT, not reachability itself.
    routed_path = solve_path(partition, start_cell=0, goal_cell=9, hazard_cells=())
    assert routed_path is not None and routed_path[0] == 0 and routed_path[-1] == 9, (
        f"expected a facet-only route around the corner to still exist, got {routed_path}"
    )
    assert 9 not in routed_path[1:-1] or len(routed_path) == 2, "sanity: path should be a real route, not a stub"

    print("test_corner_touch_not_facet_share: PASS (corner-only touch has zero shared-edge length; "
          "still transition-adjacent for NavEnv, but solve_path now correctly rejects it as an "
          f"isolated route [was solvable pre-fix, is unsolvable post-fix] and instead finds a "
          f"facet-only detour {routed_path} when one exists)")


def test_all_pooled_instances_solvable():
    for tier in TIERS:
        pool = _get_pool(tier)
        assert len(pool) >= 200, f"{tier}: pool too small ({len(pool)}) for this test"

        # Efficient bulk check: group by partition_seed, build each partition once.
        by_seed = {}
        for inst in pool:
            by_seed.setdefault((inst.tier, inst.partition_seed), []).append(inst)
        for (t, seed), instances in by_seed.items():
            partition = build_partition(t, seed)
            for inst in instances:
                assert is_solvable_on_partition(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells), (
                    f"{tier}: unsolvable instance survived generation: {inst}"
                )

        # Spot-check the instance-level convenience wrapper too (literal spec signature).
        for inst in pool[:5]:
            assert is_solvable(inst)

        print(f"test_all_pooled_instances_solvable[{tier}]: PASS ({len(pool)} instances, all solvable)")


def test_determinism():
    for tier in TIERS:
        pool_a = generate_pool(tier, 60, rng_seed=123)
        pool_b = generate_pool(tier, 60, rng_seed=123)
        assert pool_a == pool_b, f"{tier}: two generate_pool calls with the same seed produced different pools"
    print(f"test_determinism: PASS ({len(TIERS)} tiers, identical pools across repeated calls)")


def test_train_test_disjoint():
    for tier in TIERS:
        pool = _get_pool(tier)
        train, test = train_test_split(pool, test_fraction=0.2, rng_seed=7)

        assert set(train).isdisjoint(set(test)), f"{tier}: train/test share instances"

        if tier != "easy":
            train_seeds = {i.partition_seed for i in train}
            test_seeds = {i.partition_seed for i in test}
            assert train_seeds.isdisjoint(test_seeds), (
                f"{tier}: train/test share partition seeds {train_seeds & test_seeds} -- "
                "zero-shot generalization claim would be false"
            )
            print(f"test_train_test_disjoint[{tier}]: PASS (instances disjoint; "
                  f"{len(train_seeds)} train / {len(test_seeds)} test partition seeds, no overlap)")
        else:
            print(f"test_train_test_disjoint[{tier}]: PASS (instances disjoint; "
                  "single shared partition by design, split is instance-level only)")


def test_pool_size():
    for tier in TIERS:
        pool = _get_pool(tier)
        _train, test = train_test_split(pool, test_fraction=0.2, rng_seed=7)
        assert len(test) >= 120, f"{tier}: only {len(test)} test instances, need >=120"
        print(f"test_pool_size[{tier}]: PASS ({len(test)} test instances)")


def test_no_duplicates():
    for tier in TIERS:
        pool = _get_pool(tier)
        assert len(pool) == len(set(pool)), f"{tier}: duplicate instances found in pool"
        print(f"test_no_duplicates[{tier}]: PASS ({len(pool)} instances, all unique)")


if __name__ == "__main__":
    test_solvable_positive()
    test_solvable_negative()
    test_corner_touch_not_facet_share()
    test_all_pooled_instances_solvable()
    test_determinism()
    test_train_test_disjoint()
    test_pool_size()
    test_no_duplicates()
    print("Step 2c tests: ALL PASS")
