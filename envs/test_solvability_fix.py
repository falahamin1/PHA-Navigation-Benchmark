"""Step 2c-fix tests: run directly with `python test_solvability_fix.py`.

Context: Gate 0's dynamics-executability sweep found the pre-fix solvability
BFS (over the permissive `neighbors_adjacent` predicate) certifies
corner-hop-only paths as solvable, but those execute at ~5% under real
dynamics (vs 100% for facet-only paths). The fix switches the solvability
filter to STRICT facet-sharing adjacency (reusing region_graph.py's Step 3a
inter-cell predicate), while NavEnv's mode-switching stays on the permissive
predicate unchanged -- see pool.py's module docstring for the full rationale.

- test_strict_filter_rejects_corner_hop_only_route / test_facet_route_still_solvable:
  the load-bearing predicate-asymmetry tests -- a corner-hop-ISOLATED instance
  flips from solvable (pre-fix) to unsolvable (post-fix), while a genuine
  facet-sharing route remains solvable throughout.
- test_navenv_still_permissive: the regression guard -- NavEnv's mode
  switching must still use the permissive predicate unchanged (re-runs the
  existing Step 1/2a no-skip/corner-crossing test, plus a direct static check
  that the corner-touch pair is still transition-adjacent).
- test_regenerated_pools_all_strict_solvable: every instance in the
  re-filtered pools is solvable under the new strict predicate.
- test_pool_size_and_disjointness: >=120 test instances/tier, adequate train
  pools, partition-seed train/test disjointness preserved (2c regression).
- test_regeneration_determinism: regeneration with a fixed seed reproduces
  identically.
"""

import numpy as np

from partitions import MixedConvexPartition
from pool import (
    build_partition,
    generate_pool,
    generate_pool_with_stats,
    is_solvable_on_partition,
    solve_path,
    solve_path_permissive,
    strict_neighbors_adjacent,
    train_test_split,
)

TIERS = ("easy", "medium", "hard")
_POOL_SIZES = {"easy": 800, "medium": 700, "hard": 700}
_POOL_CACHE = {}
_STATS_CACHE = {}


def _get_pool(tier):
    if tier not in _POOL_CACHE:
        pool, stats = generate_pool_with_stats(tier, _POOL_SIZES[tier], rng_seed=42)
        _POOL_CACHE[tier] = pool
        _STATS_CACHE[tier] = stats
    return _POOL_CACHE[tier]


def test_strict_filter_rejects_corner_hop_only_route():
    # The exact pair proven in Step 2c / re-verified in Step 3a: cells 0, 9 of
    # MixedConvexPartition(seed=0) touch only at a single grid corner.
    partition = MixedConvexPartition(grid_seed=0)
    assert partition.neighbors_adjacent(0, 9), "precondition: cells 0,9 must be transition-adjacent via the corner"
    assert not strict_neighbors_adjacent(partition, 0, 9), (
        "precondition: cells 0,9 must NOT be strict-facet-adjacent (corner-only touch)"
    )

    # Isolate the corner touch as the ONLY remaining connection by hazarding
    # every other cell -- this is "an instance whose only start->goal route
    # goes through that corner-touch."
    all_other_cells = tuple(c for c in range(partition.num_cells) if c not in (0, 9))

    old_path = solve_path_permissive(partition, start_cell=0, goal_cell=9, hazard_cells=all_other_cells)
    assert old_path == [0, 9], f"expected the pre-fix permissive BFS to still find [0, 9], got {old_path}"

    new_path = solve_path(partition, start_cell=0, goal_cell=9, hazard_cells=all_other_cells)
    assert new_path is None, (
        f"FAIL: strict solve_path certified a corner-hop-ISOLATED instance as solvable: {new_path}"
    )
    assert not is_solvable_on_partition(partition, 0, 9, all_other_cells), (
        "is_solvable_on_partition must agree with solve_path returning None"
    )

    print("test_strict_filter_rejects_corner_hop_only_route: PASS (corner-hop-only instance is now "
          "correctly UNSOLVABLE under the strict filter, was solvable pre-fix)")


def test_facet_route_still_solvable():
    partition = MixedConvexPartition(grid_seed=0)
    # A route with no reliance on the corner touch (default, no hazards):
    # a genuine facet-only path must still be found.
    path = solve_path(partition, start_cell=0, goal_cell=9, hazard_cells=())
    assert path is not None, "expected a facet-only detour around the corner to exist"
    assert path[0] == 0 and path[-1] == 9
    for a, b in zip(path, path[1:]):
        assert strict_neighbors_adjacent(partition, a, b), (
            f"path hop {a}->{b} is not strict-facet-adjacent -- solve_path returned a non-strict route"
        )
    print(f"test_facet_route_still_solvable: PASS (facet-only detour {path} found and verified hop-by-hop)")


def test_navenv_still_permissive():
    # Static guard: the corner-touch pair remains transition-adjacent -- if a
    # future change accidentally swapped NavEnv onto the strict predicate,
    # this exact pair would flip and this assertion would catch it.
    partition = MixedConvexPartition(grid_seed=0)
    assert partition.neighbors_adjacent(0, 9), (
        "REGRESSION: cells 0,9 should still be transition-adjacent -- NavEnv's mode-switching predicate "
        "must remain permissive"
    )

    # Behavioral guard: re-run the existing Step 1/2a no-skipped-cells test
    # unmodified. It exercises real diagonal (corner-crossing) actions across
    # 20 random episodes plus a hand-verified edge crossing; if this module's
    # fix had touched NavEnv (it didn't), that test's internal
    # `assert partition.neighbors_adjacent(...)` invariant would be the first
    # thing to break.
    from test_nav_env import test_boundary_crossing_and_no_skip
    test_boundary_crossing_and_no_skip()

    print("test_navenv_still_permissive: PASS (corner pair still transition-adjacent; "
          "Step 1/2a no-skip/corner-crossing regression re-passed unmodified)")


def test_regenerated_pools_all_strict_solvable():
    for tier in TIERS:
        pool = _get_pool(tier)
        assert len(pool) >= 200, f"{tier}: pool too small ({len(pool)}) for this test"

        by_seed = {}
        for inst in pool:
            by_seed.setdefault((inst.tier, inst.partition_seed), []).append(inst)
        for (t, seed), instances in by_seed.items():
            partition = build_partition(t, seed)
            for inst in instances:
                assert is_solvable_on_partition(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells), (
                    f"{tier}: strict-unsolvable instance survived the fixed generation filter: {inst}"
                )
                path = solve_path(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells)
                for a, b in zip(path, path[1:]):
                    assert strict_neighbors_adjacent(partition, a, b), (
                        f"{tier}: instance {inst} has a non-strict-adjacent hop {a}->{b} in its certified path"
                    )
        print(f"test_regenerated_pools_all_strict_solvable[{tier}]: PASS ({len(pool)} instances, "
              "all strict-solvable, every certified path hop is facet-adjacent)")


def test_pool_size_and_disjointness():
    for tier in TIERS:
        pool = _get_pool(tier)
        stats = _STATS_CACHE[tier]
        train, test = train_test_split(pool, test_fraction=0.2, rng_seed=7)

        assert len(test) >= 120, f"{tier}: only {len(test)} test instances, need >=120"
        assert len(train) >= 4 * len(test), f"{tier}: train pool ({len(train)}) too small relative to test"
        assert set(train).isdisjoint(set(test)), f"{tier}: train/test share instances"

        if tier != "easy":
            train_seeds = {i.partition_seed for i in train}
            test_seeds = {i.partition_seed for i in test}
            assert train_seeds.isdisjoint(test_seeds), (
                f"{tier}: train/test share partition seeds {train_seeds & test_seeds}"
            )
            print(f"test_pool_size_and_disjointness[{tier}]: PASS (train={len(train)}, test={len(test)}, "
                  f"{stats['rejected_unsolvable']} rejected-unsolvable of {stats['generated_raw']} raw, "
                  f"{len(train_seeds)}/{len(test_seeds)} train/test partition seeds, no overlap)")
        else:
            print(f"test_pool_size_and_disjointness[{tier}]: PASS (train={len(train)}, test={len(test)}, "
                  f"{stats['rejected_unsolvable']} rejected-unsolvable of {stats['generated_raw']} raw, "
                  "single shared partition by design)")


def test_regeneration_determinism():
    for tier in TIERS:
        pool_a = generate_pool(tier, 60, rng_seed=321)
        pool_b = generate_pool(tier, 60, rng_seed=321)
        assert pool_a == pool_b, f"{tier}: two generate_pool calls with the same seed produced different pools"
    print(f"test_regeneration_determinism: PASS ({len(TIERS)} tiers, identical pools across repeated calls)")


if __name__ == "__main__":
    test_strict_filter_rejects_corner_hop_only_route()
    test_facet_route_still_solvable()
    test_navenv_still_permissive()
    test_regenerated_pools_all_strict_solvable()
    test_pool_size_and_disjointness()
    test_regeneration_determinism()
    print("Step 2c-fix tests: ALL PASS")
