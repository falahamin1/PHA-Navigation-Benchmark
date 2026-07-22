"""Step 2c: instance pools -- (partition, start/goal/hazard config) tuples,
generated per tier, filtered for solvability, deduplicated, and split into
disjoint train/test sets.

Scope note: this module only builds and filters instance *configurations*.
It does not touch NavEnv, does not run any RL, and does not construct the
Step 3 cell-adjacency graph -- see ROADMAP.md.

STEP 2C-FIX (Gate 0 finding -- see ROADMAP.md): the original solvability BFS
ran over `Partition.neighbors_adjacent`, the permissive transition predicate
(corner-only point touches count as adjacent). Gate 0's dynamics-executability
sweep proved this over-certifies: a scripted BFS-optimal path through the real
env reached the goal 100% of the time when every hop was facet-adjacent, but
only ~5% of the time when the path relied on a corner-only hop (confirmed on
EASY; MEDIUM/HARD showed the same failure-mode profile) -- the continuous
dynamics has no facet to cross at a point-touch, so it drifts into whichever
edge-neighbor the approach velocity happens to favor.

CRITICAL PREDICATE ASYMMETRY -- do not collapse these two notions:
- NavEnv's mode-switching (nav_env.py) stays on the PERMISSIVE predicate
  (`Partition.neighbors_adjacent`), UNCHANGED. If the object physically drifts
  into a corner-neighbor, the env must still recognize that transition -- it's
  tracking a real object, not planning a path. This module never touches
  nav_env.py.
- The solvability filter below (`solve_path`/`is_solvable*`) now runs over
  STRICT facet-sharing adjacency instead: a plannable path may only rely on
  facet crossings, since that's the only transition the dynamics can reliably
  execute. This is `strict_neighbors_adjacent`, which reuses -- does not
  reimplement -- the exact Step 3a inter-cell predicate via
  `region_graph.build_region_graph(...).inter_cell_pairs()` (positive-length
  shared boundary, opposing normals; see region_graph.py's module docstring).
  Purely geometric, so it works identically on box/mixed/Voronoi cells with no
  row/col grid concept needed.
The asymmetry is the point: the object CAN end up in a corner-neighbor
(permissive, env-level), but a plan CANNOT rely on it doing so (strict,
filter-level). `solve_path_permissive` below preserves the old behavior only
for the regression test proving this distinction (test_solvability_fix.py).
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from partitions import BoxGridPartition, IrregularConvexPartition, MixedConvexPartition
from region_graph import build_region_graph

TIERS = ("easy", "medium", "hard")


def build_partition(tier: str, partition_seed: int):
    """The one place that maps (tier, seed) -> a Partition instance. EASY has
    no seed axis (BoxGridPartition is always the same grid); partition_seed is
    ignored for it and conventionally recorded as 0 on EASY instances."""
    if tier == "easy":
        return BoxGridPartition()
    if tier == "medium":
        return MixedConvexPartition(grid_seed=partition_seed)
    if tier == "hard":
        return IrregularConvexPartition(grid_seed=partition_seed)
    raise ValueError(f"unknown tier {tier!r}")


def _hazard_candidates(tier: str, partition) -> List[int]:
    """Cells eligible to be tagged as hazards for an instance on this partition.
    HARD partitions pre-tag a small hazard-eligible subset (Step 2b); EASY/MEDIUM
    have no such concept, so any cell is a candidate (excluded from start/goal
    by the caller)."""
    if tier == "hard":
        return list(partition.hazard_eligible_cells)
    return list(range(partition.num_cells))


@dataclass(frozen=True)
class Instance:
    """A fully specified episode configuration on top of a partition.

    `partition_seed` + `tier` reproducibly identify the partition (via
    `build_partition`) rather than embedding the partition object itself --
    partitions hold numpy arrays and aren't cheaply hashable/comparable, and
    Instance needs to be both (frozen dataclass with only str/int/tuple
    fields gets `__eq__`/`__hash__` for free, which is exactly what
    train/test disjointness and duplicate-rejection need).

    `hazard_cells` is always stored sorted, so two instances naming the same
    hazard *set* in a different order compare equal and hash the same.

    `initial_velocity_sign` is metadata only at this stage (Step 2c does not
    touch NavEnv/reward/termination): a discrete (sign_vx, sign_vy) in
    {-1,+1}^2 that a later training script would use to bias the env's v0
    sampling bounds. Not consumed by anything yet.
    """

    tier: str
    partition_seed: int
    start_cell: int
    goal_cell: int
    hazard_cells: Tuple[int, ...]
    initial_velocity_sign: Tuple[int, int]


def make_instance(tier, partition_seed, start_cell, goal_cell, hazard_cells, initial_velocity_sign) -> Instance:
    return Instance(
        tier=tier,
        partition_seed=partition_seed,
        start_cell=start_cell,
        goal_cell=goal_cell,
        hazard_cells=tuple(sorted(hazard_cells)),
        initial_velocity_sign=tuple(initial_velocity_sign),
    )


# --- Solvability -------------------------------------------------------

# (partition class, grid_seed, num_cells) -> set of (min(a,b), max(a,b))
# strict facet-sharing pairs, computed once per distinct partition (topology
# doesn't depend on any instance's goal/hazard assignment) and reused across
# every candidate instance generated on that partition. Keyed by content, NOT
# id(partition): pool generation builds and discards ~140 MEDIUM/HARD
# partition objects per tier, and a garbage-collected object's id() can be
# reused by a later, geometrically-different object -- an id()-keyed cache
# silently returned a stale/wrong pairs set for such a collision (caught by
# test_solvability_fix.py's cross-check before this fix).
_strict_pairs_cache: Dict[tuple, set] = {}


def _strict_adjacent_pairs(partition):
    key = (type(partition).__name__, getattr(partition, "grid_seed", None), partition.num_cells)
    pairs = _strict_pairs_cache.get(key)
    if pairs is None:
        # goal_cell/hazard_cells only feed per-node *features* in
        # build_region_graph, never the inter-cell edge computation itself --
        # a placeholder instance is safe here, this is a pure topology query.
        placeholder = Instance(tier="_strict_adjacency_probe", partition_seed=0, start_cell=0,
                                goal_cell=-1, hazard_cells=(), initial_velocity_sign=(1, 1))
        graph = build_region_graph(partition, placeholder)
        pairs = graph.inter_cell_pairs()
        _strict_pairs_cache[key] = pairs
    return pairs


def strict_neighbors_adjacent(partition, idx_a: int, idx_b: int) -> bool:
    """STRICT facet-sharing adjacency (Step 3a's inter-cell predicate, reused
    via build_region_graph/inter_cell_pairs -- not reimplemented here). This
    is deliberately NOT `Partition.neighbors_adjacent` (the permissive,
    corner-inclusive transition predicate NavEnv's mode-switching still uses
    unchanged) -- see this module's docstring for the Step 2c-fix asymmetry."""
    if idx_a == idx_b:
        return True
    a, b = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
    return (a, b) in _strict_adjacent_pairs(partition)


def _bfs_path(partition, start_cell: int, goal_cell: int, hazard_cells, adjacency_fn) -> Optional[List[int]]:
    hazard_set = set(hazard_cells)
    if start_cell == goal_cell:
        return None
    if start_cell in hazard_set or goal_cell in hazard_set:
        return None

    parent = {start_cell: None}
    queue = deque([start_cell])
    while queue:
        cur = queue.popleft()
        if cur == goal_cell:
            path = []
            node = goal_cell
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        for nxt in range(partition.num_cells):
            if nxt in parent or nxt in hazard_set:
                continue
            if adjacency_fn(partition, cur, nxt):
                parent[nxt] = cur
                queue.append(nxt)
    return None


def solve_path(partition, start_cell: int, goal_cell: int, hazard_cells) -> Optional[List[int]]:
    """BFS shortest path start_cell -> goal_cell over STRICT facet-sharing
    adjacency (Step 2c-fix), with hazard cells removed from the graph
    entirely.

    A plannable path may only rely on facet crossings: the continuous
    dynamics cannot reliably thread a corner-only (point) touch -- see this
    module's docstring and ROADMAP.md's Gate 0 finding. This is NOT
    `partition.neighbors_adjacent` (the permissive predicate NavEnv's mode
    switching still uses, unchanged) -- see `solve_path_permissive` for that.

    Returns the path (list of cell indices, start..goal inclusive) if
    solvable, else None. start_cell == goal_cell, or either being a hazard,
    is never solvable regardless of connectivity.
    """
    return _bfs_path(partition, start_cell, goal_cell, hazard_cells, strict_neighbors_adjacent)


def solve_path_permissive(partition, start_cell: int, goal_cell: int, hazard_cells) -> Optional[List[int]]:
    """The PRE-FIX solvability BFS, over the permissive transition predicate
    (`partition.neighbors_adjacent`, corner-touches included). Kept only so
    test_solvability_fix.py can demonstrate the exact asymmetry the fix
    introduces (a corner-only-hop instance is now solve_path-unsolvable but
    remains solve_path_permissive-solvable) -- not used by pool generation."""
    return _bfs_path(partition, start_cell, goal_cell, hazard_cells,
                      lambda p, a, b: p.neighbors_adjacent(a, b))


def is_solvable_on_partition(partition, start_cell: int, goal_cell: int, hazard_cells) -> bool:
    return solve_path(partition, start_cell, goal_cell, hazard_cells) is not None


def is_solvable(instance: Instance) -> bool:
    """Instance-level convenience wrapper (rebuilds the partition from
    tier+partition_seed). Generation itself uses `is_solvable_on_partition`
    directly against an already-built partition object, to avoid rebuilding
    a MEDIUM/HARD partition on every candidate instance."""
    partition = build_partition(instance.tier, instance.partition_seed)
    return is_solvable_on_partition(partition, instance.start_cell, instance.goal_cell, instance.hazard_cells)


def solve_path_for_instance(instance: Instance) -> Optional[List[int]]:
    partition = build_partition(instance.tier, instance.partition_seed)
    return solve_path(partition, instance.start_cell, instance.goal_cell, instance.hazard_cells)


# --- Pool generation -----------------------------------------------------

# Instances sampled per partition before moving to a fresh partition seed
# (MEDIUM/HARD only -- EASY has a single fixed partition). Kept modest
# (rather than, say, 20+ instances on a handful of partitions) precisely so
# that a downstream train/test split at partition-seed granularity has many
# distinct seeds to divide between train and test, not just a few.
INSTANCES_PER_PARTITION = 5


def _sample_raw_instance(rng, tier, partition, partition_seed) -> Instance:
    n = partition.num_cells
    start_cell = int(rng.integers(0, n))
    goal_cell = int(rng.integers(0, n))

    candidates = [c for c in _hazard_candidates(tier, partition) if c not in (start_cell, goal_cell)]
    if candidates:
        max_hazards = min(4, len(candidates))
        n_hazards = int(rng.integers(1, max_hazards + 1))
        rng.shuffle(candidates)
        hazard_cells = candidates[:n_hazards]
    else:
        hazard_cells = []

    sign = tuple(int(s) for s in rng.choice([-1, 1], size=2))
    return make_instance(tier, partition_seed, start_cell, goal_cell, hazard_cells, sign)


def _generate_pool_impl(tier: str, n_instances: int, rng_seed: int, max_attempts_multiplier: int = 40):
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    rng = np.random.default_rng(rng_seed)
    pool: List[Instance] = []
    seen = set()
    stats = {
        "tier": tier,
        "generated_raw": 0,
        "rejected_unsolvable": 0,
        "rejected_duplicate": 0,
        "accepted": 0,
        "partition_seeds_used": 0,
        "max_attempts_hit": False,
    }
    max_total_attempts = n_instances * max_attempts_multiplier

    if tier == "easy":
        partition = build_partition(tier, 0)
        stats["partition_seeds_used"] = 1
        while len(pool) < n_instances and stats["generated_raw"] < max_total_attempts:
            inst = _sample_raw_instance(rng, tier, partition, 0)
            stats["generated_raw"] += 1
            if not is_solvable_on_partition(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells):
                stats["rejected_unsolvable"] += 1
                continue
            if inst in seen:
                stats["rejected_duplicate"] += 1
                continue
            seen.add(inst)
            pool.append(inst)
        stats["accepted"] = len(pool)
        if stats["generated_raw"] >= max_total_attempts and len(pool) < n_instances:
            stats["max_attempts_hit"] = True
        return pool, stats

    # MEDIUM / HARD: rotate through fresh partition seeds.
    while len(pool) < n_instances and stats["generated_raw"] < max_total_attempts:
        partition_seed = int(rng.integers(0, 2**31 - 1))
        partition = build_partition(tier, partition_seed)
        stats["partition_seeds_used"] += 1
        accepted_this_partition = 0
        attempts_this_partition = 0
        budget_this_partition = INSTANCES_PER_PARTITION * max_attempts_multiplier
        while (
            accepted_this_partition < INSTANCES_PER_PARTITION
            and len(pool) < n_instances
            and attempts_this_partition < budget_this_partition
        ):
            inst = _sample_raw_instance(rng, tier, partition, partition_seed)
            stats["generated_raw"] += 1
            attempts_this_partition += 1
            if not is_solvable_on_partition(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells):
                stats["rejected_unsolvable"] += 1
                continue
            if inst in seen:
                stats["rejected_duplicate"] += 1
                continue
            seen.add(inst)
            pool.append(inst)
            accepted_this_partition += 1

    stats["accepted"] = len(pool)
    if len(pool) < n_instances:
        stats["max_attempts_hit"] = True
    return pool, stats


def generate_pool(tier: str, n_instances: int, rng_seed: int) -> List[Instance]:
    pool, _stats = _generate_pool_impl(tier, n_instances, rng_seed)
    return pool


def generate_pool_with_stats(tier: str, n_instances: int, rng_seed: int) -> Tuple[List[Instance], Dict]:
    """Same as generate_pool, but also returns generation stats (raw/rejected/
    accepted counts, distinct partition seeds used) for the gate diagnostic."""
    return _generate_pool_impl(tier, n_instances, rng_seed)


# --- Train/test split ------------------------------------------------------

def train_test_split(pool: List[Instance], test_fraction: float, rng_seed: int):
    """Disjoint split. For MEDIUM/HARD, split at partition-seed granularity
    (every instance sharing a partition_seed goes entirely to one side) so a
    held-out evaluation is zero-shot on unseen geometry, not just unseen
    start/goal on seen geometry. For EASY there is only one partition, so
    that constraint is vacuous -- split on instance config directly, and
    "generalization" for EASY means unseen start/goal/hazard configs, not
    unseen geometry.
    """
    if not pool:
        return [], []
    tiers = {inst.tier for inst in pool}
    assert len(tiers) == 1, f"train_test_split expects a single-tier pool, got {tiers}"
    tier = next(iter(tiers))
    rng = np.random.default_rng(rng_seed)

    if tier == "easy":
        order = np.arange(len(pool))
        rng.shuffle(order)
        n_test = int(round(len(pool) * test_fraction))
        test_positions = set(order[:n_test].tolist())
        train = [inst for i, inst in enumerate(pool) if i not in test_positions]
        test = [inst for i, inst in enumerate(pool) if i in test_positions]
        return train, test

    seeds = sorted({inst.partition_seed for inst in pool})
    seeds_arr = np.array(seeds)
    rng.shuffle(seeds_arr)
    n_test_seeds = max(1, int(round(len(seeds_arr) * test_fraction)))
    test_seeds = set(seeds_arr[:n_test_seeds].tolist())
    train = [inst for inst in pool if inst.partition_seed not in test_seeds]
    test = [inst for inst in pool if inst.partition_seed in test_seeds]
    return train, test
