"""Step 3a tests: run directly with `python test_region_graph.py`.

- test_single_square / test_single_triangle: isolated hand-verifiable cells,
  intra-cell edges only, forming a k-cycle.
- test_two_edge_adjacent_squares: exactly one inter-cell edge, on the shared
  facet, hand-verifiable.
- test_corner_touch_negative_case: THE carry-forward test -- cells 0,9 of
  MixedConvexPartition(seed=0), proven in Step 2c to be transition-adjacent
  via a corner-only touch (zero shared-edge length). Must get zero inter-cell
  edges.
- test_facet_sharing_subset_of_transition_adjacent: facet-sharing pairs are
  always a subset of transition-adjacent pairs; strictly smaller on MEDIUM
  (grid diagonals guarantee corner-only touches). HARD is printed and
  explained rather than forced, since Voronoi partitions generically have
  zero corner-only touches (3-way vertices) -- see the empirical check in the
  ROADMAP.md note before assuming a tie there is a bug.
- test_node_count_invariant: total nodes == sum of facet counts, including on
  HARD cells reaching 9-10 facets.
- test_well_formedness: no self-loops, no duplicate edges, canonical
  undirected storage.
- test_determinism: identical graphs across independent builds of the same
  partition_seed.
"""

import numpy as np

from partitions import IrregularConvexPartition, MixedConvexPartition
from pool import Instance, build_partition
from region_graph import build_region_graph


class _ListPartition:
    """Minimal standalone Partition-like object wrapping an explicit list of
    convex polygons, for hand-verifiable isolated-cell tests that shouldn't
    depend on any real tier's generator. Implements only the duck-typed
    vertex access build_region_graph needs (cell_vertices + num_cells) --
    not a real Partition subclass, doesn't touch the frozen ABC."""

    def __init__(self, polygons):
        self._polygons = [np.asarray(p, dtype=float) for p in polygons]
        self.num_cells = len(self._polygons)

    def cell_vertices(self, idx):
        return self._polygons[idx]


def _dummy_instance(goal_cell=-1, hazard_cells=()):
    return Instance(
        tier="test", partition_seed=0, start_cell=0, goal_cell=goal_cell,
        hazard_cells=tuple(sorted(hazard_cells)), initial_velocity_sign=(1, 1),
    )


def _edges_by_type(graph, edge_type):
    return [tuple(e) for e, t in zip(graph.edges, graph.edge_type) if t == edge_type]


def test_single_square():
    square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    graph = build_region_graph(_ListPartition([square]), _dummy_instance())

    assert graph.num_nodes == 4, f"expected 4 nodes, got {graph.num_nodes}"
    intra, inter = _edges_by_type(graph, "intra"), _edges_by_type(graph, "inter")
    assert len(intra) == 4, f"expected 4 intra-cell edges (4-cycle), got {len(intra)}"
    assert len(inter) == 0, f"expected 0 inter-cell edges (isolated cell), got {len(inter)}"

    degree = {i: 0 for i in range(4)}
    for i, j in intra:
        degree[i] += 1
        degree[j] += 1
    assert all(d == 2 for d in degree.values()), f"expected a 4-cycle (all degree 2), got {degree}"
    print("test_single_square: PASS (4 nodes, 4 intra edges forming a 4-cycle, 0 inter edges)")


def test_single_triangle():
    triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    graph = build_region_graph(_ListPartition([triangle]), _dummy_instance())

    assert graph.num_nodes == 3
    intra, inter = _edges_by_type(graph, "intra"), _edges_by_type(graph, "inter")
    assert len(intra) == 3, f"expected 3 intra-cell edges (3-cycle), got {len(intra)}"
    assert len(inter) == 0

    degree = {i: 0 for i in range(3)}
    for i, j in intra:
        degree[i] += 1
        degree[j] += 1
    assert all(d == 2 for d in degree.values())
    print("test_single_triangle: PASS (3 nodes, 3 intra edges forming a 3-cycle, 0 inter edges)")


def test_two_edge_adjacent_squares():
    left = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    right = [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
    graph = build_region_graph(_ListPartition([left, right]), _dummy_instance())

    assert graph.num_nodes == 8
    intra, inter = _edges_by_type(graph, "intra"), _edges_by_type(graph, "inter")
    assert len(intra) == 8, f"expected 4+4=8 intra-cell edges, got {len(intra)}"
    assert len(inter) == 1, f"expected exactly 1 inter-cell edge (the shared facet), got {len(inter)}"

    i, j = inter[0]
    assert graph.node_cell[i] != graph.node_cell[j], "inter-cell edge must connect different cells"
    for n in (i, j):
        a, b = graph.node_endpoints[n]
        assert abs(a[0] - 1.0) < 1e-9 and abs(b[0] - 1.0) < 1e-9, (
            f"inter-cell edge should connect the two facets lying on the shared line x=1, "
            f"got endpoints {a},{b} for node {n}"
        )
    print("test_two_edge_adjacent_squares: PASS (8 nodes, 8 intra edges, exactly 1 inter edge on the shared facet)")


def test_corner_touch_negative_case():
    # The exact pair proven in Step 2c: cells 0 and 9 of MixedConvexPartition
    # (seed=0) share zero positive-length boundary (corner-only touch at grid
    # corner (1,1)) but ARE neighbors_adjacent (transition-adjacent). This is
    # THE load-bearing assertion for this step.
    partition = MixedConvexPartition(grid_seed=0)
    assert partition.neighbors_adjacent(0, 9), "precondition from Step 2c broke: cells 0,9 should be transition-adjacent"

    graph = build_region_graph(partition, _dummy_instance())
    inter_pairs = graph.inter_cell_pairs()
    assert (0, 9) not in inter_pairs, (
        "FAIL: cells 0 and 9 (corner-touch only, proven in Step 2c) got an inter-cell edge -- "
        "the graph is connecting cells that share no facet"
    )
    print("test_corner_touch_negative_case: PASS (cells 0,9 are transition-adjacent but correctly got "
          "zero inter-cell edges)")


def _transition_adjacent_pairs(partition):
    n = partition.num_cells
    return {(i, j) for i in range(n) for j in range(i + 1, n) if partition.neighbors_adjacent(i, j)}


def test_facet_sharing_subset_of_transition_adjacent():
    for name, partition in [("MEDIUM", MixedConvexPartition(grid_seed=1)), ("HARD", IrregularConvexPartition(grid_seed=1))]:
        graph = build_region_graph(partition, _dummy_instance())
        facet_pairs = graph.inter_cell_pairs()
        transition_pairs = _transition_adjacent_pairs(partition)

        assert facet_pairs.issubset(transition_pairs), (
            f"{name}: a facet-sharing pair is NOT transition-adjacent -- geometrically impossible, "
            "indicates a bug in the inter-cell edge predicate"
        )
        corner_only = transition_pairs - facet_pairs
        print(f"  {name}: #transition-adjacent={len(transition_pairs)}, #facet-sharing={len(facet_pairs)}, "
              f"#corner-only(no edge)={len(corner_only)}")

        if name == "MEDIUM":
            # Grid-diagonal corner touches are geometrically guaranteed here.
            assert len(corner_only) > 0, "MEDIUM: expected >=1 corner-only pair (grid diagonals), found none"
            assert len(facet_pairs) < len(transition_pairs), (
                "MEDIUM: facet-sharing count should be strictly less than transition-adjacent count -- "
                "a tie here would mean the predicate collapsed"
            )
        else:
            # HARD (Voronoi): verified empirically (50 seeds) to have zero
            # corner-only pairs -- generic Voronoi vertices are 3-way
            # meetings, so every touching pair shares a proper edge. A tie
            # here is the geometrically correct outcome, not a collapsed
            # predicate. Only assert it explicitly when corner-only pairs
            # are actually absent, so a future generator change that DOES
            # introduce them would still be caught by the strict branch.
            if len(corner_only) == 0:
                print(f"  {name}: 0 corner-only pairs is the expected outcome for Voronoi partitions "
                      "(3-way vertices), not a sign of a collapsed predicate -- see ROADMAP.md note")
            else:
                assert len(facet_pairs) < len(transition_pairs)

    print("test_facet_sharing_subset_of_transition_adjacent: PASS")


def test_node_count_invariant():
    for tier, partition in [("MEDIUM", MixedConvexPartition(grid_seed=2)), ("HARD", IrregularConvexPartition(grid_seed=2))]:
        graph = build_region_graph(partition, _dummy_instance())
        expected = sum(partition.cell_facet_count(i) for i in range(partition.num_cells))
        assert graph.num_nodes == expected, f"{tier}: {graph.num_nodes} nodes != sum of facet counts {expected}"
        max_facets = max(partition.cell_facet_count(i) for i in range(partition.num_cells))
        print(f"  {tier}: node count == sum of facet counts ({expected}), max single-cell facet count = {max_facets}")
    print("test_node_count_invariant: PASS")


def test_well_formedness():
    for tier, partition in [("MEDIUM", MixedConvexPartition(grid_seed=3)), ("HARD", IrregularConvexPartition(grid_seed=3))]:
        graph = build_region_graph(partition, _dummy_instance())
        seen = set()
        for i, j in graph.edges:
            assert i != j, f"{tier}: self-loop at node {i}"
            key = (min(int(i), int(j)), max(int(i), int(j)))
            assert key not in seen, f"{tier}: duplicate edge {key}"
            seen.add(key)
        assert all(i < j for i, j in graph.edges), f"{tier}: edges not stored in canonical (i<j) form"
    print("test_well_formedness: PASS (no self-loops, no duplicate edges, canonical undirected storage)")


def test_determinism():
    for tier, seed in [("medium", 4), ("hard", 4)]:
        partition_a, partition_b = build_partition(tier, seed), build_partition(tier, seed)
        graph_a = build_region_graph(partition_a, _dummy_instance())
        graph_b = build_region_graph(partition_b, _dummy_instance())

        assert np.array_equal(graph_a.node_cell, graph_b.node_cell)
        assert np.array_equal(graph_a.node_facet, graph_b.node_facet)
        assert np.allclose(graph_a.node_features, graph_b.node_features)
        assert np.array_equal(graph_a.edges, graph_b.edges)
        assert graph_a.edge_type == graph_b.edge_type
    print("test_determinism: PASS (identical graphs across independent builds of the same partition_seed)")


if __name__ == "__main__":
    test_single_square()
    test_single_triangle()
    test_two_edge_adjacent_squares()
    test_corner_touch_negative_case()
    test_facet_sharing_subset_of_transition_adjacent()
    test_node_count_invariant()
    test_well_formedness()
    test_determinism()
    print("Step 3a tests: ALL PASS")
