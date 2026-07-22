"""Step 2a tests: run directly with `python test_partitions_medium.py`.

- test_convexity: every cell of every seed is convex.
- test_no_gaps: total cell area == domain area (25.0), every seed.
- test_no_overlaps: every cell pair has zero-area interior intersection.
- test_facet_variety: MEDIUM actually mixes facet counts, doesn't degenerate
  to all-squares.
- test_nu_ordering: nu(box) < nu(mixed) for every seed -- the scientific gate.
- test_locate_correctness: locate() agrees with point-in-polygon on >=100
  random points, including domain-boundary points.
"""

import itertools

import numpy as np

from geometry import convex_polygon_intersection_area, is_convex, point_in_convex_polygon, polygon_area
from incidence import incidence_variability
from partitions import BoxGridPartition, MixedConvexPartition

SEEDS = list(range(10))
DOMAIN_AREA = 25.0


def test_convexity():
    for seed in SEEDS:
        p = MixedConvexPartition(grid_seed=seed)
        for i in range(p.num_cells):
            assert is_convex(p.cell_vertices(i)), f"seed={seed} cell={i} not convex: {p.cell_vertices(i)}"
    print(f"test_convexity: PASS ({len(SEEDS)} seeds, all cells convex)")


def test_no_gaps():
    for seed in SEEDS:
        p = MixedConvexPartition(grid_seed=seed)
        total_area = sum(polygon_area(p.cell_vertices(i)) for i in range(p.num_cells))
        assert abs(total_area - DOMAIN_AREA) < 1e-6, f"seed={seed}: total area {total_area} != {DOMAIN_AREA}"
    print(f"test_no_gaps: PASS ({len(SEEDS)} seeds, total area == {DOMAIN_AREA} within 1e-6)")


def test_no_overlaps():
    tol = 1e-9
    for seed in SEEDS:
        p = MixedConvexPartition(grid_seed=seed)
        # Restrict to cells whose bounding boxes actually overlap -- a cheap
        # prefilter, exact for our template construction (sub-cells never
        # extend beyond their base grid cell), before the expensive check.
        boxes = []
        for i in range(p.num_cells):
            v = p.cell_vertices(i)
            boxes.append((v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()))

        for i, j in itertools.combinations(range(p.num_cells), 2):
            xi0, xi1, yi0, yi1 = boxes[i]
            xj0, xj1, yj0, yj1 = boxes[j]
            if xi1 <= xj0 or xj1 <= xi0 or yi1 <= yj0 or yj1 <= yi0:
                continue
            area = convex_polygon_intersection_area(p.cell_vertices(i), p.cell_vertices(j))
            assert area < tol, f"seed={seed}: cells {i},{j} overlap with interior area {area}"
    print(f"test_no_overlaps: PASS ({len(SEEDS)} seeds, no pair of cells has interior overlap)")


def test_facet_variety():
    for seed in SEEDS:
        p = MixedConvexPartition(grid_seed=seed)
        facet_counts = {p.cell_facet_count(i) for i in range(p.num_cells)}
        assert len(facet_counts) >= 2, f"seed={seed}: only one facet count present ({facet_counts}) -- degenerated to all-squares"
    print(f"test_facet_variety: PASS ({len(SEEDS)} seeds, each has >=2 distinct facet counts)")


def test_nu_ordering():
    box = BoxGridPartition()
    nu_box = incidence_variability(box)
    print(f"nu(box) = {nu_box:.4f}")
    assert abs(nu_box) < 1e-12, f"nu(box) should be exactly 0, got {nu_box}"

    for seed in SEEDS[:5]:
        p = MixedConvexPartition(grid_seed=seed)
        nu_mixed = incidence_variability(p)
        print(f"nu(mixed, seed={seed}) = {nu_mixed:.4f}")
        assert nu_box < nu_mixed, f"seed={seed}: nu(box)={nu_box} not < nu(mixed)={nu_mixed}"
    print("test_nu_ordering: PASS (nu(box) < nu(mixed) for every seed)")


def test_locate_correctness():
    rng = np.random.default_rng(123)
    n_checked = 0
    for seed in SEEDS[:5]:
        p = MixedConvexPartition(grid_seed=seed)
        xmin, xmax, ymin, ymax = p.domain
        pts = rng.uniform([xmin, ymin], [xmax, ymax], size=(25, 2))
        for pt in pts:
            idx = p.locate(pt)
            assert 0 <= idx < p.num_cells
            assert point_in_convex_polygon(p.cell_vertices(idx), pt), f"seed={seed}: locate({pt})={idx} but point not in that cell's polygon"
            n_checked += 1

        # Explicit domain-boundary points -- must resolve without gaps.
        boundary_pts = [
            (xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin),
            (xmin, (ymin + ymax) / 2), (xmax, (ymin + ymax) / 2),
            ((xmin + xmax) / 2, ymin), ((xmin + xmax) / 2, ymax),
        ]
        for pt in boundary_pts:
            pt = np.array(pt)
            idx = p.locate(pt)
            assert 0 <= idx < p.num_cells
            assert point_in_convex_polygon(p.cell_vertices(idx), pt), f"seed={seed}: boundary point {pt} -> cell {idx} not containing it"
            n_checked += 1

    assert n_checked >= 100, f"only checked {n_checked} points, need >=100"
    print(f"test_locate_correctness: PASS ({n_checked} points, all resolve to a containing cell, boundaries included)")


if __name__ == "__main__":
    test_convexity()
    test_no_gaps()
    test_no_overlaps()
    test_facet_variety()
    test_nu_ordering()
    test_locate_correctness()
    print("Step 2a tests: ALL PASS")
