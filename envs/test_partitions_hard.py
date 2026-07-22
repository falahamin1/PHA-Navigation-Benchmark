"""Step 2b tests: run directly with `python test_partitions_hard.py`.

- test_convexity: every cell of every seed is convex.
- test_no_gaps: total cell area == domain area (25.0), every seed.
- test_no_overlaps: every cell pair has zero-area interior intersection.
- test_min_area_guard: no cell falls below MIN_CELL_AREA_THRESHOLD -- the
  sliver guard called out explicitly in the Step 2b brief.
- test_facet_variety: HARD has >=3 distinct facet counts (stricter than 2a's
  >=2, since HARD should be more varied than MEDIUM).
- test_nu_ordering: nu(box) < nu(mixed) < nu(hard) per seed pairing -- the
  load-bearing scientific gate. Do NOT loosen this if it fails; a collapsed
  ordering is signal that either the generator needs more facet-count spread,
  or nu itself needs to incorporate adjacency-degree variance. Flag, don't fix
  by relaxing the assertion.
- test_locate_correctness: locate() agrees with point-in-polygon on >=100
  random points, including domain-boundary points.
- test_degeneracy_guard: no cell has <3 vertices, no duplicate/collinear
  vertex triples survive (this is exactly where Voronoi clipping bites).
"""

import numpy as np

from geometry import convex_polygon_intersection_area, is_convex, point_in_convex_polygon, polygon_area
from incidence import incidence_variability
from partitions import MIN_CELL_AREA_THRESHOLD, BoxGridPartition, IrregularConvexPartition, MixedConvexPartition

SEEDS = list(range(10))
DOMAIN_AREA = 25.0


def test_convexity():
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
        for i in range(p.num_cells):
            assert is_convex(p.cell_vertices(i)), f"seed={seed} cell={i} not convex: {p.cell_vertices(i)}"
    print(f"test_convexity: PASS ({len(SEEDS)} seeds, all cells convex)")


def test_no_gaps():
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
        total_area = sum(polygon_area(p.cell_vertices(i)) for i in range(p.num_cells))
        assert abs(total_area - DOMAIN_AREA) < 1e-6, f"seed={seed}: total area {total_area} != {DOMAIN_AREA}"
    print(f"test_no_gaps: PASS ({len(SEEDS)} seeds, total area == {DOMAIN_AREA} within 1e-6)")


def test_no_overlaps():
    import itertools
    tol = 1e-9
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
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


def test_min_area_guard():
    worst = float("inf")
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
        for i in range(p.num_cells):
            area = polygon_area(p.cell_vertices(i))
            worst = min(worst, area)
            assert area >= MIN_CELL_AREA_THRESHOLD, (
                f"seed={seed} cell={i}: area {area} below MIN_CELL_AREA_THRESHOLD="
                f"{MIN_CELL_AREA_THRESHOLD} -- sliver cell got through"
            )
    print(f"test_min_area_guard: PASS ({len(SEEDS)} seeds, smallest cell area = {worst:.5f}, "
          f"threshold = {MIN_CELL_AREA_THRESHOLD})")


def test_facet_variety():
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
        facet_counts = {p.cell_facet_count(i) for i in range(p.num_cells)}
        assert len(facet_counts) >= 3, (
            f"seed={seed}: only {len(facet_counts)} distinct facet count(s) ({facet_counts}) -- "
            "HARD should be more varied than MEDIUM (>=2)"
        )
    print(f"test_facet_variety: PASS ({len(SEEDS)} seeds, each has >=3 distinct facet counts)")


def test_nu_ordering():
    box = BoxGridPartition()
    nu_box = incidence_variability(box)
    assert abs(nu_box) < 1e-12, f"nu(box) should be exactly 0, got {nu_box}"

    print(f"{'seed':>4}  {'nu(box)':>8}  {'nu(mixed)':>10}  {'nu(hard)':>9}  margin(hard-mixed)")
    margins = []
    for seed in SEEDS[:5]:
        nu_mixed = incidence_variability(MixedConvexPartition(grid_seed=seed))
        nu_hard = incidence_variability(IrregularConvexPartition(grid_seed=seed))
        margins.append(nu_hard - nu_mixed)
        print(f"{seed:>4}  {nu_box:>8.4f}  {nu_mixed:>10.4f}  {nu_hard:>9.4f}  {nu_hard - nu_mixed:>+.4f}")
        assert nu_box < nu_mixed, f"seed={seed}: nu(box)={nu_box} not < nu(mixed)={nu_mixed}"
        assert nu_mixed < nu_hard, f"seed={seed}: nu(mixed)={nu_mixed} not < nu(hard)={nu_hard}"

    min_margin = min(margins)
    print(f"test_nu_ordering: PASS (nu(box) < nu(mixed) < nu(hard) for every seed; "
          f"smallest hard-over-mixed margin = {min_margin:+.4f})")
    if min_margin < 0.2:
        print(f"  ** FLAG for review: smallest margin {min_margin:+.4f} is tight, not comfortable -- "
              "per the Step 2b brief, consider whether nu needs adjacency-degree variance "
              "rather than just facet-count entropy before Step 2c.")


def test_locate_correctness():
    rng = np.random.default_rng(123)
    n_checked = 0
    for seed in SEEDS[:5]:
        p = IrregularConvexPartition(grid_seed=seed)
        xmin, xmax, ymin, ymax = p.domain
        pts = rng.uniform([xmin, ymin], [xmax, ymax], size=(25, 2))
        for pt in pts:
            idx = p.locate(pt)
            assert 0 <= idx < p.num_cells
            assert point_in_convex_polygon(p.cell_vertices(idx), pt), (
                f"seed={seed}: locate({pt})={idx} but point not in that cell's polygon"
            )
            n_checked += 1

        boundary_pts = [
            (xmin, ymin), (xmax, ymax), (xmin, ymax), (xmax, ymin),
            (xmin, (ymin + ymax) / 2), (xmax, (ymin + ymax) / 2),
            ((xmin + xmax) / 2, ymin), ((xmin + xmax) / 2, ymax),
        ]
        for pt in boundary_pts:
            pt = np.array(pt)
            idx = p.locate(pt)
            assert 0 <= idx < p.num_cells
            assert point_in_convex_polygon(p.cell_vertices(idx), pt), (
                f"seed={seed}: boundary point {pt} -> cell {idx} not containing it"
            )
            n_checked += 1

    assert n_checked >= 100, f"only checked {n_checked} points, need >=100"
    print(f"test_locate_correctness: PASS ({n_checked} points, all resolve to a containing cell, "
          "boundaries included)")


def test_degeneracy_guard():
    dup_tol = 1e-9
    collinear_tol = 1e-9
    for seed in SEEDS:
        p = IrregularConvexPartition(grid_seed=seed)
        for i in range(p.num_cells):
            v = p.cell_vertices(i)
            n = len(v)
            assert n >= 3, f"seed={seed} cell={i}: only {n} vertices"
            for k in range(n):
                a, b, c = v[k - 1], v[k], v[(k + 1) % n]
                assert np.linalg.norm(b - a) > dup_tol, f"seed={seed} cell={i}: duplicate vertex at {k}"
                e1, e2 = b - a, c - b
                cross = e1[0] * e2[1] - e1[1] * e2[0]
                scale = max(np.linalg.norm(e1), np.linalg.norm(e2), 1.0)
                assert abs(cross) > collinear_tol * scale, (
                    f"seed={seed} cell={i}: collinear vertex triple at {k}"
                )
    print(f"test_degeneracy_guard: PASS ({len(SEEDS)} seeds, no cell with <3 vertices, "
          "no duplicate/collinear vertex triples)")


if __name__ == "__main__":
    test_convexity()
    test_no_gaps()
    test_no_overlaps()
    test_min_area_guard()
    test_facet_variety()
    test_nu_ordering()
    test_locate_correctness()
    test_degeneracy_guard()
    print("Step 2b tests: ALL PASS")
