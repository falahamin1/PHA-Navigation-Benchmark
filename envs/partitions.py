"""Pluggable cell-partition interface. Step 1 implements only BoxGridPartition
(EASY tier); MEDIUM/HARD (mixed-convex, irregular-convex) partitions slot in
later as additional Partition subclasses without NavEnv needing to change --
that's the point of this interface (see ROADMAP.md Gate 4).
"""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import Voronoi

from geometry import (
    clip_convex_polygon,
    ensure_ccw,
    point_in_convex_polygon,
    polygon_centroid,
    polygons_share_boundary,
    simplify_polygon,
)


class Partition(ABC):
    """A convex cell partition of a rectangular domain.

    Subclasses own the geometry; NavEnv only ever calls this interface.
    """

    num_cells: int
    domain: Tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)

    @abstractmethod
    def locate(self, pos: np.ndarray) -> int:
        """Index of the cell containing `pos` (assumed already inside domain)."""

    @abstractmethod
    def cell_centroid(self, idx: int) -> np.ndarray:
        """Centroid of cell `idx`, used for the potential-based reward shaping."""

    @abstractmethod
    def neighbors_adjacent(self, idx_a: int, idx_b: int) -> bool:
        """True if idx_a/idx_b are the same cell or share a boundary.

        Used to catch a physics substep whose displacement was large enough to
        jump clean over an entire cell without ever registering it -- see the
        no-skipped-cells invariant in NavEnv.step() and test_nav_env.py.
        """

    @abstractmethod
    def cell_facet_count(self, idx: int) -> int:
        """Number of facets (edges) of cell `idx`.

        Added in Step 2a, after BoxGridPartition's Step 1 gate had already
        passed -- flagged rather than silent, since it extends the interface
        that gate signed off on. Purely additive: BoxGridPartition's
        implementation is a constant 4, its existing behavior and tests are
        unchanged. Needed so incidence_variability() (incidence.py) can
        compute nu generically over any Partition subclass, box or mixed.
        """


class BoxGridPartition(Partition):
    """Standard m x n unit-box grid (EASY tier). Cell index = row * num_cols + col,
    row = floor(y), col = floor(x), both clamped to the grid extent so points
    exactly on the outer domain boundary resolve to the last row/column.
    """

    def __init__(self, num_rows=5, num_cols=5, domain=(0.0, 5.0, 0.0, 5.0)):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.domain = domain
        self.num_cells = num_rows * num_cols

        xmin, xmax, ymin, ymax = domain
        self.cell_width = (xmax - xmin) / num_cols
        self.cell_height = (ymax - ymin) / num_rows

    def cell_row_col(self, idx: int):
        return divmod(idx, self.num_cols)

    def cell_index(self, row: int, col: int) -> int:
        return row * self.num_cols + col

    def locate(self, pos: np.ndarray) -> int:
        x, y = pos[0], pos[1]
        xmin, _, ymin, _ = self.domain
        col = int(np.floor((x - xmin) / self.cell_width))
        row = int(np.floor((y - ymin) / self.cell_height))
        col = min(max(col, 0), self.num_cols - 1)
        row = min(max(row, 0), self.num_rows - 1)
        return self.cell_index(row, col)

    def cell_centroid(self, idx: int) -> np.ndarray:
        row, col = self.cell_row_col(idx)
        xmin, _, ymin, _ = self.domain
        cx = xmin + (col + 0.5) * self.cell_width
        cy = ymin + (row + 0.5) * self.cell_height
        return np.array([cx, cy])

    def cell_bounds(self, idx: int):
        """(x_lo, x_hi, y_lo, y_hi) -- exposed for tests and future encoders."""
        row, col = self.cell_row_col(idx)
        xmin, _, ymin, _ = self.domain
        x_lo = xmin + col * self.cell_width
        y_lo = ymin + row * self.cell_height
        return x_lo, x_lo + self.cell_width, y_lo, y_lo + self.cell_height

    def neighbors_adjacent(self, idx_a: int, idx_b: int) -> bool:
        ra, ca = self.cell_row_col(idx_a)
        rb, cb = self.cell_row_col(idx_b)
        return abs(ra - rb) <= 1 and abs(ca - cb) <= 1

    def cell_facet_count(self, idx: int) -> int:
        return 4


# --- MEDIUM tier: mixed convex partition (Step 2a) -------------------------

# Local template vertices for a unit cell, CCW: bottom-left, bottom-right,
# top-right, top-left. Every template is built by cutting this square.
_UNIT_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _split_square(rng) -> List[np.ndarray]:
    return [_UNIT_CORNERS.copy()]


def _split_diag(rng) -> List[np.ndarray]:
    c = _UNIT_CORNERS
    if rng.random() < 0.5:
        return [c[[0, 1, 2]], c[[0, 2, 3]]]
    return [c[[1, 2, 3]], c[[1, 3, 0]]]


def _split_trapezoid(rng) -> List[np.ndarray]:
    x_bot = rng.uniform(0.2, 0.8)
    x_top = rng.uniform(0.2, 0.8)
    left = np.array([[0.0, 0.0], [x_bot, 0.0], [x_top, 1.0], [0.0, 1.0]])
    right = np.array([[x_bot, 0.0], [1.0, 0.0], [1.0, 1.0], [x_top, 1.0]])
    return [left, right]


def _split_corner(rng) -> List[np.ndarray]:
    c = _UNIT_CORNERS
    k = int(rng.integers(0, 4))
    a = rng.uniform(0.25, 0.45)
    k1, k2, k3 = (k + 1) % 4, (k + 2) % 4, (k + 3) % 4
    p_next = c[k] + a * (c[k1] - c[k])
    p_prev = c[k] + a * (c[k3] - c[k])
    triangle = np.array([p_prev, c[k], p_next])
    pentagon = np.array([p_next, c[k1], c[k2], c[k3], p_prev])
    return [triangle, pentagon]


_TEMPLATES = {
    "square": _split_square,
    "diag": _split_diag,
    "trap": _split_trapezoid,
    "corner": _split_corner,
}


class MixedConvexPartition(Partition):
    """MEDIUM tier (SPEC.md 2.2): each base grid cell is independently cut
    into 1-2 convex sub-cells via a randomly chosen template (plain square,
    diagonal split into 2 triangles, slanted split into 2 trapezoids, or a
    corner cut into a pentagon + triangle). grid_seed makes the whole
    partition reproducible. Default template weights land the total cell
    count around ~30-34 for a 5x5 base grid (target: 25-35, SPEC.md 2.2).
    """

    def __init__(
        self,
        grid_seed: int,
        num_rows: int = 5,
        num_cols: int = 5,
        domain: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        p_square: float = 0.7,
        p_diag: float = 0.1,
        p_trap: float = 0.1,
        p_corner: float = 0.1,
    ):
        assert abs(p_square + p_diag + p_trap + p_corner - 1.0) < 1e-9
        self.grid_seed = grid_seed
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.domain = domain

        rng = np.random.default_rng(grid_seed)
        xmin, xmax, ymin, ymax = domain
        self._cell_width = (xmax - xmin) / num_cols
        self._cell_height = (ymax - ymin) / num_rows
        scale = np.array([self._cell_width, self._cell_height])

        cell_verts: List[np.ndarray] = []
        base_cell_members: Dict[Tuple[int, int], List[int]] = {}

        template_names = list(_TEMPLATES.keys())
        template_probs = [p_square, p_diag, p_trap, p_corner]

        for row in range(num_rows):
            for col in range(num_cols):
                origin = np.array([xmin + col * self._cell_width, ymin + row * self._cell_height])
                template_name = rng.choice(template_names, p=template_probs)
                polys_local = _TEMPLATES[template_name](rng)
                for verts_local in polys_local:
                    verts_global = ensure_ccw(origin + verts_local * scale)
                    idx = len(cell_verts)
                    cell_verts.append(verts_global)
                    base_cell_members.setdefault((row, col), []).append(idx)

        self._cell_verts = cell_verts
        self.num_cells = len(cell_verts)
        self._base_cell_members = base_cell_members
        self._facet_counts = [len(v) for v in cell_verts]
        self._centroids = [polygon_centroid(v) for v in cell_verts]
        self._adjacent_pairs = self._compute_adjacency()

    def _compute_adjacency(self):
        pairs = set()
        n = self.num_cells
        for i in range(n):
            for j in range(i + 1, n):
                if polygons_share_boundary(self._cell_verts[i], self._cell_verts[j]):
                    pairs.add(frozenset((i, j)))
        return pairs

    def cell_vertices(self, idx: int) -> np.ndarray:
        return self._cell_verts[idx]

    def locate(self, pos: np.ndarray) -> int:
        x, y = pos[0], pos[1]
        xmin, _, ymin, _ = self.domain
        col = int(np.floor((x - xmin) / self._cell_width))
        row = int(np.floor((y - ymin) / self._cell_height))
        col = min(max(col, 0), self.num_cols - 1)
        row = min(max(row, 0), self.num_rows - 1)

        point = np.array([x, y])
        for idx in self._base_cell_members[(row, col)]:
            if point_in_convex_polygon(self._cell_verts[idx], point):
                return idx
        raise RuntimeError(f"point {pos} in base cell ({row},{col}) matched no sub-cell polygon")

    def cell_centroid(self, idx: int) -> np.ndarray:
        return self._centroids[idx]

    def cell_facet_count(self, idx: int) -> int:
        return self._facet_counts[idx]

    def neighbors_adjacent(self, idx_a: int, idx_b: int) -> bool:
        if idx_a == idx_b:
            return True
        return frozenset((idx_a, idx_b)) in self._adjacent_pairs


# --- HARD tier: irregular convex (Voronoi) partition (Step 2b) -------------

# Rejection-sampling parameters for seed points, tuned empirically (see
# ROADMAP.md Step 2b note) against a min-cell-area sliver guard: with
# min_sep=0.35 and boundary_margin=0.2, 30 rejection-sampled seeds over
# [0,5]x[0,5] produced zero cells below area 0.23 across 30 independent
# seeds (average cell area is 25/30 ~ 0.83, so that floor is comfortably
# non-degenerate) -- and zero non-convex or degenerate (duplicate/collinear
# vertex) cells. Prevention at the sampling stage, not post-hoc
# discard/merge: merging two convex Voronoi cells isn't reliably convex, and
# discarding a sliver would break exact tiling, so keeping seeds apart in the
# first place was the more robust fix.
_HARD_MIN_SEED_SEPARATION = 0.35
_HARD_BOUNDARY_MARGIN = 0.2
_HARD_RAY_RADIUS = 20.0  # far enough past the domain (diagonal ~7.07) that ray-clipping is exact
MIN_CELL_AREA_THRESHOLD = 0.05  # ~6% of the ~0.83 average cell area; degeneracy/sliver guard


def _sample_seed_points(rng, n, domain, min_sep, boundary_margin, max_attempts=20000):
    xmin, xmax, ymin, ymax = domain
    pts = []
    for _ in range(max_attempts):
        if len(pts) >= n:
            break
        x = rng.uniform(xmin + boundary_margin, xmax - boundary_margin)
        y = rng.uniform(ymin + boundary_margin, ymax - boundary_margin)
        p = np.array([x, y])
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            pts.append(p)
    if len(pts) < n:
        raise RuntimeError(
            f"only placed {len(pts)}/{n} seed points within {max_attempts} attempts -- "
            "min_sep/boundary_margin too aggressive for this many points"
        )
    return np.array(pts)


def _voronoi_finite_polygons(vor: Voronoi, radius: float):
    """Reconstruct every Voronoi region as a finite (but possibly very large,
    pre-domain-clip) polygon, including the unbounded regions on the convex
    hull. Standard technique: each open ridge is closed by extending a ray
    from its one finite vertex, in the direction away from the point pair's
    midpoint, out to `radius` -- comfortably past the domain so the
    subsequent clip to [0,5]x[0,5] is exact regardless of the ray's exact
    length. Returns (list of vertex-index lists, vertex array), one region
    per input point in `vor.points` order.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    all_ridges = defaultdict(list)
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges[p1].append((p2, v1, v2))
        all_ridges[p2].append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in all_ridges[p1]:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # a finite ridge, already included above
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        verts = np.asarray([new_vertices[v] for v in new_region])
        c = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - c[1], verts[:, 0] - c[0])
        new_region = [new_region[i] for i in np.argsort(angles)]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


class IrregularConvexPartition(Partition):
    """HARD tier (SPEC.md 2.2): ~30 rejection-sampled seed points, Voronoi
    diagram, every cell clipped to [0,5]x[0,5]. Voronoi cells clipped to a
    convex domain are convex by construction. grid_seed makes the whole
    partition (seed points, resulting cells, hazard-eligible tagging)
    reproducible.

    Hazard tagging here is deliberately minimal: `hazard_eligible_cells`
    exposes which cell indices *could* be hazards (spread across distinct
    facet counts where possible), for Step 2c's instance pool to consume.
    No goal/hazard reward or termination logic lives here -- that belongs to
    NavEnv/the instance pool, not the geometry layer.
    """

    def __init__(
        self,
        grid_seed: int,
        num_seed_points: int = 30,
        domain: Tuple[float, float, float, float] = (0.0, 5.0, 0.0, 5.0),
        min_seed_separation: float = _HARD_MIN_SEED_SEPARATION,
        boundary_margin: float = _HARD_BOUNDARY_MARGIN,
        num_hazards: int = None,
    ):
        self.grid_seed = grid_seed
        self.domain = domain

        rng = np.random.default_rng(grid_seed)
        xmin, xmax, ymin, ymax = domain
        domain_box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

        seed_points = _sample_seed_points(rng, num_seed_points, domain, min_seed_separation, boundary_margin)
        vor = Voronoi(seed_points)
        regions, vor_vertices = _voronoi_finite_polygons(vor, radius=_HARD_RAY_RADIUS)

        cell_verts: List[np.ndarray] = []
        for region in regions:
            poly = ensure_ccw(vor_vertices[region])
            clipped = simplify_polygon(clip_convex_polygon(poly, domain_box))
            if len(clipped) < 3:
                raise RuntimeError(
                    "a Voronoi cell clipped to an empty/degenerate polygon -- a seed point likely "
                    "landed outside the domain or min_seed_separation/boundary_margin need retuning"
                )
            cell_verts.append(clipped)

        self._cell_verts = cell_verts
        self.num_cells = len(cell_verts)
        self._facet_counts = [len(v) for v in cell_verts]
        self._centroids = [polygon_centroid(v) for v in cell_verts]
        self._adjacent_pairs = self._compute_adjacency()

        if num_hazards is None:
            num_hazards = int(rng.integers(2, 5))  # 2-4 inclusive
        self.hazard_eligible_cells = self._select_hazard_cells(rng, num_hazards)

    def _compute_adjacency(self):
        pairs = set()
        n = self.num_cells
        for i in range(n):
            for j in range(i + 1, n):
                if polygons_share_boundary(self._cell_verts[i], self._cell_verts[j]):
                    pairs.add(frozenset((i, j)))
        return pairs

    def _select_hazard_cells(self, rng, num_hazards) -> Tuple[int, ...]:
        """Pick `num_hazards` cells, preferring one per distinct facet count
        before repeating a facet count, so hazard-eligible cells vary in
        shape where the partition's facet-count diversity allows it."""
        facet_to_cells = defaultdict(list)
        for idx, fc in enumerate(self._facet_counts):
            facet_to_cells[fc].append(idx)
        distinct_facet_counts = list(facet_to_cells.keys())
        rng.shuffle(distinct_facet_counts)

        chosen = []
        group = 0
        while len(chosen) < num_hazards and group < num_hazards * len(distinct_facet_counts):
            fc = distinct_facet_counts[group % len(distinct_facet_counts)]
            candidates = [c for c in facet_to_cells[fc] if c not in chosen]
            if candidates:
                chosen.append(candidates[int(rng.integers(0, len(candidates)))])
            group += 1
        return tuple(chosen)

    def cell_vertices(self, idx: int) -> np.ndarray:
        return self._cell_verts[idx]

    def locate(self, pos: np.ndarray) -> int:
        point = np.asarray(pos, dtype=float)
        for idx, verts in enumerate(self._cell_verts):
            if point_in_convex_polygon(verts, point):
                return idx
        raise RuntimeError(f"point {pos} matched no cell -- possible gap in the partition")

    def cell_centroid(self, idx: int) -> np.ndarray:
        return self._centroids[idx]

    def cell_facet_count(self, idx: int) -> int:
        return self._facet_counts[idx]

    def neighbors_adjacent(self, idx_a: int, idx_b: int) -> bool:
        if idx_a == idx_b:
            return True
        return frozenset((idx_a, idx_b)) in self._adjacent_pairs
