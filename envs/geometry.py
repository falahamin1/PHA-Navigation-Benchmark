"""Pure convex-polygon utilities shared by every Partition implementation
(and by test_partitions_medium.py's overlap check). Polygons are Nx2 float
arrays of vertices; functions that require an orientation assume CCW (use
ensure_ccw first).
"""

import numpy as np


def signed_area(verts: np.ndarray) -> float:
    x, y = verts[:, 0], verts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    return 0.5 * np.sum(x * y1 - x1 * y)


def polygon_area(verts: np.ndarray) -> float:
    return abs(signed_area(verts))


def ensure_ccw(verts: np.ndarray) -> np.ndarray:
    return verts[::-1].copy() if signed_area(verts) < 0 else verts


def polygon_centroid(verts: np.ndarray) -> np.ndarray:
    x, y = verts[:, 0], verts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = 0.5 * np.sum(cross)
    cx = np.sum((x + x1) * cross) / (6 * a)
    cy = np.sum((y + y1) * cross) / (6 * a)
    return np.array([cx, cy])


def is_convex(verts: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff every interior turn along the vertex loop has the same sign
    (all left turns or all right turns) -- orientation-agnostic."""
    n = len(verts)
    signs = []
    for i in range(n):
        a, b, c = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        v1, v2 = b - a, c - b
        signs.append(v1[0] * v2[1] - v1[1] * v2[0])
    signs = np.array(signs)
    return bool(np.all(signs >= -tol) or np.all(signs <= tol))


def point_in_convex_polygon(verts: np.ndarray, point: np.ndarray, tol: float = 1e-9) -> bool:
    """Inclusive point-in-polygon test; `verts` must be CCW. Points on an
    edge count as inside."""
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        edge = b - a
        rel = point - a
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < -tol:
            return False
    return True


def segment_overlap_length(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, tol: float = 1e-9) -> float:
    """Length of the overlap between collinear segments (a1,a2) and (b1,b2);
    0 if they aren't collinear or don't overlap."""
    d = a2 - a1
    norm = np.linalg.norm(d)
    if norm < tol:
        return 0.0
    d_unit = d / norm

    def perp_dist(p):
        rel = p - a1
        return abs(rel[0] * d_unit[1] - rel[1] * d_unit[0])

    if perp_dist(b1) > tol or perp_dist(b2) > tol:
        return 0.0

    ta1, ta2 = 0.0, norm
    tb1 = np.dot(b1 - a1, d_unit)
    tb2 = np.dot(b2 - a1, d_unit)
    lo = max(ta1, min(tb1, tb2))
    hi = min(ta2, max(tb1, tb2))
    return max(0.0, hi - lo)


def _point_on_segment(pt: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
    ab = b - a
    length = np.linalg.norm(ab)
    if length < tol:
        return np.linalg.norm(pt - a) <= tol
    ap = pt - a
    cross = ab[0] * ap[1] - ab[1] * ap[0]
    if abs(cross) > tol * length:
        return False
    t = (ap @ ab) / (length * length)
    return -tol <= t <= 1 + tol


def polygons_share_boundary(verts_a: np.ndarray, verts_b: np.ndarray, tol: float = 1e-7) -> bool:
    """True iff A and B touch anywhere on their boundaries: a shared edge
    (length > tol, including partial/T-junction overlap), OR just a single
    shared corner point (e.g. two cells meeting only at a diagonal grid
    corner). Both count as "adjacent" for the no-skipped-cells check in
    NavEnv -- BoxGridPartition's Chebyshev-distance neighbors_adjacent already
    treats diagonal corner-touching as legitimate, and MixedConvexPartition
    must agree or a genuine one-tick corner crossing would be misflagged as a
    skip. (Found via cross-checking: every seed produces at least one pair of
    diagonally-touching base-grid cells whose sub-cells share only the grid
    corner point, not an edge.)
    """
    na, nb = len(verts_a), len(verts_b)
    for i in range(na):
        a1, a2 = verts_a[i], verts_a[(i + 1) % na]
        for j in range(nb):
            b1, b2 = verts_b[j], verts_b[(j + 1) % nb]
            if segment_overlap_length(a1, a2, b1, b2) > tol:
                return True
    for i in range(na):
        for j in range(nb):
            b1, b2 = verts_b[j], verts_b[(j + 1) % nb]
            if _point_on_segment(verts_a[i], b1, b2, tol):
                return True
    for j in range(nb):
        for i in range(na):
            a1, a2 = verts_a[i], verts_a[(i + 1) % na]
            if _point_on_segment(verts_b[j], a1, a2, tol):
                return True
    return False


def _line_intersect(p1, p2, a, b):
    d1, d2 = p2 - p1, b - a
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-14:
        return p1  # parallel/degenerate; not expected for well-formed convex polygons
    t = ((a[0] - p1[0]) * d2[1] - (a[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def clip_convex_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersection of two convex CCW polygons via Sutherland-Hodgman clipping,
    returning the actual clipped polygon's vertices (shape (0,2) if empty).

    Added in Step 2b: `convex_polygon_intersection_area` (2a) only needed the
    intersection's *area*, but IrregularConvexPartition needs the actual
    clipped *vertices* (each Voronoi cell polygon clipped to the [0,5]x[0,5]
    domain box). Refactored out of the old inline Sutherland-Hodgman loop so
    both use one implementation.
    """

    def inside(p, edge_a, edge_b):
        edge = edge_b - edge_a
        rel = p - edge_a
        return edge[0] * rel[1] - edge[1] * rel[0] >= -1e-12

    def clip_against_edge(poly, edge_a, edge_b):
        if len(poly) == 0:
            return poly
        output = []
        n = len(poly)
        for i in range(n):
            cur, prev = poly[i], poly[i - 1]
            cur_in, prev_in = inside(cur, edge_a, edge_b), inside(prev, edge_a, edge_b)
            if cur_in:
                if not prev_in:
                    output.append(_line_intersect(prev, cur, edge_a, edge_b))
                output.append(cur)
            elif prev_in:
                output.append(_line_intersect(prev, cur, edge_a, edge_b))
        return np.array(output) if output else np.zeros((0, 2))

    output = subject
    m = len(clip)
    for i in range(m):
        if len(output) == 0:
            break
        output = clip_against_edge(output, clip[i], clip[(i + 1) % m])
    return output if len(output) > 0 else np.zeros((0, 2))


def convex_polygon_intersection_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Area of the intersection of two convex (CCW) polygons. Used to verify
    cells don't overlap (shared-edge touching gives ~0 intersection area;
    genuine overlap doesn't)."""
    output = clip_convex_polygon(subject, clip)
    if len(output) < 3:
        return 0.0
    return polygon_area(output)


def simplify_polygon(verts: np.ndarray, dup_tol: float = 1e-9, collinear_tol: float = 1e-9) -> np.ndarray:
    """Remove consecutive duplicate vertices and collinear middle vertices.

    Added in Step 2b: Sutherland-Hodgman clipping against an axis-aligned
    domain box routinely introduces near-duplicate vertices (a clip point
    landing right on top of an existing one) and collinear triples (a clip
    edge parallel to an existing polygon edge). These are exactly the
    degenerate facets flagged as a downstream risk (malformed
    constraint-incidence graph, one-substep cell skips) -- this is the
    generic cleanup step applied after every Voronoi-cell clip.
    """
    v = np.asarray(verts, dtype=float)

    changed = True
    while changed and len(v) > 2:
        changed = False
        # Drop consecutive duplicates.
        n = len(v)
        keep = [i for i in range(n) if np.linalg.norm(v[i] - v[i - 1]) > dup_tol]
        if len(keep) < n:
            v = v[keep]
            changed = True
            continue
        # Drop collinear middles.
        n = len(v)
        keep = []
        for i in range(n):
            a, b, c = v[i - 1], v[i], v[(i + 1) % n]
            e1, e2 = b - a, c - b
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            scale = max(np.linalg.norm(e1), np.linalg.norm(e2), 1.0)
            if abs(cross) > collinear_tol * scale:
                keep.append(i)
        if len(keep) < n and len(keep) >= 3:
            v = v[keep]
            changed = True
    return v
