"""Step 3a: region-level incidence graph construction (DESIGN.md 1.5, paper
Section 3.3 lifted to span the whole polytopic region).

Pure function from (partition, instance) -> RegionGraph. No encoder, no GNN,
no training touches this module -- see ROADMAP.md Step 3a/3b split.

Nodes: one per half-space constraint (facet) of every cell. A k-facet cell
contributes k nodes.

Edges, two kinds, recorded separately (`edge_type`):
- intra-cell (existing Section 3.3 rule): constraints i, j of the SAME cell
  are connected iff some vertex of that cell activates both (lies on both
  facets). For a simple convex polygon this is exactly the cyclic
  neighbor relation (each vertex is the shared endpoint of exactly two
  edges), so a k-facet cell's intra-cell edges form a k-cycle -- but this
  module checks vertex activation directly rather than assuming that
  shortcut, per the literal rule.
- inter-cell (the new contribution, STRICT facet-sharing): constraint i of
  cell P and constraint j of cell Q (P != Q) are connected iff their facet
  segments overlap collinearly with positive length (`segment_overlap_length`
  from geometry.py -- the same primitive the Step 2c corner-touch test used)
  and their outward normals oppose (cells are on opposite sides of the shared
  line). Corner-only touches (zero positive-length overlap) get NO edge here,
  even though they ARE `neighbors_adjacent`-true (the permissive transition
  predicate NavEnv/solvability use) -- that distinction is the entire point
  of this step; see test_region_graph.py's corner-touch negative case.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from geometry import segment_overlap_length

FEATURE_NAMES = ("normal_x", "normal_y", "offset", "is_goal_cell", "is_hazard_cell", "contains_agent")

_VERTEX_ACTIVATION_TOL = 1e-7
_FACET_OVERLAP_EPS = 1e-7


def _get_cell_vertices(partition, idx: int) -> np.ndarray:
    """Duck-typed vertex access across all three Partition tiers -- same
    pattern already used in visualize_instance_pool.py's render_instance,
    since BoxGridPartition exposes cell_bounds (axis-aligned) rather than
    cell_vertices (arbitrary polygon). Not a Partition ABC change."""
    if hasattr(partition, "cell_vertices"):
        return partition.cell_vertices(idx)
    x_lo, x_hi, y_lo, y_hi = partition.cell_bounds(idx)
    return np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])


def _cell_edge_constraints(verts: np.ndarray) -> List[dict]:
    """One half-space constraint per polygon edge: outward unit normal +
    offset, with normal @ x <= offset for interior/boundary points (verts
    must be CCW, as every Partition tier guarantees)."""
    n = len(verts)
    constraints = []
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        edge = b - a
        length = np.linalg.norm(edge)
        normal = np.array([edge[1], -edge[0]]) / length
        offset = float(normal @ a)
        constraints.append({"normal": normal, "offset": offset, "a": a, "b": b})
    return constraints


@dataclass
class RegionGraph:
    node_cell: np.ndarray       # (N,) int -- cell index owning this node
    node_facet: np.ndarray      # (N,) int -- facet/edge index within that cell
    node_features: np.ndarray  # (N, 6) float -- see FEATURE_NAMES
    node_endpoints: np.ndarray  # (N, 2, 2) float -- this node's facet segment (a, b)
    edges: np.ndarray          # (E, 2) int -- canonical (i<j) node-index pairs, undirected
    edge_type: List[str] = field(default_factory=list)  # length E: "intra" or "inter"

    @property
    def num_nodes(self) -> int:
        return len(self.node_cell)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def node_midpoint(self, node_idx: int) -> np.ndarray:
        a, b = self.node_endpoints[node_idx]
        return (a + b) / 2.0

    def set_agent_cell(self, cell_idx: int) -> None:
        """Update the dynamic contains_agent flag in place (default False at
        construction, since agent position isn't known at graph-build time)."""
        self.node_features[:, FEATURE_NAMES.index("contains_agent")] = (self.node_cell == cell_idx).astype(float)

    def inter_cell_pairs(self):
        """Set of (cell_a < cell_b) pairs connected by >=1 inter-cell edge."""
        pairs = set()
        for (i, j), et in zip(self.edges, self.edge_type):
            if et == "inter":
                ca, cb = int(self.node_cell[i]), int(self.node_cell[j])
                pairs.add((min(ca, cb), max(ca, cb)))
        return pairs


def build_region_graph(partition, instance) -> RegionGraph:
    num_cells = partition.num_cells
    cell_constraints = [_cell_edge_constraints(_get_cell_vertices(partition, c)) for c in range(num_cells)]

    node_cell: List[int] = []
    node_facet: List[int] = []
    node_features: List[List[float]] = []
    node_endpoints: List[Tuple[np.ndarray, np.ndarray]] = []
    node_index = {}  # (cell_idx, facet_idx) -> node index

    hazard_set = set(instance.hazard_cells)
    for cell_idx in range(num_cells):
        is_goal = float(cell_idx == instance.goal_cell)
        is_hazard = float(cell_idx in hazard_set)
        for facet_idx, c in enumerate(cell_constraints[cell_idx]):
            node_index[(cell_idx, facet_idx)] = len(node_cell)
            node_cell.append(cell_idx)
            node_facet.append(facet_idx)
            node_features.append([c["normal"][0], c["normal"][1], c["offset"], is_goal, is_hazard, 0.0])
            node_endpoints.append((c["a"], c["b"]))

    edge_dict = {}  # canonical (min,max) node-index pair -> edge_type

    # Intra-cell: vertex-incidence rule, checked directly (not assumed as a
    # cyclic shortcut) -- for each vertex of a cell, find every constraint it
    # activates (lies on, within tolerance) and connect all such constraints
    # pairwise. For a simple convex polygon each vertex activates exactly 2
    # constraints (its incoming/outgoing edge), giving a k-cycle for a
    # k-facet cell, but this loop doesn't assume that -- it would also
    # correctly handle a vertex that happened to activate 3+ constraints.
    for cell_idx in range(num_cells):
        verts = _get_cell_vertices(partition, cell_idx)
        constraints = cell_constraints[cell_idx]
        for v in verts:
            activated = [
                i for i, c in enumerate(constraints)
                if abs(c["normal"] @ v - c["offset"]) < _VERTEX_ACTIVATION_TOL
            ]
            for a_pos in range(len(activated)):
                for b_pos in range(a_pos + 1, len(activated)):
                    i, j = activated[a_pos], activated[b_pos]
                    ni, nj = node_index[(cell_idx, i)], node_index[(cell_idx, j)]
                    key = (min(ni, nj), max(ni, nj))
                    edge_dict[key] = "intra"

    # Inter-cell: strict facet-sharing -- positive-length collinear overlap
    # with opposing outward normals. Deliberately NOT neighbors_adjacent.
    for p in range(num_cells):
        cp = cell_constraints[p]
        for q in range(p + 1, num_cells):
            cq = cell_constraints[q]
            for i, ci in enumerate(cp):
                for j, cj in enumerate(cq):
                    overlap = segment_overlap_length(ci["a"], ci["b"], cj["a"], cj["b"])
                    if overlap <= _FACET_OVERLAP_EPS:
                        continue
                    if np.dot(ci["normal"], cj["normal"]) >= 0:
                        continue  # collinear but same-facing normals -- not a real shared boundary
                    ni, nj = node_index[(p, i)], node_index[(q, j)]
                    key = (min(ni, nj), max(ni, nj))
                    assert edge_dict.get(key, "inter") == "inter", (
                        f"node pair {key} classified as both intra- and inter-cell -- should be impossible "
                        "since intra-cell edges only ever connect nodes sharing the same cell_idx"
                    )
                    edge_dict[key] = "inter"

    ordered_keys = sorted(edge_dict.keys())
    edges = np.array(ordered_keys, dtype=int) if ordered_keys else np.zeros((0, 2), dtype=int)
    edge_type = [edge_dict[k] for k in ordered_keys]

    return RegionGraph(
        node_cell=np.array(node_cell, dtype=int),
        node_facet=np.array(node_facet, dtype=int),
        node_features=np.array(node_features, dtype=float),
        node_endpoints=np.array(node_endpoints, dtype=float),
        edges=edges,
        edge_type=edge_type,
    )
