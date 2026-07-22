"""Step 3a gate diagnostic: run directly with `python visualize_region_graph.py`.

Renders one MEDIUM instance's region graph overlaid on its partition: a dot
per constraint-node at its facet midpoint, intra-cell edges in one color,
inter-cell edges in another, drawn across the actual shared facets. Prints a
summary: total nodes, #intra, #inter, and #transition-adjacent-but-not-
facet-sharing pairs (corner touches that correctly got no edge).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from pool import Instance
from partitions import MixedConvexPartition
from region_graph import build_region_graph

OUT_PATH = "gate3a_region_graph.png"
DEMO_SEED = 0


def main():
    partition = MixedConvexPartition(grid_seed=DEMO_SEED)
    instance = Instance(
        tier="medium", partition_seed=DEMO_SEED, start_cell=0, goal_cell=24,
        hazard_cells=(8,), initial_velocity_sign=(1, 1),
    )
    graph = build_region_graph(partition, instance)

    n = partition.num_cells
    transition_pairs = {(i, j) for i in range(n) for j in range(i + 1, n) if partition.neighbors_adjacent(i, j)}
    facet_pairs = graph.inter_cell_pairs()
    corner_only = transition_pairs - facet_pairs

    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.cm.Pastel1
    for i in range(n):
        verts = partition.cell_vertices(i)
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=cmap(i % 9), edgecolor="black", linewidth=1.0, alpha=0.7))

    # Nodes are drawn pulled slightly inward from their facet midpoint toward
    # their cell's centroid -- otherwise an inter-cell edge's two endpoints
    # (both ON the shared boundary) coincide with the black cell boundary and
    # are invisible against it. This offset makes both the intra-cell k-cycle
    # and the inter-cell bridge clearly traceable by eye without changing
    # which edges exist.
    midpoints = np.array([graph.node_midpoint(k) for k in range(graph.num_nodes)])
    centroids = np.array([partition.cell_centroid(int(c)) for c in graph.node_cell])
    display_pos = midpoints + 0.15 * (centroids - midpoints)

    for (i, j), et in zip(graph.edges, graph.edge_type):
        p, q = display_pos[i], display_pos[j]
        color = "tab:blue" if et == "intra" else "tab:red"
        lw = 1.2 if et == "intra" else 2.8
        z = 3 if et == "intra" else 4
        ax.plot([p[0], q[0]], [p[1], q[1]], color=color, linewidth=lw, zorder=z, alpha=0.9)

    ax.scatter(display_pos[:, 0], display_pos[:, 1], s=16, color="black", zorder=5)

    xmin, xmax, ymin, ymax = partition.domain
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    n_intra = sum(1 for t in graph.edge_type if t == "intra")
    n_inter = sum(1 for t in graph.edge_type if t == "inter")
    ax.set_title(
        f"Step 3a: region graph over MixedConvexPartition (seed={DEMO_SEED})\n"
        f"{graph.num_nodes} nodes, {n_intra} intra-cell edges (blue), {n_inter} inter-cell edges (red)\n"
        f"#transition-adjacent-but-not-facet-sharing pairs (corner touches, no edge) = {len(corner_only)}"
    )
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="tab:blue", lw=1.5, label="intra-cell edge"),
        Line2D([0], [0], color="tab:red", lw=2.2, label="inter-cell edge (facet-sharing)"),
    ], loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")

    print(f"\nsummary: total_nodes={graph.num_nodes}, intra_edges={n_intra}, inter_edges={n_inter}")
    print(f"#transition-adjacent pairs = {len(transition_pairs)}")
    print(f"#facet-sharing pairs       = {len(facet_pairs)}")
    print(f"#corner-only (no edge)     = {len(corner_only)}")
    if corner_only:
        print(f"  e.g. corner-only pairs (sample): {sorted(corner_only)[:5]}")


if __name__ == "__main__":
    main()
