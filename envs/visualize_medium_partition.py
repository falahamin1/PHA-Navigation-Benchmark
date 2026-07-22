"""Step 2a gate diagnostic: run directly with `python visualize_medium_partition.py`.

Renders one MixedConvexPartition instance (cells filled with distinct colors,
facet count labeled per cell, nu in the title) and prints a table of
nu(box) vs. nu(mixed) across 5 seeds.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from incidence import incidence_variability
from partitions import BoxGridPartition, MixedConvexPartition

OUT_PATH = "gate2a_medium_partition.png"
DEMO_SEED = 0


def main():
    partition = MixedConvexPartition(grid_seed=DEMO_SEED)
    nu = incidence_variability(partition)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.tab20
    for i in range(partition.num_cells):
        verts = partition.cell_vertices(i)
        color = cmap(i % 20)
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=color, edgecolor="black", linewidth=1.0))
        c = partition.cell_centroid(i)
        ax.text(c[0], c[1], str(partition.cell_facet_count(i)), ha="center", va="center", fontsize=9, fontweight="bold")

    xmin, xmax, ymin, ymax = partition.domain
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"Step 2a: MixedConvexPartition (seed={DEMO_SEED}), "
                 f"{partition.num_cells} cells, nu = {nu:.4f}\n(label = facet count per cell)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")

    print("\nnu table:")
    nu_box = incidence_variability(BoxGridPartition())
    print(f"  nu(box)                = {nu_box:.4f}")
    for seed in range(5):
        p = MixedConvexPartition(grid_seed=seed)
        nu_seed = incidence_variability(p)
        print(f"  nu(mixed, seed={seed})     = {nu_seed:.4f}  ({p.num_cells} cells)")


if __name__ == "__main__":
    main()
