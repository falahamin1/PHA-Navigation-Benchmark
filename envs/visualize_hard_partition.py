"""Step 2b gate diagnostic: run directly with `python visualize_hard_partition.py`.

Renders one IrregularConvexPartition instance (cells filled with distinct
colors, facet count labeled per cell, hazard-eligible cells marked with a
hatch pattern + red outline, nu in the title) and prints the full nu table:
box, 5x mixed, 5x hard, side by side.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from incidence import incidence_variability
from partitions import BoxGridPartition, IrregularConvexPartition, MixedConvexPartition

OUT_PATH = "gate2b_hard_partition.png"
DEMO_SEED = 0


def main():
    partition = IrregularConvexPartition(grid_seed=DEMO_SEED)
    nu = incidence_variability(partition)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.tab20
    for i in range(partition.num_cells):
        verts = partition.cell_vertices(i)
        is_hazard = i in partition.hazard_eligible_cells
        color = cmap(i % 20)
        ax.add_patch(MplPolygon(
            verts, closed=True, facecolor=color, edgecolor="black", linewidth=1.0,
            hatch="///" if is_hazard else None,
        ))
        if is_hazard:
            ax.add_patch(MplPolygon(verts, closed=True, fill=False, edgecolor="red", linewidth=2.5))
        c = partition.cell_centroid(i)
        label = f"{partition.cell_facet_count(i)}" + (" (H)" if is_hazard else "")
        ax.text(c[0], c[1], label, ha="center", va="center", fontsize=8, fontweight="bold")

    xmin, xmax, ymin, ymax = partition.domain
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(f"Step 2b: IrregularConvexPartition (seed={DEMO_SEED}), {partition.num_cells} cells, "
                 f"nu = {nu:.4f}\n(label = facet count; red outline/hatch = hazard-eligible)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")

    print("\nnu table (box, 5x mixed, 5x hard):")
    nu_box = incidence_variability(BoxGridPartition())
    print(f"  nu(box)                 = {nu_box:.4f}")
    for seed in range(5):
        nu_mixed = incidence_variability(MixedConvexPartition(grid_seed=seed))
        print(f"  nu(mixed, seed={seed})      = {nu_mixed:.4f}")
    for seed in range(5):
        nu_hard = incidence_variability(IrregularConvexPartition(grid_seed=seed))
        print(f"  nu(hard,  seed={seed})      = {nu_hard:.4f}")


if __name__ == "__main__":
    main()
