"""Step 2c gate diagnostic: run directly with `python visualize_instance_pool.py`.

Renders one MEDIUM and one HARD instance (partition + start/goal/hazard cells
+ the BFS solution path found by solve_path) and prints the generation/split
summary table for all three tiers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from pool import build_partition, generate_pool_with_stats, solve_path, train_test_split

POOL_SIZES = {"easy": 800, "medium": 700, "hard": 700}
DEMO_TIERS = ("medium", "hard")


def render_instance(tier, instance, out_path):
    partition = build_partition(tier, instance.partition_seed)
    path = solve_path(partition, instance.start_cell, instance.goal_cell, instance.hazard_cells)
    assert path is not None, "gate-diagnostic instance must be solvable by construction"

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.Pastel1
    path_set = set(path)
    for i in range(partition.num_cells):
        verts = partition.cell_vertices(i) if hasattr(partition, "cell_vertices") else None
        if verts is None:
            x_lo, x_hi, y_lo, y_hi = partition.cell_bounds(i)
            verts = np.array([[x_lo, y_lo], [x_hi, y_lo], [x_hi, y_hi], [x_lo, y_hi]])

        if i == instance.start_cell:
            face, label = "limegreen", "START"
        elif i == instance.goal_cell:
            face, label = "gold", "GOAL"
        elif i in instance.hazard_cells:
            face, label = "red", "HAZ"
        else:
            face, label = cmap(i % 9), ""
        ax.add_patch(MplPolygon(verts, closed=True, facecolor=face, edgecolor="black", linewidth=1.0, alpha=0.85))
        c = partition.cell_centroid(i)
        text = f"{i}"
        if label:
            text += f"\n{label}"
        ax.text(c[0], c[1], text, ha="center", va="center", fontsize=7)

    # Overlay the BFS solution path as centroid-to-centroid line segments.
    centroids = [partition.cell_centroid(i) for i in path]
    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    ax.plot(xs, ys, color="blue", linewidth=2.5, marker="o", markersize=5, zorder=5)

    xmin, xmax, ymin, ymax = partition.domain
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(
        f"Step 2c: {tier.upper()} instance (partition_seed={instance.partition_seed})\n"
        f"start={instance.start_cell}, goal={instance.goal_cell}, hazards={instance.hazard_cells}\n"
        f"BFS path ({len(path)} cells): {path}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    print(f"{'tier':>7}  {'target':>7}  {'raw':>6}  {'rej_unsolv':>10}  {'rej_dup':>8}  "
          f"{'accepted':>8}  {'train':>6}  {'test':>5}  {'seeds(tr/te)':>13}")

    for tier in ("easy", "medium", "hard"):
        n = POOL_SIZES[tier]
        pool, stats = generate_pool_with_stats(tier, n, rng_seed=42)
        train, test = train_test_split(pool, test_fraction=0.2, rng_seed=7)

        if tier == "easy":
            seed_summary = "n/a (1 partition)"
        else:
            train_seeds = {i.partition_seed for i in train}
            test_seeds = {i.partition_seed for i in test}
            seed_summary = f"{len(train_seeds)}/{len(test_seeds)}"

        print(f"{tier:>7}  {n:>7}  {stats['generated_raw']:>6}  {stats['rejected_unsolvable']:>10}  "
              f"{stats['rejected_duplicate']:>8}  {stats['accepted']:>8}  {len(train):>6}  {len(test):>5}  "
              f"{seed_summary:>13}")

        if tier in DEMO_TIERS:
            # Pick an illustrative instance: longest BFS path among the first
            # 40 test instances, so the demo actually shows routing around
            # hazards rather than a trivial direct hop.
            candidates = (test or train)[:40]
            best_instance, best_path = None, None
            for inst in candidates:
                partition = build_partition(tier, inst.partition_seed)
                path = solve_path(partition, inst.start_cell, inst.goal_cell, inst.hazard_cells)
                if best_path is None or len(path) > len(best_path):
                    best_instance, best_path = inst, path
            render_instance(tier, best_instance, f"gate2c_{tier}_instance.png")


if __name__ == "__main__":
    main()
