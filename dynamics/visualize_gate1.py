"""Gate 1 diagnostic: one trajectory per direction, starting from rest at the
origin, no boundaries. Run directly: `python visualize_gate1.py`. Look at the
saved figure and confirm each trajectory curves the way physical intuition
says (apply "East", the thing drifts east).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from integrator import DIRECTIONS, integrate

OUT_PATH = "gate1_trajectories.png"
N_STEPS = 60  # dt=0.05 -> 3s, enough to show the curve without leaving the plot frame


def main():
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.hsv(np.linspace(0, 1, len(DIRECTIONS), endpoint=False))

    for (name, u), color in zip(DIRECTIONS.items(), colors):
        state0 = np.zeros(4)
        traj = integrate(state0, u, n_steps=N_STEPS)
        x1, x2 = traj[:, 0], traj[:, 1]
        ax.plot(x1, x2, color=color, label=name, linewidth=2)
        ax.annotate(name, (x1[-1], x2[-1]), color=color, fontsize=11, fontweight="bold")
        ax.plot(x1[0], x2[0], "ko", markersize=3)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Gate 1: open-loop trajectories from rest, one per direction")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
