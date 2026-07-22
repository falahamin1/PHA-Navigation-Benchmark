"""nu -- incidence-variability index (DESIGN.md Section 1.3).

nu(partition) = Shannon entropy (natural log) of the per-cell facet-count
distribution: for each distinct facet count k appearing across the
partition's cells, let p_k be the fraction of cells with that facet count;

    nu = -sum_k p_k * ln(p_k)

nu = 0 exactly when every cell has the same facet count (e.g.
BoxGridPartition, all 4-facet squares). nu increases as facet counts become
more varied/balanced across the partition. This is the x-axis of the paper's
headline plot (DESIGN.md 1.3): (GNN reward - best DeepSet reward) vs. nu.
"""

from collections import Counter

import numpy as np


def incidence_variability(partition) -> float:
    counts = Counter(partition.cell_facet_count(i) for i in range(partition.num_cells))
    total = partition.num_cells
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs))) + 0.0  # normalize away -0.0 for the all-equal case
