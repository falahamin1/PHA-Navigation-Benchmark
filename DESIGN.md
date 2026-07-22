# Nav-Benchmark — Design Sketch (Part 1)

Verbatim reference. This is the standing rationale document — *why* each later
component exists. Do not edit without flagging the change; if the design
changes, note the change explicitly rather than silently rewriting history.

## 1.1 The core idea and why it earns the GNN a win

The Fehnker–Ivančić navigation benchmark is a standard, heavily-cited hybrid-systems benchmark: an object moves in the plane over a grid of cells, each cell being a discrete mode with affine drift dynamics ẋ = Ax − Bu(i,j) steering velocity toward that cell's desired direction u(i,j). Some cells are goals (reach), some are hazards (avoid). You turn verification into control: the agent picks each cell's desired direction to drive the object from start to goal while avoiding hazards.

The vanilla benchmark uses unit-box cells, which have fixed, trivial incidence — the same trap that neutralized the GNN on Rush Hour and the lawn mower. So the design deliberately introduces an incidence-variability axis: three difficulty tiers that hold the task fixed while increasing how much the cell geometry's incidence structure varies and matters. The GNN operates over a cell-adjacency graph whose topology is trivial at the box tier and rich at the irregular tier. The central experimental claim: the GNN's advantage over set encoders scales with incidence variability, with the box tier serving as the controlled low-incidence anchor that reproduces (and now explains) the existing PHA nulls.

## 1.2 The MDP framing

- Continuous state: s = (x₁, x₂, v₁, v₂) — position and velocity.
- Discrete mode: index of the current cell (which polytope contains the position).
- Dynamics: within a cell, ẋ = A·x − B·u(cell), integrated by fixed-step Euler (Δt = 0.05, tunable). Mode switches when the position crosses a cell boundary.
- Action: at each decision point (cell entry), the agent selects the desired-direction label for steering — one of the 8 compass directions {E, NE, N, NW, W, SW, S, SE} (unit vectors). This is the RL control knob replacing the benchmark's fixed map. Discrete action space |A| = 8.
- Episode: starts at an initial region; ends on reaching a goal cell (success), entering a hazard cell (failure), or hitting horizon T.
- Observation = the polytopic region encoding (see §1.4) — the set of cells with their half-space/vertex/graph structure, the agent's current (x, v), and goal/hazard flags.

## 1.3 The incidence-variability difficulty ladder

This is the experimental spine. Three tiers, same task, increasing incidence richness:

- **EASY (box grid).** Standard m×n unit-box partition (e.g. 5×5). Every cell is a unit square: 4 facets, identical fixed incidence, adjacency graph is a regular lattice. Prediction: GNN ties DeepSets — this is the control that reproduces the lawn-mower/water-monitor null and confirms the "no varying incidence → no GNN benefit" mechanism.
- **MEDIUM (mixed convex partition).** Partition the plane into a mix of convex polygons — squares, triangles, trapezoids — so cells have varying facet counts (3–5) and the adjacency graph has non-uniform degree. Prediction: GNN advantage begins to appear.
- **HARD (irregular convex partition + hazard shapes).** A fully irregular convex partition (e.g. from a randomized convex decomposition or a perturbed Voronoi diagram clipped to convex cells), plus multiple non-box hazard regions of varying shape. Incidence topology varies strongly across states and is task-relevant (which hazard facet borders which traversable cell governs safe routing). Prediction: GNN advantage largest.

Define a scalar incidence-variability index ν for the x-axis of the key figure — e.g. the entropy (or variance) of the per-cell facet-count distribution, or of the cell-adjacency-graph degree distribution, across the partition. ν(EASY) ≈ 0 (all cells identical), rising through MEDIUM to HARD. The headline plot is (GNN reward − best DeepSet reward) vs. ν, expected monotone increasing.

## 1.4 Encoders (the ladder, consistent with Rush Hour/Tangram)

Six encoders, each consuming the same information, differing only in structure:

1. **MLP** — flat padded vector of per-cell features (centroid, facet params) + agent (x,v). No object structure, not permutation-invariant. Floor.
2. **CNN** — rasterize the partition + goal/hazard/agent-position channels to an image. Spatial structure, no object/set structure.
3. **H-Rep DeepSet** — each cell = bag of half-spaces {(c, b)}; per-cell DeepSet → region pooling.
4. **V-Rep DeepSet** — each cell = bag of vertices; DeepSet → region pooling.
5. **DeepSet + relational features** — H-rep bag augmented with hand-designed pairwise cell-adjacency indicators, then pooled. The critical ablation — if the GNN only ties this, the finding is "explicit relational features suffice; GNN is one way to get them."
6. **GNN over the region-level cell-adjacency graph** — the proposed method.

## 1.5 The region-level cell-adjacency graph (the architectural point)

Analogous to the region-level contact graph idea, but for cells:

- **Nodes**: one per cell (or, richer, one per half-space constraint across all cells). Node feature = cell's facet parameters (normals + offsets), centroid, desired-direction one-hot, and flags {is-goal, is-hazard, contains-agent}.
- **Edges**: connect two cells that share a facet (are geometrically adjacent). Edge feature = the shared boundary's normal/length. Optionally add edges to goal/hazard cells within k-hops to propagate routing signal.

Message passing over this graph lets the policy reason about routes through the adjacency structure — precisely what a bag-of-cells DeepSet cannot represent, and precisely what matters more as the partition becomes irregular.

Register in writing: "message passing over cell adjacency should help in proportion to how much adjacency topology varies across states; the box grid (uniform adjacency) is the null control."

## 1.6 Reward

- Step penalty: −0.01 (encourage short paths).
- Progress shaping: γΦ(s′) − Φ(s), Φ = negative distance-to-goal-cell-centroid (potential-based, policy-invariant per Ng et al.).
- Hazard entry: −10 (absorbing).
- Goal reach: +10 (absorbing).
- Optional smoothness penalty on direction changes if desired for cleaner trajectories.

## 1.7 Statistics & protocol (inherit the Rush Hour machinery)

- Instance pool per tier: generate many partitions/maps per tier (randomized direction maps + goal/hazard placements), disjoint train/test split, evaluate zero-shot on held-out instances → hundreds of independent test outcomes.
- Solve rate → Wilson intervals; paired method comparison → McNemar on shared held-out instances; steps-to-goal / optimality gap → Wilcoxon signed-rank vs. a BFS/Dijkstra oracle on the cell-adjacency graph (cheap to compute, gives ground-truth shortest routes).
- ≥5–10 seeds per method per tier; aggregate with IQM + stratified bootstrap CIs (rliable), report probability of improvement for GNN-vs-each. Holm–Bonferroni across the multiple pairwise tests.
- Report parameter counts for all six encoders in a table; match them within a small factor.

## 1.8 What "success" and "honest failure" look like

- **Success**: monotone ν-vs-advantage curve, GNN > DeepSet at HARD with non-overlapping CIs, GNN ≈ DeepSet at EASY. Clean causal story.
- **Honest failure to plan for**: GNN ties encoder #5 (relational DeepSet) even at HARD → report as "relational features are what matter"; still a publishable, well-controlled result. This is why #5 is non-negotiable.
