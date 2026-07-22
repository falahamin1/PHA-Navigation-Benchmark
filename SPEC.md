# Nav-Benchmark — Numerical Specification (Part 2, implementation target)

Verbatim reference. This is the implementation-time source of truth for exact
constants. Do not edit without flagging the change — if a constant needs to
change during implementation (e.g. after Gate 1 validation), note it here
explicitly with the reason rather than silently overwriting.

## 2.1 State, dynamics, integration

State vector x = [x₁, x₂, v₁, v₂]ᵀ ∈ ℝ⁴ (position, velocity).

Per-cell continuous dynamics (affine with drift), following the benchmark's ẋ = A·x − B·u(i,j):

A matrix (velocity-coupling block; position derivatives are just velocity):

```
ẋ₁ = v₁
ẋ₂ = v₂
[v̇₁]   [ -1.2   0.1 ] [v₁]        [u₁(cell)]
[v̇₂] = [  0.1  -1.2 ] [v₂]  + C · [u₂(cell)]
```

Equivalently the classic form ẋ = A(x − u_ext) with the 4×4 A:

```
A = [ 0     0     1      0   ]
    [ 0     0     0      1   ]
    [ 0     0    -1.2    0.1 ]
    [ 0     0     0.1   -1.2 ]
```

and drift term B·u(i,j) injected into the v̇ rows, where u(i,j) is the cell's desired velocity (unit vector scaled by the coupling). Use the C2E2 instance values as the verified default (they reproduce a known-good vector field). The original Fehnker–Ivančić paper uses A with diagonal −0.8 and off-diagonal −0.2 as an alternative; expose A as a config so both can be tried. Implementation note: parameterize A and the drift scaling; default to the C2E2 −1.2/0.1 coupling since we have its exact per-mode drift values below.

Verified per-mode drift (from C2E2, four canonical directions — use these to validate the integrator before generalizing to 8 directions):

- Mode 0: v̇₁ = −1.2v₁ + 0.1v₂ − 0.1, v̇₂ = 0.1v₁ − 1.2v₂ + 1.2
- Mode 1: v̇₁ = −1.2v₁ + 0.1v₂ − 4.8, v̇₂ = 0.1v₁ − 1.2v₂ + 0.4
- Mode 2: v̇₁ = −1.2v₁ + 0.1v₂ + 2.4, v̇₂ = 0.1v₁ − 1.2v₂ − 0.2
- Mode 3: v̇₁ = −1.2v₁ + 0.1v₂ + 3.9, v̇₂ = 0.1v₁ − 1.2v₂ − 3.9

The constant terms are exactly −B·u for each direction; extract B and the 8 direction vectors by matching this pattern.

Desired-direction set (8 compass unit vectors, the action space):

```
E  = ( 1,  0)      NE = ( √½,  √½)
N  = ( 0,  1)      NW = (−√½,  √½)
W  = (−1,  0)      SW = (−√½, −√½)
S  = ( 0, −1)      SE = ( √½, −√½)
```

Integration: fixed-step Euler, Δt = 0.05 (config). Detect boundary crossing each step by testing which cell polytope contains the new position (PPL point-in-polytope). On crossing, switch mode and, at decision points, query the policy for the new desired direction.

## 2.2 Grid / partition specs per tier

- **EASY**: 5×5 unit-box grid over [0,5]×[0,5]. Cells = 25 unit squares. Facets/cell = 4, all identical. Adjacency = 4-neighbor lattice.
- **MEDIUM**: same [0,5]×[0,5] domain, partitioned into ~25–35 convex polygons mixing squares, right triangles (split some squares diagonally), and trapezoids. Facet counts ∈ {3,4,5}. Generate procedurally with a seed so instances are reproducible and poolable.
- **HARD**: irregular convex partition of [0,5]×[0,5] — e.g. sample ~30 seed points, compute Voronoi, clip cells to the domain, keep convex cells (or convex-decompose non-convex ones). Add 2–4 hazard regions as convex polygons of varying facet count placed on interior cells. Facet counts ∈ {3,…,6}.

For all tiers, expose: `grid_seed`, `num_cells`, `goal_cells` (set), `hazard_cells` (set), and the per-cell `desired_direction` map (this last one becomes the action target, not fixed input, in the RL setting).

## 2.3 Initial state, goal, hazard (canonical defaults)

- Initial region (from the benchmark): x₀ ∈ [0.5, 1.5]×[0.5, 1.5] (a start cell), v₀ ∈ [−0.2, 0.2]×[−0.2, 0.2]. Sample uniformly at episode reset.
- Goal: designate the top-right cell (or a configurable cell) as absorbing reach-cell, reward +10.
- Hazard: designate ≥1 interior cell(s) as absorbing avoid-cell(s), reward −10. (Benchmark labels these A/B respectively.)
- Horizon T: 200 decision steps (config).
- Discount γ = 0.99.

## 2.4 Instance-pool generation (for train/test split)

Per tier, generate N_train + N_test instances by randomizing: partition seed (MEDIUM/HARD), goal-cell location, hazard-cell locations, and initial velocity sign. Target ≥120 held-out test instances per tier. Guarantee solvability by checking BFS reachability from start to goal avoiding hazards on the cell-adjacency graph; discard unsolvable draws.

## 2.5 Encoder input tensors

- **H-Rep**: per cell, up to F_max facets × (normal_x, normal_y, offset) → tensor [num_cells, F_max, 3], plus per-cell flag vector (goal, hazard, agent-here, desired-dir one-hot). Pad facets to F_max (=6 for HARD).
- **V-Rep**: per cell, up to V_max vertices × (x, y) → [num_cells, V_max, 2] + same flags.
- **Graph**: node list = cells; node feat = [centroid_x, centroid_y, facet-count, goal, hazard, agent-here, desired-dir one-hot]; edges = shared-facet adjacency with edge feat = [shared_normal_x, shared_normal_y, shared_len]. Include agent continuous state (x,v) as a global feature concatenated post-pooling for all encoders.
- **CNN**: rasterize domain to, say, 40×40; channels = {occupancy/traversable, goal-mask, hazard-mask, agent-position (splatted), desired-dir field}. Keep param count near the DeepSets (use global pooling before FC, not flatten).
- **MLP**: concatenate per-cell (centroid, facet params padded, flags) + agent (x,v) into one padded vector.

## 2.6 Training config (inherit the PHA/PPO setup)

PPO + GAE, λ=0.95, γ=0.99, clip 0.2, entropy coef 0.05, lr 1e-4, DeepSet ~53k params baseline; match others. Rollout length per the PHA harness. Seeds: 5–10. Geometry ops (containment, adjacency, facet-sharing, V↔H conversion) via the Parma Polyhedra Library (PPL), already used elsewhere in this repo and — conveniently — exactly what PHAVer uses on this same benchmark.

## 2.7 Validation checkpoints (build order)

1. Implement 4D Euler integrator; reproduce the four C2E2 modes' vector fields; sanity-check a trajectory converges toward each desired direction.
2. Implement box-grid (EASY) partition + point-in-cell + mode switching; verify a fixed direction map produces sensible trajectories to goal.
3. Wrap as Gym-style env with the 8-direction action space + reward; train H-Rep DeepSet; confirm it solves EASY.
4. Add MEDIUM/HARD partition generators + PPL adjacency; build the cell-adjacency graph; add remaining 5 encoders.
5. Add instance-pool generation + BFS solvability filter + train/test split.
6. Add metrics/stats layer (Wilson, McNemar, Wilcoxon, rliable IQM/CIs, BFS oracle).
7. Run the ν-vs-advantage sweep across tiers.

See `ROADMAP.md` for these checkpoints restated as gates with concrete test criteria and stop markers for this specific repo.
