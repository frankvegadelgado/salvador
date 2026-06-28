# car/ — epsilon-aware sqrt(2) experiment for Salvador v0.0.5

This folder tests whether the accuracy-controlled Salvador pipeline keeps the
approximation ratio at or below `sqrt(2) ≈ 1.41421`.

## The pipeline being tested

`salvador.algorithm.find_vertex_cover(G, epsilon)`:

1. cleans `G` (self-loops, isolated vertices),
2. builds a spanning-forest core and the weighted MIDS gadget (itself a forest),
3. solves the gadget with a Baker-style PTAS of layering width `k = ceil(1/epsilon)`
   (tree-decomposition dynamic programming on each component),
4. decodes a cover, repairs uncovered edges by the higher-degree endpoint, and
   prunes redundant vertices.

Smaller `epsilon` means a more thorough solve. No single `epsilon` is uniformly
best, so the experiment runs a fixed schedule and keeps the smallest valid cover.

## What the script does

`car_sqrt2_experiment.py`:

- Uses `find_vertex_cover` as the candidate solver, passing a real `epsilon`.
- Computes **exact** optima feasibly via `tau(G) = |V| - alpha(G)` with a
  deterministic branch-and-bound maximum-independent-set solver (**no MILP**), so
  every ratio is measured against a true OPT on feasible-size graphs.
- Sweeps `epsilon in {1.0, 0.5, 0.25, 0.1, 0.05}` and records, for each graph,
  the ratio at every `epsilon` and the **best** (smallest) ratio over the sweep.
- Tests the threshold `sqrt(2)`: it reports the maximum best-ratio and counts any
  instance whose best ratio still exceeds `sqrt(2)`.
- Covers many families: the NetworkX graph atlas (all graphs up to 7 vertices),
  structured families (paths, cycles, cliques, stars, complete bipartite, wheels,
  grids, barbells), Erdos–Renyi random graphs, and the hardest near-regular class
  (Petersen, dodecahedral, Desargues, random `d`-regular), plus an explicit
  balanced 8-vertex stress instance.

## Run

From the repository root:

```bash
python car/car_sqrt2_experiment.py            # full feasible suite
python car/car_sqrt2_experiment.py --quick    # smaller, faster sweep
```

Outputs:

- `car_sqrt2_experiment.json`: parameters, environment, per-epsilon summaries,
  per-family summaries, the worst instance with an exact optimum cover, and raw rows.
- `car_sqrt2_summary.csv`: compact per-family table.

## Reading the result

The key fields are `per_epsilon_summary` (how the max ratio moves with `epsilon`)
and `conclusion.max_best_ratio` / `conclusion.count_best_above_sqrt2` (does any
graph's best-over-schedule ratio exceed `sqrt(2)`?). The `sqrt(2)` claim is
supported **only** if `count_best_above_sqrt2 == 0`; otherwise the worst instance
is reported so the claim can be revised. This is empirical evidence, not a proof.
