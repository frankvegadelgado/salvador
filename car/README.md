# car/ — default-call approximation-ratio experiment for Salvador v0.0.5

This folder tests the public default call `find_vertex_cover(G, epsilon=0.1)`
against exact optima and checks the conservative threshold `7/4 = 2 - 1/4`.

## The pipeline being tested

`salvador.algorithm.find_vertex_cover(G, epsilon)`:

1. cleans `G` (self-loops, isolated vertices),
2. builds a spanning-forest core and the weighted MIDS gadget (itself a forest),
3. solves the gadget with a Baker-style PTAS of layering width `k = ceil(1/epsilon)`
   (tree-decomposition dynamic programming on each component),
4. decodes a cover, repairs uncovered edges by the higher-degree endpoint, and
   prunes redundant vertices.

The returned cover depends on `epsilon` through the layer choices and is **not**
monotone: the bipartite obstruction below returns size 5 at the default
`epsilon = 0.1` but is solved optimally at `epsilon in {0.25, 0.5}`.

## What the script does

`car_experiment.py`:

- Uses `find_vertex_cover(G, epsilon=0.1)` — the public default call.
- Computes **exact** optima with **no MILP**: maximum matching + Koenig's theorem
  on bipartite instances, and branch-and-bound maximum independent set
  (`tau(G) = |V| - alpha(G)`) otherwise.
- Reports `|C| / tau(G)` per graph and counts any instance exceeding `7/4`.
- Test classes: the explicit seven-vertex bipartite obstruction and a
  relabel/edge-order stress sweep of it, random bipartite graphs, a bipartite
  hill-climbing search, bipartite grids, the NetworkX graph atlas (all graphs up
  to 7 vertices), and random general graphs (`n <= 12`).

The known worst case is the bipartite obstruction with edge set

```text
(0,1), (0,3), (2,1), (4,1), (4,3), (5,0), (5,2), (5,4), (5,6), (6,1), (6,3)
```

for which the default call returns `{0,1,4,5,6}` (size 5) while an exact minimum
cover is `{1,3,5}` (size 3, certified by Koenig's theorem), giving ratio 5/3.

## Run

From the repository root:

```bash
python car/car_experiment.py            # full feasible suite
python car/car_experiment.py --quick    # smaller, faster sweep
```

Outputs:

- `car_experiment.json`: parameters, environment, per-group summaries, the worst
  instance, and all raw rows.
- `car_summary.csv`: compact per-group table.

## Reading the result

The `7/4` claim is supported only if `conclusion.count_above_7_4 == 0`; otherwise
the worst instance is reported so the conjecture can be revised. This is empirical
evidence, not a proof.
