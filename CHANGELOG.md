# Salvador 0.1.0

## Strategies

- **c1..c9:** the linear-time ensemble of 0.0.9, unchanged. `ensemble_candidates` returns c1..c11; c9 still starts from the best of c1..c8.
- **c10: edge-dominating-set gadget (new)** (`salvador/eds_gadget.py`).
  - **Gadget:** every original edge {u, v} becomes an edge-node. Every vertex u with neighbours v1..vd gets incidence nodes (u,1)..(u,ceil(d/2)), where (u,k) is joined to the edge-nodes of u's (2k-1)-th and 2k-th edges. There are no pendants.
  - **Structure:** the gadget is bipartite with maximum degree 2, a disjoint union of paths and even cycles with 2m edges. Minimum edge dominating set is NP-hard in general, even on bipartite graphs of maximum degree 3 (Yannakakis–Gavril 1980), but here it is solved exactly in linear time: a path or cycle with L edges needs exactly ceil(L/3) edges, every third one. On cycles, the rotation that adds the fewest new vertices is kept.
  - **Decoding:** u enters the cover when a chosen gadget edge uses an incidence node of u. This always covers every original edge, with no repair, and the cover is then pruned. c10 runs in O(n + m).
- **c11: bounded-treewidth exact DP (new)** (`salvador/treewidth_dp.py`).
  - **Method:** bucket (variable) elimination along a greedy minimum-degree elimination order, solving maximum independent set exactly on the tree decomposition this order defines.
  - **Width cap:** a vertex is eliminated only while its current fill-graph degree is at most `tw_max_width`, default 10. Bags therefore have at most 11 vertices.
  - **Work budget:** eliminations stop once the dynamic-programming tables (Σ 2^|bag|) would exceed `tw_budget × (n + m)`, default 256. This is the "treewidth small enough relative to the size of the instance" condition, and it keeps c11 worst-case O(n + m).
  - **Core:** the vertices left un-eliminated form the core R, which is fixed as in the best of c1..c10. The rest is solved exactly given that choice.
  - **Guarantees:** if R is empty, c11 is a **minimum vertex cover**. This covers trees, cycles, series-parallel and outerplanar graphs, thin grids, and any graph whose min-degree elimination width is at most the cap. In every case c11 is valid and no larger than the best of c1..c10.
  - **Checks:** optimal on all 298 small random graphs with an empty core (brute force); 0.9 s on a 50 000-vertex tree, exact.
- **c12: swap-maximize refinement (new, final strategy).** `refine_cover` refines every candidate c1..c11:
  - For each u in C ∪ {v_max}, it moves u into the independent set and evicts N(u).
  - It then regrows a maximal independent set in ascending-degree order (`maximize_solution`).
  - It keeps the result if strictly smaller.
- **Return values:** `find_vertex_cover` returns c12. `final_candidates(G, return_refined=True)` returns all twelve strategies plus each candidate's refinement.

## Guarantees

- c12 is never larger than any of c1..c11.
- c11 is a minimum vertex cover whenever its elimination core is empty. Every guarantee of 0.0.9 therefore still holds, including ratio at most 2 on every graph.
- Every strategy returns a valid cover.

## Running time: worst-case O(n + m)

- **c1..c9:** O(n + m), as before.
- **c10:** O(n + m).
- **c11:** O(n · w² + m) for the elimination plus at most `tw_budget × (n + m)` table entries: O(n + m) for the fixed defaults.
- **c12, local evaluation:** every candidate is a minimal cover, so only neighbours of evicted vertices can re-enter the independent set. Each move is therefore evaluated locally, and the result is identical to the global `maximize_solution` pass (tested move for move).
- **c12, work budget:** every adjacency scan, and the sort of each local candidate list, is charged to a work counter. Moves stop when the next one would exceed `refine_budget × (n + m)`, with a default of 100, per distinct candidate cover.
  - This makes c12, and the whole algorithm, worst-case O(n + m).
  - When the budget is not exhausted, which is the usual case, the result equals the unbounded pass.
  - `refine_budget=None` removes the cap; the worst case is then O(|C|·(n + m)).

## Removed (from the intermediate 0.1.0 draft)

- The earlier c10 (König–Egerváry exact) and c11 (Nemhauser–Trotter kernel), with the modules `ke_exact.py` and `lp_kernel.py`, and `find_vertex_cover_certified`. They needed maximum matchings or an LP, so they were not linear.

## car/

- **`car_strategies.py`** reports:
  - wins, sole wins and win rate for the base strategies c1..c8, c10 and c11;
  - c9's strict gain over the best of c1..c8;
  - c12's strict gain over the best of c1..c11;
  - `c12_from`, the strategies whose refinement produced the final cover.

  - per instance: whether c11 was exact, its core size and its width; per family: c11's exact rate, mean core fraction and maximum width.

  New options: `--refine-budget` (-1 = unbounded), `--tw-max-width` and `--tw-budget`.
- **`car_ratio.py`:**
  - It skips the sparse Erdős–Rényi G(n, p) Part B instance with n = 200 000, before building it, because construction takes too long. `--keep-gnp-200k` restores it.
  - `--skip PATTERN` skips any other Part B instance whose name or group contains the pattern.
  - Skipped instances are listed in `car_ratio.json`.

## Tests

`tests/test_v010.py`:
- every strategy is a valid cover, and c12 is never larger than c1..c11;
- c11 is optimal when its core is empty, exact on trees, and never larger than its reference;
- the c10 gadget has maximum degree 2 and 2m edges, and its edge dominating set is minimum (brute force on small gadgets);
- the local refinement equals the global `maximize_solution` pass;
- the default budget does not change results on small graphs;
- a tiny budget still gives a valid cover.
