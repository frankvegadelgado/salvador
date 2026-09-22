# Changelog

## v0.0.7

- **Linear-time fix for the Min-to-Min (MtM) heuristic.** `min_to_min_vertex_cover_linear` previously pooled the neighborhoods of *every* vertex sharing the current minimum degree on each outer-loop step and rescanned that pooled set to break ties. On regular or near-regular graphs the entire active vertex set can share one bucket, and since only one vertex was consumed per rescan, the pooled neighborhood was recomputed from scratch up to `|bucket|` times, degrading to `O(n^2)`; profiling on random 3-regular graphs confirmed super-linear wall-clock growth (e.g. ~11x time for a 4x increase in `n`). The routine now pops and processes exactly one minimum-degree vertex per step and picks its cover target in `O(1)` from that vertex's current neighbor set, restoring an unconditional `O(n + m)` bound (verified on random 3-regular graphs up to `n=32000`, where time-per-vertex stays flat instead of growing).
- **Unified the production `epsilon` default to `1`.** `salvador.vc_reduction.reduce_vc_to_mids` and `salvador.vc_reduction.solve_vc` previously defaulted to `epsilon=0.1` (Baker layering width `k=10`), inconsistent with `salvador.algorithm.find_vertex_cover`'s `epsilon=1` (`k=1`) default. Both now default to `epsilon=1`, which skips the tree-decomposition PTAS pass entirely and falls back to the linear-time greedy weighted-IDS baseline, so the whole production ensemble is strictly `O(n + m)` by default. Smaller `epsilon` remains available for offline quality experiments but is no longer the default anywhere in the package.
- **Robustness fix for heterogeneous node labels.** `min_weighted_vertex_cover_max_degree_1`'s tie-break compared node labels directly with `<`, which raises `TypeError` on graphs whose node labels are not mutually comparable (e.g. a mix of strings and tuples, as produced by some of the new `car/` adversarial generators). The tie-break now compares `str(node)` instead.
- **Documented the Hvala ensemble provenance.** Four of the six linear-time candidates evaluated by `find_vertex_cover` (maximal matching, bucket-queue max-degree greedy, the degree-1 weighted-reduction "Hallelujah" heuristic, and redundant-vertex pruning) adapt the ensemble published as the Hvala algorithm (Frank Vega, *The Hvala Algorithm*, Gauge Freedom Journal, v1 i1-004, DOI: 10.65323/gfj.2026.004, 2026; PyPI package `hvala`), which proves an unconditional `O(n + m)` time/space bound and a worst-case approximation ratio at most 2. This is now documented directly in `salvador/algorithm.py`.
- **Replaced the `car/` suite** with a two-part reproducibility script: Part A is the existing small-graph exact-ratio suite, now run under the linear-time default `epsilon=1`; Part B is a new large adversarial-graph suite (complete bipartite, crown graphs, double-star bridges, a hierarchical greedy-tie-break stress construction, random regular graphs, Barabasi-Albert hub graphs, sparse Erdos-Renyi graphs, and Watts-Strogatz small-world graphs) with a doubling-size scaling study that fits wall-clock time against `n + m` to empirically verify the `O(n + m)` guarantee at scale.
- Bumped package version metadata to 0.0.7.

## v0.0.6

- Replaced the `car/` suite with a default-call (`epsilon=0.1`) experiment that measures the approximation ratio against exact optima (Koenig certificates on bipartite instances, no MILP) and checks the `7/4` threshold. The largest observed ratio is `7/4`, attained by an eleven-vertex bipartite witness; no instance exceeds it.
- Updated regression smoke tests and documentation for the default-call `7/4` results.
- Bumped package version metadata to 0.0.6.

## v0.0.5

- **Activated the `epsilon` parameter.** The weighted-IDS pass is the Baker-style PTAS in `baker_ptas.baker_ptas_ids_weighted`: `epsilon` controls the layering width `k = ceil(1/epsilon)`, so smaller `epsilon` yields a more thorough (and never worse) solve, with the greedy maximal independent set as the `k = 1` baseline and fallback.
- Because the forest-core gadget is itself a forest, the PTAS solves it near-optimally, so the decoded cover reflects a minimum-weight independent dominating set of the core.
- Added the `car/` experiment folder: a reproducible suite that measures the default-call (`epsilon=0.1`) approximation ratio against exact optima on feasible graphs (Koenig certificates on bipartite instances, no MILP) and checks the `7/4` threshold.
- Bumped package version metadata to 0.0.6.

## v0.0.4

- Centralized package version metadata in `salvador.version` and updated all CLI version flags.
- Kept the existing Salvador algorithmic pipeline intact: cleanup, spanning-forest core, weighted MIDS gadget, greedy weighted IDS pass, edge repair, and redundancy pruning.
- Improved DIMACS parsing, compressed-file handling, deterministic CLI formatting, and generated DIMACS edge counts.
- Clarified documentation to distinguish implemented guarantees from conjectural approximation-ratio claims.
- Added regression smoke tests and a GitHub Actions test workflow.
- Updated packaging metadata and build configuration.
