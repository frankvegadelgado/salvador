# Changelog

## v0.0.5

- **Activated the `epsilon` parameter.** The weighted-IDS pass is the Baker-style PTAS in `baker_ptas.baker_ptas_ids_weighted`: `epsilon` controls the layering width `k = ceil(1/epsilon)`, so smaller `epsilon` yields a more thorough (and never worse) solve, with the greedy maximal independent set as the `k = 1` baseline and fallback.
- Because the forest-core gadget is itself a forest, the PTAS solves it near-optimally, so the decoded cover reflects a minimum-weight independent dominating set of the core.
- Added the `car/` experiment folder: a reproducible suite that measures the default-call (`epsilon=0.1`) approximation ratio against exact optima on feasible graphs (Koenig certificates on bipartite instances, no MILP) and checks the `7/4` threshold.
- Bumped package version metadata to 0.0.5.

## v0.0.4

- Centralized package version metadata in `salvador.version` and updated all CLI version flags.
- Kept the existing Salvador algorithmic pipeline intact: cleanup, spanning-forest core, weighted MIDS gadget, greedy weighted IDS pass, edge repair, and redundancy pruning.
- Improved DIMACS parsing, compressed-file handling, deterministic CLI formatting, and generated DIMACS edge counts.
- Clarified documentation to distinguish implemented guarantees from conjectural approximation-ratio claims.
- Added regression smoke tests and a GitHub Actions test workflow.
- Updated packaging metadata and build configuration.
