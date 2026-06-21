# Changelog

## v0.0.4

- Centralized package version metadata in `salvador.version` and updated all CLI version flags.
- Kept the existing Salvador algorithmic pipeline intact: cleanup, spanning-forest core, weighted MIDS gadget, greedy weighted IDS pass, edge repair, and redundancy pruning.
- Improved DIMACS parsing, compressed-file handling, deterministic CLI formatting, and generated DIMACS edge counts.
- Clarified documentation to distinguish implemented guarantees from conjectural approximation-ratio claims.
- Added regression smoke tests and a GitHub Actions test workflow.
- Updated packaging metadata and build configuration.
