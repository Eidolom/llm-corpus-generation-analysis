# ROADMAP

## Milestones
- [x] Baseline pragmatic visualizer retained in src/analyzers/visualizer.py
- [x] Added dedicated idiomaticity heatmap visualizer
- [ ] Add CLI flags for input path, output names, and lemma selection
- [ ] Add unit-style validation script for schema and normalization checks

## Phase Status
- Current phase: Analysis tooling hardening
- Status: In progress

## Implementation Log
- 2026-03-30: Created src/analyzers/visualizer_idiomaticity_heatmap.py for normalized lemma x register literal/idiomatic heatmaps using outputs/thesis_textbook_data_filtered.csv.
- 2026-03-30: Added matrix CSV export and aggregate AI vs textbook idiomaticity gap printout for thesis reporting.
- 2026-03-30: Added src/analyzers/visualizer_aggregate_register_proportions.py to plot aggregate LITERAL vs IDIOMATIC proportions by register and export a register summary CSV.
- 2026-03-30: Fixed src/utils/compute_split_irr.py by installing scikit-learn, switching default IRR paths to outputs/, and adding robust schema/source-group inference for current key/annotation files.
