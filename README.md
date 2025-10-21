# Unified Drive-Cycle Analysis (Public Snapshot)

This folder contains an snapshot of the scripts that power the unified drive-cycle analysis pipeline. Paths referring to the original environment were replaced with `${PROJECT_ROOT}` so the code can be reused in a different location.

## Folder Layout

- `unified_analysis_main_all_in_one.py` – orchestrates the end-to-end analysis (data loading, route segmentation, wavelets, AE variability checks, reporting).
- `unified_analysis_config.py` / `unified_analysis_core.py` – configuration and shared loading/processing utilities.
- `analysis_shared.py`, `route_variability_analysis.py`, `detailed_route_segment_analyzer.py`, `investigate_predictable_segments.py`, `section5_reporting.py` – supporting analyses (chaos metrics, detailed segments, predictable segments, Section 5 tables).
- `export_analyze_cycles.py`, `publication_visualizations.py` – export tooling and publication-quality figures.
- `unified_wavelet_module.py` – continuous wavelet transform analysis and visualisation helpers.
- `pytorch_parameter_analysis.py` – trains the autoencoder variability detector, saves model/scaler artifacts, and computes standard-cycle uplift metrics.

## Getting Started

1. Set `${PROJECT_ROOT}` to the location where you place data outputs.
2. Ensure required dependencies are installed (pandas, numpy, scipy, matplotlib, torch, etc.).
3. Run `unified_analysis_main_all_in_one.py` to generate reports/figures. Optional: provide a standard-cycle feature matrix (env `STANDARD_FEATURE_MATRIX`) so the PyTorch uplift step can compare standards vs. real driving.

## Notes

- The scripts expect preprocessed datasets (enriched drive logs, standard cycle features) organised similar to the original project. Adjust paths in `unified_analysis_config.py` to match your environment.
- Autoencoder outputs (model/scalers/uplift) will be written under `outputs/pytorch_parameter_analysis/` relative to `${PROJECT_ROOT}`.
- This snapshot strips proprietary or location-specific paths but preserves the algorithmic logic.
