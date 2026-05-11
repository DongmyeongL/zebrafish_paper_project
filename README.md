# Paper Figure Project

This folder is a self-contained figure-reproduction package.  Downloading this
folder alone should be enough to regenerate the included paper figures.

## Layout

```text
paper_project/
├── data/        # Raw and precomputed data required by the figure scripts
├── figures/     # Plotting scripts and shared plotting helpers
├── configs/     # Lightweight figure metadata
├── docs/        # Short notes on figure content and calculations
└── output/      # Regenerated PNG/PDF files and statistics tables
```

## Regenerate Figures

Run from the `paper_project` directory:

```bash
python3.10 figures/figure9_clean.py
python3.10 figures/figure12_clean.py
python3.10 figures/figure13_clean.py
python3.10 figures/figure_fc_dynamics_final_fc_shift_summary.py
python3.10 figures/figure_sc_fc_final_overview.py
```

Or run all figures:

```bash
python3.10 run_all.py
```

Outputs are written to:

```text
output/png/
output/pdf/
output/stats/
```

`output/stats/` contains the statistical information generated while plotting:
Mann-Whitney/Holm tests, bootstrap differences, Pearson correlations, PLS
weights, cross-validated FCV predictions, and region-level summary tables where
available.

## Notes

The scripts use only paths inside this folder.  External absolute paths from the
original working directory were replaced by package-relative paths.
