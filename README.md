# Zebrafish Paper Figure Project

This repository is a self-contained figure-reproduction package. Cloning the
GitHub repository should be enough to regenerate and inspect the bundled paper
figures without access to the original working directory.

The included data are the precomputed tables, matrices, and compact example
recordings needed by the plotting scripts. Very large upstream raw archives are
not required for the default reproduction path.

## Quick Start

```bash
git clone https://github.com/DongmyeongL/zebrafish_paper_project.git
cd zebrafish_paper_project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python figures/figure14_celegans_full_combined_FINAL.py
python figures/figure15_drosophila_full_combined_FINAL.py
```

## Layout

```text
paper_project/
├── data/        # Bundled raw examples and precomputed data required by scripts
├── figures/     # Plotting scripts and shared plotting helpers
├── configs/     # Lightweight figure metadata
├── docs/        # Figure notes and method documentation
└── output/      # Regenerated PNG/PDF files and statistics tables
```

Cross-species figure data and documentation are organized as:

```text
data/figure14_celegans/
data/figure15_drosophila/
docs/figure14_celegans/
docs/figure15_drosophila/
```

## Regenerate Figures

Run from the `paper_project` directory:

```bash
python figures/figure9_clean.py
python figures/figure12_clean.py
python figures/figure13_clean.py
python figures/figure14_clean.py
python figures/figure14_celegans_full_combined_FINAL.py
python figures/figure15_drosophila_full_combined_FINAL.py
python figures/figure_fc_dynamics_final_fc_shift_summary.py
python figures/figure_sc_fc_final_overview.py
python figures/figure_supply_0.py
python figures/figure_supply_1.py
python figures/figure_supply_2.py
python figures/figure_supply_5.py
python figures/figure_supply_10.py
python figures/figure_supply_11.py
python figures/figure_supply_13.py
python figures/figure_supply_14.py
python figures/figure_supply_15.py
python figures/figure_supply_16.py
```

Or run all figures:

```bash
python run_all.py
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

The scripts use only paths inside this repository. External absolute paths from
the original working directory were replaced by package-relative paths.

For the C. elegans full figure, the original WormWideWeb archive is large, so
the repository includes the compact high/low example traces needed to reproduce
the final panel. For the Drosophila full figure, the repository includes the
precomputed final result tables and the two example Branson recordings used by
the trace panels.

For supplementary figure panels that originally depended on large raw zebrafish
simulation or activity archives, the repository includes compact precomputed
inputs under `data/` so the public scripts do not require the original working
directory.

Expected cross-species outputs:

```text
output/png/figure14_celegans_full_combined_FINAL.png
output/png/figure15_drosophila_full_combined_FINAL.png
output/pdf/figure15_drosophila_full_combined_FINAL.pdf
output/png/figure_supply_*.png
output/pdf/figure_supply_*.pdf
```
