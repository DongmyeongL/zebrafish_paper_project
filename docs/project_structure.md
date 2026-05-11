# Paper Project Folder Structure

`paper_project/` is organized so that the figure scripts, required data, output figures, and calculation summaries can be moved together as a self-contained figure-generation package.

```text
paper_project/
├── data/          # Raw and precomputed data required to regenerate figures
├── figures/       # Python scripts that generate each figure
├── configs/       # Optional figure-specific configuration files
├── docs/          # Documentation for figure content and calculation methods
├── output/        # Generated figures and statistical result tables
├── README.md      # Main usage guide
├── requirements.txt
├── run_all.py     # Runs all packaged figure scripts
└── manifest.csv   # Mapping between figure scripts, data, and outputs
```

## Main Figure Scripts

The primary figure scripts are stored in `figures/`.

```text
figures/
├── figure9_clean.py
├── figure12_clean.py
├── figure13_clean.py
├── figure_fc_dynamics_final_fc_shift_summary.py
└── figure_sc_fc_final_overview.py
```

Additional helper scripts used by the main figures are also included.

```text
figures/
├── figure_regionwise_multivariate_coupling.py
├── figure_supply_sc_heatmap.py
├── plot_layer_asymmetric_epsilon_linear_model.py
└── figure_style.py
```

## Output Structure

Generated files are saved under `output/`.

```text
output/
├── pdf/      # Publication-ready PDF figures
├── png/      # PNG versions for quick inspection
└── stats/    # CSV files containing statistical results
```

For example, Figure 13 is organized as follows.

```text
figures/figure13_clean.py
docs/figure13.md
output/pdf/figure13_final.pdf
output/png/figure13_final.png
output/stats/figure13_stats.csv
```

## Running Figures

To regenerate all packaged figures, run this command from inside `paper_project/`.

```bash
python3.10 run_all.py
```

To regenerate only one figure, run the corresponding script.

```bash
python3.10 figures/figure13_clean.py
```

In short, `data/` contains the input materials, `figures/` contains the plotting and analysis code, `docs/` explains what each figure calculates, and `output/` stores the final figures and statistics.
