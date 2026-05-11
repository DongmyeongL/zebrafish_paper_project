# Figure 12

## Purpose

Figure 12 summarizes structural-connectivity organization.  It combines a
region-level SC feature heatmap, representative network diagrams, and
division-level graph/DCA summaries.

## Input Data

- `data/sc_original_per_area_network_metrics.pkl`: per-region samples for
  clustering coefficient, modularity Q, and global efficiency.
- `data/total_selected_region_dac_data.npz`: raw pre-DCA and post-DCA arrays by
  region.
- `data/fig1_prism_D_FCS_FCV_bar.csv`: reference region list for alignment.
- `data/fig4_prism_C_degree_FCV.csv`: degree-related feature matched to the
  FCV table.
- `data/fig3_prism_A_PostDCA.csv`: division-formatted post-DCA values.
- `data/fig3_prism_B_PreDCA.csv`: division-formatted pre-DCA values.
- `data/temp_*_network_diagram.png`: representative diagrams for panel B.
- `data/network_*_network_diagarm.png`: representative pair diagrams for the
  lower network panel.

## Calculation

Panel A is produced by `figure_supply_sc_heatmap.py`.  For each region, the
script builds a structural feature vector containing clustering, modularity Q,
global efficiency, post-DCA, pre-DCA, and log10 out/in-degree.  Values that are
stored as repeated samples are averaged within region.  Each feature row is
z-scored across regions, missing values are filled by the feature mean, and the
regional columns are ordered by hierarchical clustering with Ward linkage.  The
display keeps telencephalic regions grouped first when reordering the linkage.

Panel B shows pre-rendered representative network diagrams.  The image files
are copied into `data/` so the figure can be redrawn without the original
network-generation scripts.  The lower diagrams are annotated with the paired
area names in the order shown in the panel.

Panels C-G summarize division-level structural values:

- C: clustering coefficient
- D: modularity Q
- E: global efficiency
- F: post-DCA
- G: pre-DCA

Division labels are `Tel`, `Di`, `Mes`, and `Hind`.  Pairwise comparisons use
two-sided Mann-Whitney U tests with Holm correction.  These panels are
descriptive summaries of the precomputed structural metrics, not recomputation
of the original graph construction.

## Output

- `output/png/figure12_final.png`
- `output/pdf/figure12_final.pdf`
- `output/stats/figure12_stats.csv`: pairwise division Mann-Whitney U tests
  with uncorrected and Holm-corrected p values for panels C-G.

Run from `paper_project/`:

```bash
python3.10 figures/figure12_clean.py
```
