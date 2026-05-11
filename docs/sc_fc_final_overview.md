# SC-FC Final Overview

## Purpose

This figure gives a compact overview of multivariate coupling between
functional dynamics and structural-connectivity features.  It also tests how
well structural features predict regional FCV.

## Input Data

- `data/fig1_prism_D_FCS_FCV_bar.csv`: region-level spontaneous FCS and FCV.
- `data/fc_dynamics_metastability_by_subject_region.csv`: subject-by-region
  metastability table.
- `data/region_community_io/subject_*/subject_*_causality.npz`: transfer
  entropy matrices and subject-specific region order.
- `data/region_community_io/subject_*/subject_*_net_te_drive_fc_neighbors.npz`:
  neighbor-drive summaries.
- `data/sc_original_per_area_network_metrics.pkl`: SC graph metrics.
- `data/total_selected_region_dac_data.npz`: pre-DCA and post-DCA arrays.
- `data/fig4_prism_C_degree_FCV.csv`: log10 out/in-degree feature.

## Calculation

Functional features are loaded by
`figures/figure_regionwise_multivariate_coupling.py`:

- FCS
- FCV
- metastability
- Net TE
- Neighbor TE

FCS and FCV come from `fig1_prism_D_FCS_FCV_bar.csv` and are z-scored across
regions.  Metastability is averaged across subjects for each region and
z-scored.  Net TE is computed as the mean outgoing net transfer entropy per
region, averaged across subjects, and z-scored.  Neighbor TE uses the
precomputed `fc_neighbor_mean_drive` arrays, averaged across subjects and
z-scored.

Structural features are loaded from the SC heatmap helper:

- clustering coefficient
- modularity Q
- global efficiency
- post-DCA
- pre-DCA
- log10 out/in-degree

These structural features are averaged by region where needed and z-scored
before multivariate analysis.  Functional and structural matrices are aligned
by the shared region names.

Panel A shows the pairwise Pearson correlation matrix between functional and
structural features.  Panels B-D fit a one-component `PLSCanonical` model after
standardizing both matrices with `StandardScaler`.  The reported score
correlation is the Pearson correlation between the first functional and
structural PLS scores; bar plots show the corresponding feature weights.

Panels E-F test SC-based FCV prediction.  FCV is predicted from the structural
feature matrix using a pipeline of `StandardScaler` plus ordinary
`LinearRegression`.  Predictions are generated with shuffled 5-fold
cross-validation (`KFold(n_splits=5, shuffle=True, random_state=0)`) and
summarized with Pearson correlation between observed and predicted FCV.  The
beta panel is fit once on all standardized regions to show the direction and
relative size of each structural coefficient.

## Output

- `output/png/figure_sc_fc_final_overview.png`
- `output/pdf/figure_sc_fc_final_overview.pdf`
- `output/stats/figure_sc_fc_final_overview_stats.csv`: PLS score coupling
  and cross-validated FCV prediction statistics.
- `output/stats/figure_sc_fc_final_overview_pairwise_correlations.csv`:
  pairwise Pearson correlations for panel A.
- `output/stats/figure_sc_fc_final_overview_weights.csv`: PLS weights and
  standardized FCV-prediction betas.
- `output/stats/figure_sc_fc_final_overview_fcv_predictions.csv`: observed,
  predicted, and residual FCV values for each region.

Run from `paper_project/`:

```bash
python3.10 figures/figure_sc_fc_final_overview.py
```
