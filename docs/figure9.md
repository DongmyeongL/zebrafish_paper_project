# Figure 9

## Purpose

Figure 9 summarizes regional functional dynamics and directed interaction
features across anatomical divisions.  In this packaged version, the previous
coherence-time readout is replaced by regionwise metastability.

## Input Data

- `data/fig1_prism_D_FCS_FCV_bar.csv`: region-level spontaneous FCS, FCV, and
  their SEM values.
- `data/fig2_prism_A_FCS.csv`: division-formatted FCS values used for the FCS
  boxplot.
- `data/fig2_prism_B_FCV.csv`: division-formatted FCV values used for the FCV
  boxplot.
- `data/fc_dynamics_metastability_by_subject_region.csv`: precomputed
  subject-by-region metastability table.
- `data/region_community_io/subject_*/subject_*_causality.npz`: subject-level
  transfer-entropy matrices and region order.
- `data/region_community_io/subject_*/subject_*_net_te_drive_fc_neighbors.npz`:
  precomputed neighbor-drive summaries for each subject.

## Calculation

The script first assigns each region to `Tel`, `Di`, `Mes`, or `Hind` from the
region-name prefix.  FCS and FCV are read directly from
`fig1_prism_D_FCS_FCV_bar.csv` and z-scored across regions.

Metastability is read from
`fc_dynamics_metastability_by_subject_region.csv`.  The script averages
`RegionwiseMetastability` across subjects for each region, aligns the result to
the FCS/FCV region order, and z-scores the regional mean.

Net TE is computed from each subject causality file as the mean outgoing net
transfer entropy for each region.  Subject-level values are mapped back to the
shared region list, averaged across subjects, and z-scored.  Neighbor Net TE is
loaded from the precomputed `fc_neighbor_mean_drive` arrays, aligned in the same
way, averaged across subjects, and z-scored.

Panel A displays the aligned regional matrix:

- FCS
- FCV
- Metastability
- Net TE
- Neighbor Net TE

Panels C-G show division-level boxplots for FCS, FCV, metastability, Net TE,
and Neighbor Net TE.  Pairwise division comparisons use two-sided
Mann-Whitney U tests followed by Holm correction.  Panel B uses the same
regional feature set to show the hierarchical organization/ordering used in
the figure.

## Output

- `output/png/figure9_final.png`
- `output/pdf/figure9_final.pdf`
- `output/stats/figure9_stats.csv`: pairwise division Mann-Whitney U tests
  with uncorrected and Holm-corrected p values for panels C-G.

Run from `paper_project/`:

```bash
python3.10 figures/figure9_clean.py
```
