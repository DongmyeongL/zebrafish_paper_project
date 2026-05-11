# FC Dynamics Final FC Shift Summary

## Purpose

This figure summarizes stimulus-evoked FC variability and FC-state shifts, then
compares stimulus readouts with spontaneous FCV and structural post-DCA.

## Input Data

- `data/fc_dynamics_state_fc_shift_by_subject_region.csv`: precomputed
  subject-by-stimulus-by-region FC-shift table.
- `data/fc_dynamics_stimulus_fcv_by_subject_region.csv`: precomputed
  subject-by-stimulus-by-region FCV table.
- `data/fc_dynamics_separate_fcv_region_mean.csv`: region-level stimulus FCV
  summary from the separated readout analysis.
- `data/fig1_prism_D_FCS_FCV_bar.csv`: spontaneous FCS/FCV table; this script
  uses `FCV` as `SpontaneousFCV`.
- `data/total_selected_region_dac_data.npz`: raw post-DCA values by region.

## Calculation

The script focuses on stimulus indices 10-12.  FC shift itself is already
precomputed in `fc_dynamics_state_fc_shift_by_subject_region.csv`; it is defined
as the Euclidean distance between a region's stimulus FC profile and spot/basal
FC profile, excluding the diagonal:

```text
FCShift_i = sqrt(sum_j((FC_stimulus_ij - FC_spot_ij)^2))
```

Stimulus FCV is loaded from
`fc_dynamics_stimulus_fcv_by_subject_region.csv`.  For the panel-A heatmap, FCV
is z-scored within each subject, then averaged by `StimulusIndex` and `Region`.
The side bar shows each region's mean subject-z-scored FCV across stimuli.

Region-level summary values are computed as follows:

- `MeanFCShift`: mean FC shift across subjects and selected stimuli.
- `SEMFCShift`: SEM of FC shift across all subject/stimulus observations.
- `MeanStimulusFCV`: mean stimulus FCV across subjects and selected stimuli.
- `SEMStimulusFCV`: SEM of stimulus FCV across all subject/stimulus
  observations.
- `StdStimulusFCV`: standard deviation of FCV across stimulus indices for each
  region.
- `StdFCShift`: standard deviation of FC shift across stimulus indices for each
  region.

Spontaneous FCV is merged by region from `fig1_prism_D_FCS_FCV_bar.csv`.
Scatter panels use Pearson correlation to compare spontaneous FCV with mean
stimulus FCV and stimulus-dependent FCV modulation.  The division boxplot tests
whether telencephalic regions have larger `StdFCShift` than non-telencephalic
regions using a one-sided Mann-Whitney U test.

The script also writes a compact summary CSV containing the reported
correlations, p values, and top regions.

## Output

- `output/png/figure_fc_dynamics_final_fc_shift_summary.png`
- `output/pdf/figure_fc_dynamics_final_fc_shift_summary.pdf`
- `output/figure_fc_dynamics_final_fc_shift_summary.csv`
- `output/stats/figure_fc_dynamics_final_fc_shift_summary_stats.csv`:
  correlation statistics and top-region summaries.
- `output/stats/figure_fc_dynamics_final_fc_shift_summary_region_values.csv`:
  per-region values used in the plotted correlations.

Run from `paper_project/`:

```bash
python3.10 figures/figure_fc_dynamics_final_fc_shift_summary.py
```
