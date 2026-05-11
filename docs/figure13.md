# Figure 13

## Purpose

Figure 13 tests outgoing-cascade effects using directional asymmetry.  Panels
A-C use a compact layer linear model, and panel D uses the large-scale
connectome-constrained model.

## Input Data

- `data/figure13/layer_asymmetric_epsilon_linear_data.npz`: precomputed layer
  linear-model results across directional-asymmetry epsilon values.
- `data/figure13/tfigure5_compare_xy_scatter_data.pkl`: base large-scale model
  FCS/FCV samples.
- `data/figure13/wsim_fc_null_p_sp_in_fc_mean_std_data.pkl`: Null-In
  large-scale model samples.
- `data/figure13/wsim_fc_null_p_sp_out_fc_mean_std_data.pkl`: Null-Out
  large-scale model samples.
- `data/figure13/figure1_whole_brain_fc_mean_std_data.npz`: empirical region
  list used to select model regions.

## Calculation

Panel A is drawn from the helper script
`figures/plot_layer_asymmetric_epsilon_linear_model.py`.  It shows the layer
linear-model network used to express directional asymmetry.  The asymmetry
parameter epsilon changes the balance between feedforward and feedback
coupling, which is used here as a simple representation of outgoing-cascade
effects.

Panels B and C load precomputed FCS and FCV responses from
`layer_asymmetric_epsilon_linear_data.npz`.  For each epsilon, the script plots
the mean response with SEM.  These panels are labeled as the `Layer linear
model`.

Panel D is labeled as the `Large-scale model`.  The script loads base,
Null-In, and Null-Out simulation outputs, keeps the empirical model-region
selection from `figure1_whole_brain_fc_mean_std_data.npz`, and compares FCS and
FCV across conditions.  Pairwise condition differences are bootstrapped with
10,000 resamples:

- Null-Out minus Base
- Null-In minus Base
- Null-Out minus Null-In

Two-sided bootstrap p values are estimated from the sign of the resampled
differences and then Holm-corrected.  Kruskal-Wallis tests are also computed
across the three conditions as omnibus checks.  The plotted violins show the
bootstrap distributions of pairwise differences.

## Output

- `output/png/figure13_final.png`
- `output/pdf/figure13_final.pdf`
- `output/stats/figure13_stats.csv`: Kruskal-Wallis omnibus tests and
  bootstrap mean-difference tests for large-scale FCS and FCV in panel D.

Run from `paper_project/`:

```bash
python3.10 figures/figure13_clean.py
```
