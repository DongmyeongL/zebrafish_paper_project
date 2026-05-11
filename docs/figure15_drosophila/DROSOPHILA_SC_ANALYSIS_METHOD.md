# Drosophila SC Analysis Method

This document records the structural-connectome analysis used for the final
Drosophila figure workflow. The current standard follows
`figure15_drosophila_final/code/figure15_drosophila_full_combined_FINAL.py`.

## Final Analysis Unit

The structural-connectome analysis is based on FlyWire proofread neurons and
Ito left/right neuropil regions. The key distinction from the C. elegans
analysis is that Drosophila calcium imaging and FlyWire connectome neurons are
not neuron-identity matched across animals. Therefore, DCA is computed at the
FlyWire cell level and then summarized into region or side-aware block scores
for comparison with Branson calcium FCV.

## Primary Input Files

- FlyWire cell-level raw data:
  - `figure15_drosophila_final/data/raw/flywire_783/proofread_connections_783.feather`
  - `figure15_drosophila_final/data/raw/flywire_783/per_neuron_neuropil_count_pre_783.feather`
  - `figure15_drosophila_final/data/raw/flywire_783/per_neuron_neuropil_count_post_783.feather`
  - `figure15_drosophila_final/data/raw/flywire_783/proofread_root_ids_783.npy`
- Ito region order and grouping:
  - `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/ito_R_then_L_region_order_FINAL.csv`
- Final SC matrix and DCA/Post-DCA summaries:
  - `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/SC_flywire783_ito_R_then_L_matrix_FINAL.csv`
  - `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/DCA_PostDCA_ito48_FINAL.csv`
  - `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/cell_level_DCA_PostDCA_ito48_FINAL.csv`
  - `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/sideaware_cell_PostDCA_block_scores_FINAL.csv`

## Ito Region Mapping

Each FlyWire cell is assigned to an Ito L/R neuropil region using its
presynaptic and postsynaptic neuropil counts. For each cell, the region with the
largest total localization count is used as the dominant Ito region:

\[
r_i =
\arg\max_{r}
\left(
N^{\mathrm{pre}}_{i,r} + N^{\mathrm{post}}_{i,r}
\right).
\]

The current cell-level DCA workflow uses confidently mapped cells, with the
mapping-confidence threshold implemented in
`drosophila_flywire783_cell_ito48_dca_postdca.py`.

## Region-Level SC Matrix Used for Visualization

For the Ito48 directed SC matrix, a cell-level pre/post region-presence rule is
used. For each FlyWire cell, if it has at least one presynaptic site in Ito
region \(a\) and at least one postsynaptic site in Ito region \(b\), then:

\[
SC_{a,b} \leftarrow SC_{a,b} + 1.
\]

Thus, \(SC_{a,b}\) counts the number of cells spanning a presynaptic region
\(a\) and a postsynaptic region \(b\). It is a directed matrix:

- rows are presynaptic/source regions;
- columns are postsynaptic/target regions;
- diagonal entries are set to zero for visualization and downstream regional
  SC summaries when self-region effects should be excluded.

For the final combined figure, Ito48 regions are further collapsed to matched
Branson/FlyWire nodes when needed. In particular, mushroom-body subregions are
collapsed to side-specific MB nodes and AOTU is harmonized with the Branson
labeling used in the final figure script.

## Cell-Level DCA

DCA is computed within each Ito L/R region, not on the whole-brain region-level
SC matrix. For each region \(r\), cells assigned to that region are selected and
their within-region directed cell-to-cell SC submatrix is constructed:

\[
SC^{(r)}_{i,j},
\quad i,j \in r.
\]

For each region-specific submatrix, non-negative rank-1 directed coreness is
computed:

\[
(c_{\mathrm{out}}^{(r)}, c_{\mathrm{in}}^{(r)})
=
\arg\max_{c_{\mathrm{out}},c_{\mathrm{in}}}
c_{\mathrm{out}}^{\top} SC^{(r)} c_{\mathrm{in}},
\]

subject to:

\[
c_{\mathrm{out}},c_{\mathrm{in}} \ge 0,\qquad
\|c_{\mathrm{out}}\|_2=\|c_{\mathrm{in}}\|_2=1.
\]

The cell-level DCA is then:

\[
DCA_i = c_{\mathrm{out},i}^{(r)} - c_{\mathrm{in},i}^{(r)}.
\]

Positive DCA indicates output-like embedding within the local Ito region;
negative DCA indicates input-like embedding.

## Cell-Level Post-DCA

Post-DCA is computed after cell-level DCA has been estimated. For a source cell
\(i\), downstream target cells outside the source cell's own Ito L/R region are
used. With synapse-count weights \(SC_{i,j}\), the signed Post-DCA is:

\[
\mathrm{PostDCA}_i
=
\frac{
\sum_j SC_{i,j} DCA_j
}{
\sum_j SC_{i,j}
}.
\]

The final preferred Drosophila result uses positive downstream embedding:

\[
\mathrm{PostDCA}^{+}_i
=
\frac{
\sum_j SC_{i,j}\max(DCA_j,0)
}{
\sum_j SC_{i,j}
}.
\]

This definition asks whether a source cell preferentially projects to
downstream cells that are output-like within their own local Ito regions.

## Region and Side-Aware Block Summaries

For comparison with Branson calcium FCV, cell-level values are summarized into
side-aware region/block scores. The main score used in the final Drosophila
figure is:

\[
\overline{\mathrm{PostDCA}^{+}}_R
=
\frac{1}{|C_R|}
\sum_{i \in C_R}
\mathrm{PostDCA}^{+}_i,
\]

where \(C_R\) is the set of FlyWire cells assigned to the same side-aware
region or block. This score is stored as `mean_PostDCA_positive` or
`weighted_mean_PostDCA_positive` depending on the comparison table.

The final working combination recorded in
`figure15_drosophila_full_combined_FINAL.py` is:

- Branson FC/FCV window: 30 frames;
- step: 8 frames;
- SC: FlyWire783 cell-level weighted directed SC;
- Post-DCA score: mean positive downstream Post-DCA;
- FCV score: region FCV z-score excluding the same side-aware block.

## SC Heatmap Measures to Use

For the Drosophila SC-measure heatmap, the final figure method implies that the
preferred x-axis unit should match the region/block units used for FCV
comparison. The following four SC measures are computed and displayed:

1. Post-DCA+ summary;
2. Pre-DCA- summary, computed as upstream negative-DCA embedding;
3. \(\log_{10}\left((k_{\mathrm{out}}+1)/(k_{\mathrm{in}}+1)\right)\);
4. output-to-output motif fraction.

The Pre-DCA- score is defined as:

\[
\mathrm{PreDCA}^{-}_i
=
\frac{
\sum_j SC_{j,i}\max(-DCA_j,0)
}{
\sum_j SC_{j,i}
}.
\]

This makes the sign convention parallel to Post-DCA+: high Post-DCA+ indicates
stronger embedding toward output-like downstream targets, whereas high
Pre-DCA- indicates stronger embedding from input-like upstream sources.

## Final Matched41 Choice

For the SC-measure heatmap and downstream Figure 15-style comparisons, the
selected analysis unit is the **matched41** region set. This choice follows the
final combined Drosophila figure, where FlyWire SC measures are compared with
Branson999 FC/FCV only for regions that can be matched across both datasets.

The matched41 version is **not** computed by first estimating DCA on a single
whole-brain region-level SC matrix and then subsetting. Instead, it uses the
same cell-level method as the final Figure 15 analysis:

1. start from the FlyWire proofread cell pool;
2. map cells to dominant Ito L/R regions;
3. compute cell-level DCA separately within each Ito L/R region;
4. compute downstream cell-level Post-DCA/Post-DCA+ and upstream Pre-DCA using
   inter-region cell-to-cell connections;
5. summarize cell-level measures into Ito regions;
6. collapse/harmonize Ito labels to the 41 Branson/FlyWire matched regions;
7. retain only the matched41 regions for the final SC heatmap.

Thus, the matched41 table should be interpreted as a matched-region summary of
FlyWire cell-level structural embedding, not as a DCA computed directly on the
41-by-41 regional SC matrix.

The selected output files are:

- `figure15_drosophila_final/results/drosophila_flywire783_matched41_sc_cell_measures.csv`

The final combined figure redraws this heatmap directly and does not retain a
separate SC heatmap PNG. The Ito48 version was used as a QC/sensitivity
intermediate, but it is not retained in the cleaned final result folder.


## Current Saved Result

The current saved SC-measure result was generated by:

- `figure15_drosophila_final/code/figure15_drosophila_sc_measure_heatmap.py`

The final combined figure redraws the SC heatmap directly inside:

- `figure15_drosophila_final/code/figure15_drosophila_full_combined_FINAL.py`

It reads the matched41 SC-measure table:

- `figure15_drosophila_final/results/drosophila_flywire783_matched41_sc_cell_measures.csv`

The final matched41 SC heatmap uses the following four row-wise z-scored
measures:

1. `mean_PostDCA_positive`: mean downstream positive Post-DCA across cells in
   each matched region;
2. `mean_PreDCA_negative`: mean upstream negative-DCA embedding
   \(\mathrm{PreDCA}^{-}\);
3. `mean_Log10_OutInput_degree`: mean cell-level
   \(\log_{10}((k_{\mathrm{out}}+1)/(k_{\mathrm{in}}+1))\);
4. `mean_OO_fraction`: mean output-to-output motif fraction.

The final combined figure does not depend on a separately saved SC heatmap PNG.
The only required SC-side summary inputs retained for the final Figure 15
workflow are:

- `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/SC_flywire783_ito_R_then_L_matrix_FINAL.csv`
- `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/sideaware_cell_PostDCA_block_scores_FINAL.csv`
- `figure15_drosophila_final/final_results/SC_FC_FCV_current_standard/ito_R_then_L_region_order_FINAL.csv`
- `figure15_drosophila_final/results/drosophila_flywire783_matched41_sc_cell_measures.csv`
- `figure15_drosophila_final/results/drosophila_flywire783_sc_cell_measure_qc.csv`

## Final Figure 15 Structural Panels

In `figure15_drosophila_full_combined_FINAL.py`, panel A uses the current
matched41 block-level `weighted_mean_PostDCA_positive` values from:

- `figure15_drosophila_final/final_results/Branson999_full_FC_FCV/Branson999_full_ROI_5measure_sidekey_summary_w30_step8_FINAL.csv`

This ensures the horizontal network embedding matches the current final
Branson999-first FC comparison table. The network contains 41 matched
side-aware regions representing 685 Branson999 ROIs.

Panel K uses Branson999 matched ROI-level Post-DCA+ points from:

- `figure15_drosophila_final/final_results/Branson999_full_FC_FCV/Branson999_w30_weighted_vs_binary_PostDCApositive_scores_FINAL.csv`

The displayed K panel therefore uses 685 ROI-level points grouped by major
anatomical division, rather than only 41 block-average points. The current
Kruskal-Wallis result for this panel is:

\[
H=295.10,\qquad p=1.13\times10^{-61},\qquad n=685.
\]

Point-count labels are intentionally not shown above groups in the final
figure.

The current QC values are:

- confident FlyWire cells used: 119,866;
- inter-region cell-to-cell edges after thresholding: 735,710;
- synapse threshold: `syn_count >= 5`;
- Ito48 regions: 48;
- matched Figure 15 regions: 41;
- nonzero Ito48 SC region pairs: 2,206;
- Ito48 SC asymmetry \(L_1\): 62,284.

The final matched41 table contains no missing values for the four displayed SC
measures. The strongest matched41 Post-DCA+ regions in the current result are
`MB_L`, `AL_R`, `EPA_L`, and `MB_R`, with `MB_L`, `AL_R`, and `MB_R` being the
main high-cell-count regions among these.

## Relevant Code

- Current final figure:
  - `figure15_drosophila_final/code/figure15_drosophila_full_combined_FINAL.py`
- Ito48 region-presence SC matrix:
  - `figure15_drosophila_final/code/drosophila_flywire783_cell_prepost_presence_sc_ito48.py`
- Cell-level within-region DCA/Post-DCA:
  - `figure15_drosophila_final/code/drosophila_flywire783_cell_ito48_dca_postdca.py`
- Weighted versus binary SC comparison:
  - `figure15_drosophila_final/code/drosophila_branson999_w30_weighted_vs_binary_postdca.py`
