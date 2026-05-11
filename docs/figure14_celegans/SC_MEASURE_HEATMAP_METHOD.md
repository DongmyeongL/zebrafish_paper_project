# C. elegans SC Measure Heatmap Method

This document records the finalized structural-connectivity measure heatmap
analysis for C. elegans. The analysis adapts the zebrafish Figure 12A heatmap
style to identified C. elegans neurons.

## Analysis Unit and Data

- Analysis unit: identified neuron/cell.
- Measure calculation set: full C. elegans chemical synapse SC containing 297
  neurons.
- Final plotted neuron set: SC/FC common neuron subset used in the final
  C. elegans analysis.
- Final number of plotted neurons: 122.
- Structural connectivity: directed chemical synapse matrix.
- Rows of the SC matrix are source neurons and columns are target neurons.
- Full SC input used for final measure calculation:
  `figure14_celegans_final/data/herm_full_edgelist.csv`
- The 122-neuron SC/FC common subset and display order are taken from:
  `figure14_celegans_final/matrices/figure14_celegans_w60_nozero_matrix_neuron_order_nrec3_spontaneous.csv`
- Neuron order input:
  `figure14_celegans_final/matrices/figure14_celegans_w60_nozero_matrix_neuron_order_nrec3_spontaneous.csv`
- Cluster input:
  `figure14_celegans_final/matrices/figure14_celegans_w60_nozero_SC_modularity_sohn2011_clusters_k5_nrec3_spontaneous.csv`

## Final Analysis Choice

The final accepted SC heatmap uses a global-then-subset strategy:

1. Compute all SC measures on the full 297-neuron C. elegans directed chemical
   synapse matrix.
2. Extract the 122 neurons that are also available in the final FC analysis.
3. Z-score and plot the measures across this 122-neuron subset.

This preserves each neuron's global structural role in the complete C. elegans
connectome, while still restricting the displayed neurons to the SC/FC common
set. The earlier 122-neuron subnetwork calculation is retained only as a
comparison.

## Diagonal Handling

Self-connections are removed from the full 297-neuron SC before computing the
final measures:

\[
SC_{ii}=0.
\]

The diagonal-free full-297 calculation is the final accepted analysis.

## Directed Coreness and DCA

Directed coreness is computed by solving a non-negative rank-1 directed
embedding problem:

\[
(c_{\mathrm{out}}, c_{\mathrm{in}})
=
\arg\max_{c_{\mathrm{out}},c_{\mathrm{in}}}
c_{\mathrm{out}}^\top SC c_{\mathrm{in}},
\]

subject to:

\[
c_{\mathrm{out}},c_{\mathrm{in}}\geq0,\quad
\|c_{\mathrm{out}}\|_2=\|c_{\mathrm{in}}\|_2=1.
\]

The implementation uses an iterative power-method update:

\[
c_{\mathrm{out}} \leftarrow SC c_{\mathrm{in}},
\quad
c_{\mathrm{in}} \leftarrow SC^\top c_{\mathrm{out}},
\]

with non-negative clipping and unit-norm normalization after each update.

Directed coreness asymmetry is:

\[
DCA_i = c_{\mathrm{out},i} - c_{\mathrm{in},i}.
\]

Positive DCA indicates relatively output-like directed embedding, whereas
negative DCA indicates relatively input-like directed embedding.

## Post-DCA

Post-DCA measures the DCA of downstream targets reached by neuron \(i\):

\[
PostDCA_i =
\frac{
\sum_j SC_{ij}DCA_j
}{
\sum_j SC_{ij}
}.
\]

If a neuron has no outgoing weight in the full 297-neuron SC, the value is
assigned missing.

## Pre-DCA

Pre-DCA measures the DCA of upstream source neurons projecting to neuron \(i\):

\[
PreDCA_i =
\frac{
\sum_j SC_{ji}DCA_j
}{
\sum_j SC_{ji}
}.
\]

If a neuron has no incoming weight in the full 297-neuron SC, the value is
assigned missing.

## Log10 Out/Input Degree

Binary in-degree and out-degree are computed after diagonal removal:

\[
k_i^{out}=\sum_j \mathbf{1}(SC_{ij}>0),
\quad
k_i^{in}=\sum_j \mathbf{1}(SC_{ji}>0).
\]

The out/input degree balance is:

\[
\log_{10}
\left(
\frac{k_i^{out}+1}{k_i^{in}+1}
\right).
\]

The pseudocount prevents undefined values when a degree is zero.

## Output-to-Output Motif Fraction

The output-to-output motif fraction follows the zebrafish motif convention:
for each outgoing edge from neuron \(i\), count whether both source and target
are output-like:

\[
OO_i =
\frac{
\sum_j \mathbf{1}(SC_{ij}>0)
\mathbf{1}(DCA_i>0)
\mathbf{1}(DCA_j>0)
}{
\sum_j \mathbf{1}(SC_{ij}>0)
}.
\]

If a neuron has no outgoing binary edges, the value is assigned missing. In the
final dataset, every analyzed neuron had at least one outgoing edge.

## Final Heatmap

The final heatmap contains four rows, where all raw measures are first computed
on the full 297-neuron SC and then subset to the 122 plotted neurons:

1. Post-DCA
2. Pre-DCA
3. Log10 Out/Input degree
4. Output-to-output motif fraction

Each measure row is z-scored across the 122 plotted neurons for visualization
and clipped to the plotting range:

\[
z_{m,i} =
\frac{x_{m,i}-\mu_m}{\sigma_m}.
\]

The displayed heatmap is clipped to \([-2,2]\). Raw values are saved in the
output CSV.

## Clustering and Visualization

The final figure follows the zebrafish Figure 12A style:

- top dendrogram from hierarchical clustering
- M1--M5 cluster color bar
- measure heatmap
- cluster legend
- neuron labels colored by M1--M5 cluster color

Hierarchical clustering is performed on the neuron-wise four-measure z-score
matrix using Ward linkage. The dendrogram orientation is adjusted so that M1
neurons appear mostly on the left. Cluster boundary lines are drawn using a
hierarchical cut that produces six visual blocks.

Heatmap borders are emphasized with a black outline.

## Final Output Files

- Code:
  `figure14_celegans_final/code/figure14_celegans_sc_measure_heatmap.py`
- Main full-297-to-subset PNG figure:
  `figure14_celegans_final/figures/figure14_celegans_sc_cell_measure_heatmap_full297_subset122.png`
- Main full-297-to-subset data table:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_full297_subset122.csv`
- Main full-297-to-subset QC table:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_full297_subset122_qc.csv`
- Comparison between 122-subnetwork and full-297-to-subset:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_122subnet_vs_full297_subset122_comparison.csv`
- Earlier 122-subnetwork PNG figure retained for comparison:
  `figure14_celegans_final/figures/figure14_celegans_sc_cell_measure_heatmap.png`
- Earlier 122-subnetwork data table retained for comparison:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures.csv`
- Earlier 122-subnetwork QC table retained for comparison:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_qc.csv`
- With-diagonal comparison:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_with_diagonal.csv`
- Diagonal-removal comparison:
  `figure14_celegans_final/results/figure14_celegans_sc_cell_measures_diagonal_comparison.csv`

## Final QC Summary

Full-297-to-subset calculation:

- Number of cells used for measure calculation: 297.
- Number of plotted subset cells: 122.

Earlier 122-subnetwork calculation retained for comparison:

- Number of cells: 122.
- Nonzero directed SC edges after diagonal removal: 1416.
- Total SC weight after diagonal removal: 9413.
- Original diagonal nonzero entries: 21.
- Original diagonal weight sum: 67.
- Final diagonal nonzero entries: 0.
- Final diagonal weight sum: 0.
- Zero out-degree neurons: 0.
- Zero in-degree neurons: 0.
- Missing Post-DCA values: 0.
- Missing Pre-DCA values: 0.
- Missing output-to-output motif values: 0.
