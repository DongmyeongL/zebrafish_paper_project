# C. elegans FC Measure Heatmap Method

This document records the agreed analysis plan for the C. elegans FC-based
measure heatmap. The goal is to match the logic of zebrafish Figure 9A while
using identified C. elegans neurons as the analysis units.

## Analysis Unit and Data

- Analysis unit: identified neuron/cell.
- Main neuron set: SC/FC common neuron subset used in the final C. elegans
  analysis, currently 122 neurons.
- Activity source: WormWideWeb / Atanas and Kim 2023 calcium activity.
- Epoch: spontaneous section only. Heat-evoked sections are not used.
- Repetition unit: compute measures separately for each spontaneous recording
  section, then average each measure across recordings for each neuron.
- Final visualization style: Figure 9A heatmap style with a dendrogram, cluster
  color bar, five measure rows, and M1--M5 cluster legend.

## Recording-Level Preprocessing

For each recording section:

1. Select labeled neurons that are present in the SC/FC common neuron set.
2. Extract only spontaneous activity.
3. Z-score each neuron activity trace within the recording.
4. Exclude recordings with too few valid neurons or too few time points for the
   requested window/TE analysis.

## Script Organization

The FC analysis is intentionally split into separate calculation scripts and a
final combination script:

1. Basic spontaneous FC measures:
   `figure14_celegans_final/code/figure14_celegans_fc_spontaneous_basic_measures.py`
2. Spontaneous transfer-entropy measures:
   `figure14_celegans_final/code/figure14_celegans_fc_spontaneous_te_measures.py`
3. Final five-measure table and heatmap:
   `figure14_celegans_final/code/figure14_celegans_fc_spontaneous_combine_heatmap.py`

This separation keeps the expensive TE calculation independent from the faster
FCS, FCV, and metastability calculations. The final heatmap is generated only
after the two measure tables are saved.

## Functional Connectivity Strength

For each recording, compute a pairwise FC matrix from spontaneous traces. For
neuron \(i\),

\[
FCS_i = \left< |FC_{ij}| \right>_{j \ne i}.
\]

The FCS vector is z-scored across neurons within the recording. The final
z-FCS value for each neuron is the mean across recordings.

## Functional Connectivity Variability

Sliding-window FC is computed using the C. elegans final-analysis window
settings unless explicitly changed:

- window = 60 frames
- step = 15 frames

For each edge,

\[
FCV_{ij} = \mathrm{std}_w\left(FC^{(w)}_{ij}\right),
\]

where \(w\) indexes sliding windows. For neuron \(i\),

\[
FCV_i = \left< FCV_{ij} \right>_{j \ne i}.
\]

The FCV vector is z-scored across neurons within each recording. The final
z-FCV value for each neuron is the mean across recordings.

## Metastability

Metastability follows the zebrafish phase-based definition, adapted to
identified neurons.

For each z-scored trace \(x_i(t)\), instantaneous phase is obtained with the
Hilbert transform:

\[
\phi_i(t) = \arg\left(\mathrm{hilbert}(x_i(t))\right).
\]

Neuron-wise phase coordination is:

\[
R_i(t) =
\frac{1}{N-1}\sum_{j \ne i}
\cos\left(\phi_i(t)-\phi_j(t)\right).
\]

Metastability is the temporal variance of this coordination signal:

\[
M_i = \mathrm{Var}_t\left(R_i(t)\right).
\]

The metastability vector is z-scored across neurons within each recording. The
final value for each neuron is the mean across recordings.

## Net Transfer Entropy

NetTE follows the zebrafish entropy-based transfer entropy framework, adapted
from region/community units to identified neurons.

For each recording:

1. Z-score neuron traces.
2. Discretize each trace into quantile-based bins.
3. Use lag = 1 frame.
4. Main analysis uses 4 quantile-based bins. Three-bin TE was also computed as
   a robustness check and gave very similar TE rankings.

Transfer entropy from source \(X\) to target \(Y\) is:

\[
TE_{X \rightarrow Y}
=
H(Y_{t+1},Y_t)-H(Y_t)
-H(Y_{t+1},Y_t,X_t)+H(Y_t,X_t),
\]

where \(H(\cdot)\) is Shannon entropy in bits.

For the current C. elegans heatmap run, NetTE drive uses the raw entropy-based
TE asymmetry matrix without TE-pair permutation filtering. This matches the
region-level zebrafish Figure 9A aggregation step, where the plotted NetTE row
uses mean net directed drive. Permutation-filtered TE can be added as a
robustness analysis, but it is not the current primary heatmap value.

Directional asymmetry is:

\[
NetTE_{ij}=TE_{i\rightarrow j}-TE_{j\rightarrow i}.
\]

Neuron-level NetTE drive is:

\[
NetTE_i = \left< NetTE_{ij} \right>_{j \ne i}.
\]

The NetTE vector is z-scored across neurons within each recording. The final
value for each neuron is the mean across recordings.

## Neighbor NetTE

Neighbor NetTE follows the zebrafish FC-neighbor drive definition.

For each recording:

1. Compute the pairwise FC matrix.
2. Compute Pearson correlation p-values.
3. Apply Benjamini--Hochberg FDR correction.
4. Significant FC neighbors are defined at \(q < 0.001\).

The outgoing NetTE drive of neuron \(i\) is:

\[
D_i =
\frac{1}{N-1}
\sum_{j \ne i}
NetTE_{ij}.
\]

The FC-neighbor NetTE drive is:

\[
D_i^{\mathrm{neigh}}
=
\frac{1}{|\mathcal{N}_i|}
\sum_{j \in \mathcal{N}_i}
D_j,
\quad
\mathcal{N}_i =
\left\{j : j \ne i,\; FC_{ij}\ \mathrm{significant\ after\ BH-FDR}\right\}.
\]

Neurons with no significant FC neighbors are assigned missing values for that
recording. The neighbor NetTE vector is z-scored across valid neurons within
each recording. The final value for each neuron is the mean across recordings.

## Final Heatmap

The final heatmap contains five rows:

1. z-FCS
2. z-FCV
3. Metastability
4. Net TE
5. Neighbor Net TE

Rows are measure-wise z-scored for visualization. Columns are neurons. Neuron
ordering is determined by hierarchical clustering of the five-measure matrix,
following the Figure 9A style. The cluster color bar uses the final C. elegans
M1--M5 modularity clusters. The dendrogram branch orientation is mirrored so
that neurons with high \(z\)-FCV appear as far left as possible without changing
the hierarchical distances. Heatmap borders are emphasized with a black outline.

## Output Files

- Helper/plotting code:
  `figure14_celegans_final/code/figure14_celegans_fc_measure_heatmap.py`
- Basic recording-level table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_basic_recording_level.csv`
- Basic neuron-level table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_basic_summary.csv`
- Basic QC table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_basic_qc.csv`
- TE recording-level table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_te_recording_level_bins4.csv`
- TE neuron-level table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_te_summary_bins4.csv`
- TE QC table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_te_qc_bins4.csv`
- Final combined neuron-level table:
  `figure14_celegans_final/results/figure14_celegans_fc_spontaneous_5measure_summary.csv`
- Final PNG figure:
  `figure14_celegans_final/figures/figure14_celegans_fc_cell_measure_heatmap.png`

## QC Items to Save

- Number of recordings included.
- Number of valid neurons per recording.
- Number of time points per recording.
- Number of sliding windows per recording.
- Number of finite TE pairs.
- Number of significant FC-neighbor pairs after BH-FDR.
- Number of neurons with missing neighbor NetTE.
- TE bin count and lag.

## Completed Current Run

- Epoch: spontaneous only.
- Window/step for FCV: 60/15 frames.
- TE bins/lag: 4 bins, lag 1 frame.
- Valid spontaneous recordings: 18.
- Final neuron summary: 120 neurons.
- Missing values in the final five heatmap rows: none.
- Heatmap branch orientation: high \(z\)-FCV neurons are placed as far left as
  possible while preserving the Ward-linkage dendrogram.
- Three-bin versus four-bin TE robustness:
  \(r=0.967\) for NetTE_z and \(r=0.928\) for NeighborNetTE_z across neurons.
