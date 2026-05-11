# Figure 15 Drosophila Cross-Species Validation

This folder contains the final adult Drosophila Figure 15 workflow. The current
standard script is:

- `figure15_drosophila_final/code/figure15_drosophila_full_combined_FINAL.py`

The final output figure is:

- `figure15_drosophila_final/figures/figure15_drosophila_full_combined_FINAL.png`

## Final intended data sources

- Structural connectome: FlyWire whole-brain adult female Drosophila
  connectome, Dorkenwald et al. 2024. Region-level SC should be aggregated
  from directed neuron-neuron synapse counts using FlyWire neuropil labels.
- Functional data: Drosophila calcium imaging, preferably Brezovec et al. 2024
  DANDI raw data for de novo neuropil-level FC/FCV computation. Turner et al.
  2021 37-region resting-state data can be used as a first lightweight
  pipeline check, but it is not the final FCV analysis.

## Important difference from C. elegans

C. elegans used named neuron identity to align SC and FC. Drosophila public SC
and FC datasets come from different animals, so the final analysis must be done
at the population-average neuropil/region level.

## Expected final inputs

Place final processed inputs here:

- `data/processed/region_sc_edges.csv`
  - columns: `source_region,target_region,weight`
- `data/processed/region_fcv_recording_points.csv`
  - columns: `recording_id,source_region,target_region,FCV,FCV_z`
- optional `data/processed/region_fc_matrix.csv`
  - square region-by-region FC matrix

## Current smoke test

`code/drosophila_region_dca_pipeline.py --smoke-test-larva` uses the existing
local larval Drosophila eLife connectome files only to verify that the DCA and
Post-DCA code runs. These smoke-test outputs are not the final adult FlyWire
analysis.

## Current adult analysis decision

Use the adult Turner/Mann/Clandinin `branson_responses` calcium traces for
FC/FCV and FlyWire783 proofread neurons for directed SC. The final cleaned
workflow keeps Branson999-first FC/FCV outputs and FlyWire matched-region SC
summaries.

Current final settings:

- FC/FCV window: 30 frames;
- step: 8 frames;
- TE discretization: 4 quantile bins;
- structural score: synapse-count weighted Post-DCA+;
- SC figure unit: 41 matched side-aware Ito/Branson regions;
- panel J: all available Branson999 recording-level ROI FCV points
  (`n=8935`);
- panel K: Branson999 matched ROI-level Post-DCA+ points (`n=685`);
- L/M heatmaps are drawn directly inside the final combined figure from CSV
  results, not loaded from standalone PNGs.

Method records:

- `figure15_drosophila_final/DROSOPHILA_SC_ANALYSIS_METHOD.md`
- `figure15_drosophila_final/DROSOPHILA_FC_ANALYSIS_METHOD.md`
