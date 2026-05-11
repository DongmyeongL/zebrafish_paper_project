# Drosophila Analysis Decisions

Final working criteria decided during the analysis session.

## Calcium data

- Species: adult Drosophila melanogaster
- Dataset: Turner/Mann/Clandinin resting-state calcium data
- Source: Figshare, `Drosophila central brain connectivity`
- DOI: `10.6084/m9.figshare.13349282.v3`
- Downloaded archive:
  - `data/raw/turner_mann_clandinin/data_TurnerMannClandinin.tar.gz`
- Extracted response folders:
  - `data/raw/turner_mann_clandinin/data/ito_responses/`
  - `data/raw/turner_mann_clandinin/data/branson_responses/`

## Final FC/FCV unit

- Use `branson_responses`.
- Analysis unit is Branson atlas ROI, not single neurons.
- Use only the common Branson ROI set present in all recordings.
- Common ROI count: 586.
- Total recordings: 20.
- Sampling rate: 1.2 Hz.
- Recording length:
  - 18 recordings: 2000 timepoints, approximately 27.8 min.
  - 2 recordings: 4000 timepoints, approximately 55.6 min.
- Recording state: spontaneous / resting-state.

## Structural connectivity

Current SC candidates:

1. `branson295`
   - Source file: `data/raw/turner_mann_clandinin/StructuralMatrix_branson.csv`
   - Matrix size: 295 x 295.
   - DCA/Post-DCA already computed.
   - Needs final matching check against the 586 common Branson calcium ROI set.
   - For downstream SC clustering, DCA, and SC-FCV analyses, remove regions with
     zero total structural connectivity (`in_strength + out_strength = 0`).
     Currently removed isolated Branson regions: `MB_L_16`, `MB_L_4`.
   - After removing isolated regions, Branson SC clustering should use 3 clusters
     as the working block structure (`n = 293`; cluster sizes 128, 104, 61).

2. `turner36_weighted`
   - Source file: `data/raw/turner_mann_clandinin/data/connectome_connectivity/WeightedSynapseNumber_computed_20210114.pkl`
   - Matrix size: 36 x 36.
   - DCA/Post-DCA already computed.
   - Useful as coarse atlas-level validation.

3. FlyWire 783
   - Source: Zenodo `10.5281/zenodo.10676866`
   - Downloaded files:
     - `proofread_root_ids_783.npy`
     - `per_neuron_neuropil_count_pre_783.feather`
     - `per_neuron_neuropil_count_post_783.feather`
     - `proofread_connections_783.feather`
   - Requires later neuropil/ROI aggregation before direct use.

## Completed outputs

- Calcium recording summary:
  - `stats/adult_calcium_recording_summary.csv`
  - `stats/branson_response_roi_counts_by_recording.csv`
- SC/DCA outputs:
  - `stats/drosophila_adult_sc_dca_summary.csv`
  - `stats/drosophila_adult_turner36_weighted_dca_metrics.csv`
  - `stats/drosophila_adult_branson295_dca_metrics.csv`
  - `figures/drosophila_adult_turner36_weighted_sc_dca.png`
  - `figures/drosophila_adult_branson295_sc_dca.png`

## Final SC/DCA snapshot

The adult Drosophila Branson structural-connectivity-only analysis is frozen as
of 2026-05-04.

- Final SC rule:
  - Use `StructuralMatrix_branson.csv`.
  - Remove zero-total-strength regions before analysis:
    `MB_L_4`, `MB_L_16`.
  - Final SC node count: `n = 293`.
- Final SC-derived cluster structure:
  - Three directed-profile clusters with sizes `128`, `104`, `61`.
  - These clusters are SC-derived blocks, not canonical Drosophila brain classes.
- Final major brain block mapping:
  - Region names were mapped to broad functional/anatomical blocks:
    olfactory system, mushroom body, central complex, optic/lateral
    protocerebrum, superior protocerebrum, inferior/ventrolateral
    protocerebrum, accessory/midline regions.
  - No Branson DCA region remained unmapped after isolated-region removal.
- Final DCA results:
  - Top `Post-DCA`: `MB_R_4`.
  - Top `DCA`: `MB_R_2`.
  - Highest major-block mean `Post-DCA`: olfactory system.
- Final SC result copies:
  - `final_sc_results/SC1_branson295_sc_clustered_FINAL.pdf`
  - `final_sc_results/SC2_branson295_sc_major_block_ordered_FINAL.pdf`
  - `final_sc_results/SC3_branson295_dca_result_FINAL.pdf`
  - `final_sc_results/SC4_branson295_postdca_major_blocks_FINAL.pdf`
  - `final_sc_results/SC5_branson295_major_block_sc_graph_FINAL.pdf`

## Next calculation

Proceed with FC/FCV calculation using the 586 common Branson ROI set across all
20 spontaneous/resting-state recordings.
