from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CELEGANS_CODE = BASE_DIR
DROSOPHILA_CODE = BASE_DIR

OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_16.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_16.pdf"


def _load_module(path: Path, module_name: str):
    code_dir = str(path.parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


celegans = _load_module(
    CELEGANS_CODE / "figure14_celegans_full_combined_FINAL.py",
    "figure_supply_16_celegans_full",
)
drosophila = _load_module(
    DROSOPHILA_CODE / "figure15_drosophila_full_combined_FINAL.py",
    "figure_supply_16_drosophila_full",
)


def draw_celegans_sc_measure_panel(fig: plt.Figure, subspec, panel_label: str) -> None:
    df = pd.read_csv(celegans.SC_MEASURE_TABLE)
    measure_cols = ["PostDCA", "PreDCA", "Log10_OutInput_degree", "OO_fraction"]
    measure_labels = ["Post-DCA", "Pre-DCA", r"$\log_{10}$ out/in degree", "Output-output motif"]
    values = df[measure_cols].to_numpy(float).T
    z_values = celegans.zscore_rows(values)
    z_cluster = np.nan_to_num(z_values, nan=0.0, posinf=0.0, neginf=0.0)
    z_linkage = linkage(z_cluster.T, method="ward")
    z_linkage = celegans.orient_linkage_target_left(
        z_linkage,
        df["cluster_label"].to_numpy(),
        "M1",
    )
    leaf_order = np.asarray(dendrogram(z_linkage, no_plot=True)["leaves"], dtype=int)
    reverse_dendrogram = False
    leaf_clusters = df.iloc[leaf_order]["cluster_label"].to_numpy()
    m1_positions = np.flatnonzero(leaf_clusters == "M1")
    if len(m1_positions) and float(m1_positions.mean()) > (len(leaf_order) - 1) / 2:
        leaf_order = leaf_order[::-1]
        reverse_dendrogram = True
    dend_clusters = fcluster(z_linkage, t=7, criterion="maxclust")[leaf_order]
    celegans.draw_measure_panel(
        fig,
        subspec,
        df,
        measure_cols,
        measure_labels,
        "C. elegans SC cell measures",
        panel_label,
        leaf_order,
        z_linkage,
        dend_clusters,
        reverse_dendrogram,
    )


def draw_celegans_fc_measure_panel(fig: plt.Figure, subspec, panel_label: str) -> None:
    df = pd.read_csv(celegans.FC_MEASURE_TABLE)
    measure_cols = ["FCS_z", "FCV_z", "Metastability", "NetTE_z", "NeighborNetTE_z"]
    measure_labels = ["z-FCS", "z-FCV", "Metasta-\nbility", "Net TE", "Neighbor\nNet TE"]
    values = df[measure_cols].to_numpy(float).T
    z_values = celegans.zscore_rows(values)
    z_cluster = np.nan_to_num(z_values, nan=0.0, posinf=0.0, neginf=0.0)
    z_linkage = linkage(z_cluster.T, method="ward")
    z_linkage = celegans.orient_linkage_target_left(
        z_linkage,
        df["cluster_label"].to_numpy(),
        "M1",
    )
    leaf_order = np.asarray(dendrogram(z_linkage, no_plot=True)["leaves"], dtype=int)
    reverse_dendrogram = False
    leaf_clusters = df.iloc[leaf_order]["cluster_label"].to_numpy()
    m1_positions = np.flatnonzero(leaf_clusters == "M1")
    if len(m1_positions) and float(m1_positions.mean()) > (len(leaf_order) - 1) / 2:
        leaf_order = leaf_order[::-1]
        reverse_dendrogram = True
    dend_clusters = fcluster(z_linkage, t=6, criterion="maxclust")[leaf_order]
    celegans.draw_measure_panel(
        fig,
        subspec,
        df,
        measure_cols,
        measure_labels,
        "C. elegans FC cell measures",
        panel_label,
        leaf_order,
        z_linkage,
        dend_clusters,
        reverse_dendrogram,
    )


def draw_drosophila_sc_measure_panel(fig: plt.Figure, subspec, panel_label: str) -> None:
    sc = pd.read_csv(drosophila.SC_MEASURE_RESULTS)
    drosophila.draw_measure_heatmap_panel(
        fig,
        subspec,
        sc,
        drosophila.SC_MEASURE_COLS,
        drosophila.SC_MEASURE_LABELS,
        "region",
        "mean_PostDCA_positive",
        panel_label,
        "Drosophila FlyWire783 SC measures",
        "Drosophila regions",
    )


def draw_drosophila_fc_measure_panel(fig: plt.Figure, subspec, panel_label: str) -> None:
    fc = pd.read_csv(drosophila.FC5_SIDEKEY)
    roi_summary = pd.read_csv(drosophila.FC5_ROI_SUMMARY)
    sidekey_map = pd.read_csv(drosophila.SCATTER_SCORES)[
        ["roi", "label", "side_key"]
    ].drop_duplicates()
    raw_meta = (
        sidekey_map.merge(
            roi_summary[["roi", "label", "Metastability"]],
            on=["roi", "label"],
            how="left",
        )
        .groupby("side_key", as_index=False)
        .agg(Metastability=("Metastability", "mean"))
    )
    fc = fc.drop(columns=["Metastability"], errors="ignore").merge(raw_meta, on="side_key", how="left")
    drosophila.draw_measure_heatmap_panel(
        fig,
        subspec,
        fc,
        drosophila.FC_MEASURE_COLS,
        drosophila.FC_MEASURE_LABELS,
        "side_key",
        "FCV_z",
        panel_label,
        "Drosophila Branson999 FC measures",
        "Drosophila side-aware blocks",
    )


def main() -> None:
    celegans.configure_shared_style()
    drosophila.set_style()

    fig = plt.figure(figsize=(8.0, 10.8))
    gs = fig.add_gridspec(
        4,
        1,
        left=0.070,
        right=0.985,
        top=0.975,
        bottom=0.045,
        hspace=0.34,
    )

    draw_celegans_sc_measure_panel(fig, gs[0, 0], "A")
    draw_celegans_fc_measure_panel(fig, gs[1, 0], "B")
    draw_drosophila_sc_measure_panel(fig, gs[2, 0], "C")
    draw_drosophila_fc_measure_panel(fig, gs[3, 0], "D")

    fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
