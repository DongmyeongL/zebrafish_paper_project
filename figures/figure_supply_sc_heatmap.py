import os
import pickle
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

import figure_style as fs

warnings.filterwarnings("ignore", category=RuntimeWarning)

fs.set_paper_style()
plt.rcParams.update({
    "font.size": fs.AXIS_LABEL_FS_2COL,
    "axes.labelsize": fs.AXIS_LABEL_FS_2COL,
    "axes.titlesize": fs.AXIS_LABEL_FS_2COL,
    "xtick.labelsize": fs.TICK_FS_2COL,
    "ytick.labelsize": fs.TICK_FS_2COL,
})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data")
NETWORK_METRIC_FILE = os.path.join(DATA, "sc_original_per_area_network_metrics.pkl")
DAC_FILE = os.path.join(DATA, "total_selected_region_dac_data.npz")
REGION_FCV_FILE = os.path.join(DATA, "fig1_prism_D_FCS_FCV_bar.csv")
DEGREE_FCV_FILE = os.path.join(DATA, "fig4_prism_C_degree_FCV.csv")

DIVISION_LABELS = {
    2: "Tel",
    1: "Di",
    3: "Mes",
    0: "Hind",
}
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}


def _clean_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))


def _zscore(values):
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values * np.nan
    return (values - np.nanmean(values)) / std


def _fill_nan_with_mean(values):
    values = np.asarray(values, dtype=float)
    mean_val = np.nanmean(values)
    if not np.isfinite(mean_val):
        return values
    filled = values.copy()
    filled[~np.isfinite(filled)] = mean_val
    return filled


def _load_degree_feature_by_region():
    region_fcv_df = pd.read_csv(REGION_FCV_FILE)
    degree_fcv_df = pd.read_csv(DEGREE_FCV_FILE)

    fcv_to_region = {
        round(float(fcv), 8): region
        for region, fcv in zip(region_fcv_df["Region"], region_fcv_df["FCV"])
    }

    degree_by_region = {}
    for _, row in degree_fcv_df.iterrows():
        fcv_key = round(float(row["FCV"]), 8)
        region = fcv_to_region.get(fcv_key)
        if region is None:
            continue
        degree_by_region[region] = float(row["log10_OutIn_degree"])

    return degree_by_region


def _load_sc_feature_matrix():
    with open(NETWORK_METRIC_FILE, "rb") as f:
        metrics = pickle.load(f)

    dac_data = np.load(DAC_FILE, allow_pickle=True)
    pre_dca = dac_data["arr_1"]
    post_dca = dac_data["arr_2"]
    degree_by_region = _load_degree_feature_by_region()
    allowed_regions = set(pd.read_csv(REGION_FCV_FILE)["Region"].tolist())

    feature_sources = [
        ("Clustering", metrics["clustering_data"]),
        ("Modularity\nQ", metrics["q_data"]),
        ("Global\nEfficiency", metrics["Eglob_data"]),
        (r"$\mathrm{DCA}_{\mathrm{post}}$", post_dca),
        (r"$\mathrm{DCA}_{\mathrm{pre}}$", pre_dca),
        ("log10\n(Out/In-deg)", degree_by_region),
    ]

    selected_regions = []
    selected_divisions = []
    feature_rows = [[] for _ in feature_sources]

    for region_idx, division_idx in enumerate(fs.brain_division_list):
        if division_idx not in DIVISION_LABELS:
            continue

        region_name = fs.region[region_idx]
        if region_name not in allowed_regions:
            continue
        means = []
        for label, source in feature_sources:
            if isinstance(source, dict):
                value = source.get(region_name, np.nan)
            else:
                value = _clean_mean(source[region_idx])
            if label != "log10(Out/In-degree)" and not np.isfinite(value):
                means = None
                break
            means.append(value)
        if means is None:
            continue

        selected_regions.append(region_name)
        selected_divisions.append(DIVISION_LABELS[int(division_idx)])
        for row, mean_value in zip(feature_rows, means):
            row.append(mean_value)

    raw_matrix = np.asarray(feature_rows, dtype=float)
    z_matrix = np.asarray([
        _fill_nan_with_mean(_zscore(row))
        for row in raw_matrix
    ], dtype=float)
    feature_labels = [label for label, _ in feature_sources]
    return z_matrix, feature_labels, np.asarray(selected_regions), np.asarray(selected_divisions)


def _reorder_linkage_tel_first(z_linkage, divisions):
    """Flip dendrogram branches so Tel-containing groups appear on the left."""
    reordered = np.array(z_linkage, copy=True)
    n_leaves = len(divisions)
    tel_counts = {idx: int(divisions[idx] == "Tel") for idx in range(n_leaves)}

    for row_idx, row in enumerate(reordered):
        left = int(row[0])
        right = int(row[1])
        parent = n_leaves + row_idx

        if tel_counts.get(right, 0) > tel_counts.get(left, 0):
            reordered[row_idx, 0], reordered[row_idx, 1] = reordered[row_idx, 1], reordered[row_idx, 0]
            left, right = right, left

        tel_counts[parent] = tel_counts.get(left, 0) + tel_counts.get(right, 0)

    return reordered


def main():
    feature_matrix, feature_labels, regions, divisions = _load_sc_feature_matrix()
    n_regions = feature_matrix.shape[1]

    z_linkage = linkage(feature_matrix.T, method="ward")
    z_linkage = _reorder_linkage_tel_first(z_linkage, divisions)
    dend_info = dendrogram(z_linkage, no_plot=True)
    leaf_order = np.asarray(dend_info["leaves"])

    cluster_labels = fcluster(z_linkage, t=3, criterion="maxclust")
    cluster_ordered = cluster_labels[leaf_order]
    cluster_boundaries = [
        i - 0.5
        for i in range(1, len(cluster_ordered))
        if cluster_ordered[i] != cluster_ordered[i - 1]
    ]

    matrix_ordered = feature_matrix[:, leaf_order]
    regions_ordered = regions[leaf_order]
    divisions_ordered = divisions[leaf_order]

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.0, 0.18, 2.6],
        width_ratios=[1, 1, 1, 1, 0.72, 0.16],
        left=0.08,
        right=0.96,
        top=0.92,
        bottom=0.24,
        hspace=0.12,
        wspace=0.08,
    )

    ax_dendro = fig.add_subplot(gs[0, 0:4])
    ax_divbar = fig.add_subplot(gs[1, 0:4])
    ax_heat = fig.add_subplot(gs[2, 0:4])
    ax_cbar = fig.add_subplot(gs[2, 5])

    ax_dendro.text(
        -0.02,
        1.22,
        "A",
        transform=ax_dendro.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )

    dendrogram(
        z_linkage,
        ax=ax_dendro,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#333333",
    )
    dend_max_h = max(max(coords) for coords in dend_info["dcoord"])
    ax_dendro.set_xlim(0, n_regions * 10)
    ax_dendro.set_ylim(0, dend_max_h * 1.05)
    ax_dendro.axis("off")

    for i, division in enumerate(divisions_ordered):
        ax_divbar.add_patch(
            plt.Rectangle(
                (i - 0.5, 0),
                1,
                1,
                color=DIVISION_COLORS[division],
                linewidth=0,
            )
        )
    ax_divbar.set_xlim(-0.5, n_regions - 0.5)
    ax_divbar.set_ylim(0, 1)
    ax_divbar.set_xticks([])
    ax_divbar.set_yticks([])
    for spine in ax_divbar.spines.values():
        spine.set_visible(False)
    for boundary in cluster_boundaries:
        ax_divbar.axvline(boundary, color="black", linewidth=1.5, zorder=5)

    im = ax_heat.imshow(
        matrix_ordered,
        aspect="auto",
        cmap="PiYG_r",
        vmin=-2,
        vmax=2,
        interpolation="nearest",
    )
    ax_heat.set_xlim(-0.5, n_regions - 0.5)
    ax_heat.set_yticks(range(len(feature_labels)))
    ax_heat.set_yticklabels(feature_labels, fontsize=fs.TICK_FS_2COL, fontweight="bold")
    ax_heat.tick_params(axis="y", length=0, labelright=True, labelleft=False)
    ax_heat.set_xticks(range(n_regions))
    ax_heat.set_xticklabels(
        regions_ordered,
        rotation=90,
        fontsize=fs.TICK_FS_2COL - 2,
        ha="center",
        fontweight="bold",
    )
    ax_heat.tick_params(axis="x", length=2, width=0.5)
    for tick_label, division in zip(ax_heat.get_xticklabels(), divisions_ordered):
        tick_label.set_color(DIVISION_COLORS[division])
    for y in np.arange(-0.5, len(feature_labels), 1):
        ax_heat.axhline(y=y, color="white", linewidth=0.4)
    for boundary in cluster_boundaries:
        ax_heat.axvline(boundary, color="black", linewidth=1.5, zorder=5)
    for spine in ax_heat.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("black")

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.ax.set_title("Z-score", fontsize=fs.AXIS_LABEL_FS_2COL, fontweight="bold")
    cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)
    cbar.set_ticks([-2, -1, 0, 1, 2])

    legend_handles = [
        mpatches.Patch(color=DIVISION_COLORS[label], label=label)
        for label in ["Tel", "Di", "Mes", "Hind"]
    ]
    legend = ax_heat.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.26, 1.55),
        ncol=4,
        fontsize=fs.TICK_FS_2COL,
        frameon=False,
        title_fontsize=fs.AXIS_LABEL_FS_2COL,
    )
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.savefig(os.path.join(DATA, "figure_supply_sc_heatmap.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(DATA, "figure_supply_sc_heatmap.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
