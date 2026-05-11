"""
Full C. elegans Figure 14.

Main figure: representation panels and clustered SC/FC/FCV summary panels.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import pearsonr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "figure14_celegans"
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(FIGURES_DIR))

import figure_style as fs
import figure14_celegans_combined_horizontal as summary
import figure14_celegans_representation_timeseries as rep


OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure14_celegans_full_combined_FINAL.png"
SC_MEASURE_TABLE = BASE / "results" / "figure14_celegans_sc_cell_measures_full297_subset122.csv"
FC_MEASURE_TABLE = BASE / "results" / "figure14_celegans_fc_spontaneous_5measure_summary.csv"
REPRESENTATION_CACHE = BASE / "results" / "figure14_celegans_representation_examples.npz"
REPRESENTATION_META = BASE / "results" / "figure14_celegans_representation_examples.json"

FINAL_CLUSTER_ORDER = ["M1", "M2", "M4", "M3", "M5"]
PANEL_GHI_LABEL_X = -0.33
PANEL_GHI_LABEL_Y = 1.08


def configure_shared_style() -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    fs.set_paper_style()
    plt.rcParams.update(
        {
            "font.size": fs.TICK_FS_1COL,
            "axes.labelsize": fs.AXIS_LABEL_FS_1COL,
            "xtick.labelsize": fs.TICK_FS_1COL,
            "ytick.labelsize": fs.TICK_FS_1COL,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    summary.CLUSTER_ORDER = FINAL_CLUSTER_ORDER
    summary.CLUSTER_SORT_ORDER = {label: i for i, label in enumerate(FINAL_CLUSTER_ORDER)}
    rep.CLUSTER_ORDER = FINAL_CLUSTER_ORDER


def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def zscore_rows(values: np.ndarray) -> np.ndarray:
    out = values.copy().astype(float)
    for i in range(out.shape[0]):
        row = out[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        sd = np.nanstd(row[mask])
        if sd > 0:
            out[i, mask] = (row[mask] - np.nanmean(row[mask])) / sd
    return np.clip(out, -2.0, 2.0)


def orient_linkage_target_left(z_linkage: np.ndarray, labels: np.ndarray, target_label: str) -> np.ndarray:
    oriented = z_linkage.copy()
    n = len(labels)
    target_counts: dict[int, int] = {i: int(labels[i] == target_label) for i in range(n)}
    leaf_counts: dict[int, int] = {i: 1 for i in range(n)}
    for row_idx in range(oriented.shape[0]):
        node_id = n + row_idx
        left = int(oriented[row_idx, 0])
        right = int(oriented[row_idx, 1])
        left_frac = target_counts[left] / max(leaf_counts[left], 1)
        right_frac = target_counts[right] / max(leaf_counts[right], 1)
        if target_counts[right] > target_counts[left] or (
            target_counts[right] == target_counts[left] and right_frac > left_frac
        ):
            oriented[row_idx, 0], oriented[row_idx, 1] = oriented[row_idx, 1], oriented[row_idx, 0]
            left, right = right, left
        target_counts[node_id] = target_counts[left] + target_counts[right]
        leaf_counts[node_id] = leaf_counts[left] + leaf_counts[right]
    return oriented


def add_scatter_panel(ax, neurons: pd.DataFrame, label: str) -> None:
    colors = neurons["cluster_label"].map(summary.CLUSTER_COLORS).fillna("#8c8c8c")
    sizes = 13 + 3 * np.sqrt(neurons["n_recordings"].to_numpy())
    ax.scatter(
        neurons["PostDCA"],
        neurons["FCV_z"],
        s=sizes,
        c=colors,
        alpha=0.76,
        edgecolor="white",
        linewidth=0.35,
    )
    coef = np.polyfit(neurons["PostDCA"], neurons["FCV_z"], 1)
    xx = np.linspace(neurons["PostDCA"].min(), neurons["PostDCA"].max(), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.0)
    r, p = pearsonr(neurons["PostDCA"], neurons["FCV_z"])
    ax.text(0.05, 0.95, f"r={r:.2f}\np={p:.2g}", transform=ax.transAxes, va="top", fontsize=7)
    ax.axhline(0, color="#777777", lw=0.6, alpha=0.35, zorder=0)
    ax.axvline(0, color="#777777", lw=0.6, alpha=0.35, zorder=0)
    ax.set_xlabel("Post-DCA")
    ax.set_ylabel("zFCV")
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=3.2, width=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    add_panel_label(ax, label, x=PANEL_GHI_LABEL_X, y=PANEL_GHI_LABEL_Y)


def load_representation_examples() -> tuple[str, dict, dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    if not REPRESENTATION_CACHE.exists() or not REPRESENTATION_META.exists():
        raise FileNotFoundError(
            "Missing bundled C. elegans representation cache. "
            f"Expected {REPRESENTATION_CACHE} and {REPRESENTATION_META}."
        )
    arrays = np.load(REPRESENTATION_CACHE)
    meta = json.loads(REPRESENTATION_META.read_text())
    high_example = {
        "pair": meta["high"]["pair"],
        "time_s": arrays["high_time_s"],
        "traces": arrays["high_traces"],
        "corr_t": arrays["high_corr_t"],
        "corr_values": arrays["high_corr_values"],
    }
    low_example = {
        "pair": meta["low"]["pair"],
        "time_s": arrays["low_time_s"],
        "traces": arrays["low_traces"],
        "corr_t": arrays["low_corr_t"],
        "corr_values": arrays["low_corr_values"],
    }
    network_nodes, network_sc = rep.load_network_data()
    return "bundled-cache", high_example["pair"], low_example["pair"], high_example, low_example, network_nodes, network_sc


def draw_representation_row(fig, subspec) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    _, _, _, high_example, low_example, network_nodes, network_sc = load_representation_examples()
    gs = subspec.subgridspec(
        2,
        3,
        width_ratios=[1.55, 1.0, 1.0],
        height_ratios=[1.35, 0.8],
        hspace=0.18,
        wspace=0.34,
    )
    ax_network = fig.add_subplot(gs[:, 0])
    ax_high_trace = fig.add_subplot(gs[0, 1])
    ax_low_trace = fig.add_subplot(gs[0, 2])
    ax_high_corr = fig.add_subplot(gs[1, 1])
    ax_low_corr = fig.add_subplot(gs[1, 2])

    rep.add_network_panel(ax_network, network_nodes, network_sc, high_example, low_example)
    rep.add_trace_panel(ax_high_trace, high_example, "high")
    rep.add_trace_panel(ax_low_trace, low_example, "low")
    rep.add_corr_panel(ax_high_corr, high_example, "high")
    rep.add_corr_panel(ax_low_corr, low_example, "low")
    ax_high_trace.set_title("High FCV", fontsize=9, pad=2)
    ax_low_trace.set_title("Low FCV", fontsize=9, pad=2)

    add_panel_label(ax_network, "A", x=-0.06, y=1.02)
    add_panel_label(ax_high_trace, "B", x=-0.12, y=1.02)
    add_panel_label(ax_low_trace, "C", x=-0.12, y=1.02)
    return high_example, low_example, network_nodes, network_sc


def draw_summary_row(fig, subspec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neurons, scatter_neurons, recording_points = summary.load_panel_data()
    neuron_order = neurons["neuron"].tolist()
    boundaries = summary.cluster_boundaries(neurons)
    cluster_ticks, cluster_labels = summary.cluster_tick_positions(neurons)

    sc = np.log1p(summary.ordered_matrix(summary.SC_MATRIX, neuron_order).to_numpy())
    fc = summary.ordered_matrix(summary.FC_MATRIX_SPONT, neuron_order).to_numpy()
    fcv = summary.ordered_matrix(summary.FCV_MATRIX_SPONT, neuron_order).to_numpy()
    np.fill_diagonal(fc, 0.0)
    np.fill_diagonal(fcv, 0.0)
    fc_lim = np.nanpercentile(np.abs(fc), 98)

    gs = subspec.subgridspec(
        1,
        6,
        width_ratios=[1.0, 1.0, 1.0, 1.18, 1.18, 1.18],
        wspace=0.42,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

    matrix_cbars = []
    matrix_cbars.append(summary.add_matrix_panel(fig, axes[0], sc, "magma", "D", "SC", boundaries, cluster_ticks, cluster_labels))
    matrix_cbars.append(summary.add_matrix_panel(
        fig,
        axes[1],
        fc,
        "coolwarm",
        "E",
        "FC",
        boundaries,
        cluster_ticks,
        cluster_labels,
        vmin=-fc_lim,
        vmax=fc_lim,
        boundary_lw=0.55,
        boundary_bg_lw=1.2,
        boundary_color="#00843d",
    ))
    matrix_cbars.append(summary.add_matrix_panel(
        fig,
        axes[2],
        fcv,
        "coolwarm",
        "F",
        "FCV",
        boundaries,
        cluster_ticks,
        cluster_labels,
        boundary_lw=0.55,
        boundary_bg_lw=1.2,
        boundary_color="#00843d",
    ))
    add_scatter_panel(axes[3], scatter_neurons, "G")
    summary.add_box_panel(axes[4], recording_points, "FCV_z", "zFCV", "H", rasterized=True)
    axes[4].set_ylim(top=5.0)
    summary.add_box_panel(axes[5], neurons, "PostDCA", "Post-DCA", "I")

    fig.canvas.draw()
    for ax, cbar in zip(axes[:3], matrix_cbars):
        ax_pos = ax.get_position()
        cbar_pos = cbar.ax.get_position()
        cbar.ax.set_position([cbar_pos.x0 - 0.001, ax_pos.y0, cbar_pos.width, ax_pos.height])
    summary.match_panel_heights(axes[2], axes[3:])
    summary.shrink_panel_widths(axes[3:], scale=0.8)
    pos_g = axes[3].get_position()
    axes[3].set_position([pos_g.x0 + 0.0130, pos_g.y0, pos_g.width, pos_g.height])
    return neurons, scatter_neurons, recording_points


def cluster_boundaries_from_labels(labels: np.ndarray) -> list[float]:
    return [i - 0.5 for i in range(1, len(labels)) if labels[i] != labels[i - 1]]


def add_heatmap_outline(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.75)


def draw_cluster_color_bar(ax, ordered: pd.DataFrame) -> None:
    n = len(ordered)
    for i, cluster in enumerate(ordered["cluster_label"]):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, 0),
                1,
                1,
                color=summary.CLUSTER_COLORS.get(cluster, "#888888"),
                linewidth=0,
            )
        )
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_aligned_dendrogram(ax, linkage_matrix: np.ndarray, reverse: bool, n_leaves: int) -> None:
    """Draw scipy dendrogram using heatmap cell coordinates rather than 5,15,... coordinates."""
    dend = dendrogram(linkage_matrix, no_plot=True)
    max_d = 0.0
    for icoord, dcoord in zip(dend["icoord"], dend["dcoord"]):
        x = (np.asarray(icoord, dtype=float) - 5.0) / 10.0
        if reverse:
            x = (n_leaves - 1) - x
        y = np.asarray(dcoord, dtype=float)
        max_d = max(max_d, float(np.nanmax(y)))
        ax.plot(x, y, color="#333333", lw=0.45, solid_capstyle="round")
    ax.set_xlim(-0.5, n_leaves - 0.5)
    ax.set_ylim(0, max_d * 1.03 if max_d > 0 else 1.0)
    ax.axis("off")


def draw_measure_panel(
    fig,
    subspec,
    df: pd.DataFrame,
    measure_cols: list[str],
    measure_labels: list[str],
    panel_title: str,
    panel_label: str,
    leaf_order: np.ndarray,
    linkage_matrix: np.ndarray,
    dend_clusters: np.ndarray,
    reverse_dendrogram: bool,
    cmap: str = "PiYG_r",
) -> None:
    ordered = df.iloc[leaf_order].reset_index(drop=True)
    values = df[measure_cols].to_numpy(float).T
    z_values = zscore_rows(values)
    z_ordered = z_values[:, leaf_order]
    n = len(ordered)

    gs = subspec.subgridspec(
        3,
        2,
        height_ratios=[0.24, 0.055, 1.0],
        width_ratios=[1.0, 0.035],
        hspace=0.025,
        wspace=0.04,
    )
    ax_dend = fig.add_subplot(gs[0, 0])
    ax_cluster = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[2, 1])

    draw_aligned_dendrogram(ax_dend, linkage_matrix, reverse_dendrogram, n)

    draw_cluster_color_bar(ax_cluster, ordered)

    im = ax_heat.imshow(z_ordered, aspect="auto", cmap=cmap, vmin=-2, vmax=2, interpolation="nearest")

    ax_heat.set_yticks(np.arange(len(measure_labels)))
    ax_heat.set_yticklabels(measure_labels, fontsize=6.2)
    ax_heat.set_xticks(np.arange(n))
    ax_heat.set_xticklabels(ordered["neuron"], rotation=90, fontsize=2.55)
    for tick, cluster in zip(ax_heat.get_xticklabels(), ordered["cluster_label"]):
        tick.set_color(summary.CLUSTER_COLORS.get(cluster, "black"))
    ax_heat.tick_params(axis="x", length=0, pad=0.5)
    ax_heat.tick_params(axis="y", length=0, pad=2)
    ax_heat.set_xlim(-0.5, n - 0.5)
    ax_dend.set_title(panel_title, fontsize=8.0, pad=1.0)
    add_heatmap_outline(ax_heat)

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Z", fontsize=5.5, labelpad=1.2)
    cbar.ax.tick_params(labelsize=4.9, length=1.8, width=0.55)

    ax_dend.text(
        -0.045,
        1.03,
        panel_label,
        transform=ax_dend.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def draw_sc_measure_panel(fig, subspec) -> None:
    df = pd.read_csv(SC_MEASURE_TABLE)
    measure_cols = ["PostDCA", "PreDCA", "Log10_OutInput_degree", "OO_fraction"]
    measure_labels = ["Post-DCA", "Pre-DCA", r"$\log_{10}$ out/in degree", "Output-output motif"]
    values = df[measure_cols].to_numpy(float).T
    z_values = zscore_rows(values)
    z_cluster = np.nan_to_num(z_values, nan=0.0, posinf=0.0, neginf=0.0)
    z_linkage = linkage(z_cluster.T, method="ward")
    z_linkage = orient_linkage_target_left(z_linkage, df["cluster_label"].to_numpy(), "M1")
    leaf_order = np.asarray(dendrogram(z_linkage, no_plot=True)["leaves"], dtype=int)
    reverse_dendrogram = False
    leaf_clusters = df.iloc[leaf_order]["cluster_label"].to_numpy()
    m1_positions = np.flatnonzero(leaf_clusters == "M1")
    if len(m1_positions) and float(m1_positions.mean()) > (len(leaf_order) - 1) / 2:
        leaf_order = leaf_order[::-1]
        reverse_dendrogram = True
    dend_clusters = fcluster(z_linkage, t=7, criterion="maxclust")[leaf_order]
    draw_measure_panel(
        fig,
        subspec,
        df,
        measure_cols,
        measure_labels,
        "SC cell measures",
        "J",
        leaf_order,
        z_linkage,
        dend_clusters,
        reverse_dendrogram,
    )


def draw_fc_measure_panel(fig, subspec) -> None:
    df = pd.read_csv(FC_MEASURE_TABLE)
    measure_cols = ["FCS_z", "FCV_z", "Metastability", "NetTE_z", "NeighborNetTE_z"]
    measure_labels = ["z-FCS", "z-FCV", "Metasta-\nbility", "Net TE", "Neighbor\nNet TE"]
    values = df[measure_cols].to_numpy(float).T
    z_values = zscore_rows(values)
    z_cluster = np.nan_to_num(z_values, nan=0.0, posinf=0.0, neginf=0.0)
    z_linkage = linkage(z_cluster.T, method="ward")
    z_linkage = orient_linkage_target_left(z_linkage, df["cluster_label"].to_numpy(), "M1")
    leaf_order = np.asarray(dendrogram(z_linkage, no_plot=True)["leaves"], dtype=int)
    reverse_dendrogram = False
    leaf_clusters = df.iloc[leaf_order]["cluster_label"].to_numpy()
    m1_positions = np.flatnonzero(leaf_clusters == "M1")
    if len(m1_positions) and float(m1_positions.mean()) > (len(leaf_order) - 1) / 2:
        leaf_order = leaf_order[::-1]
        reverse_dendrogram = True
    dend_clusters = fcluster(z_linkage, t=6, criterion="maxclust")[leaf_order]
    draw_measure_panel(
        fig,
        subspec,
        df,
        measure_cols,
        measure_labels,
        "FC cell measures",
        "K",
        leaf_order,
        z_linkage,
        dend_clusters,
        reverse_dendrogram,
    )


def draw_measure_heatmap_row(fig, subspec) -> None:
    gs = subspec.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.12)
    draw_sc_measure_panel(fig, gs[0, 0])
    draw_fc_measure_panel(fig, gs[0, 1])


def main() -> None:
    configure_shared_style()
    fig = plt.figure(figsize=(16.0, 7.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.05], hspace=0.18)

    high_example, low_example, network_nodes, network_sc = draw_representation_row(fig, outer[0])
    neurons, scatter_neurons, recording_points = draw_summary_row(fig, outer[1])

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved PNG: {OUT_PNG}")
    print(f"Network nodes={len(network_nodes)}, edges={(network_sc.to_numpy() > 0).sum()}")
    print(
        "High pair: "
        f"{high_example['pair']['neuron_a']}-{high_example['pair']['neuron_b']}, "
        f"matrix FCV={high_example['pair']['matrix_fcv']:.3f}, "
        f"recording FCV={high_example['pair']['recording_fcv']:.3f}"
    )
    print(
        "Low pair: "
        f"{low_example['pair']['neuron_a']}-{low_example['pair']['neuron_b']}, "
        f"matrix FCV={low_example['pair']['matrix_fcv']:.3f}, "
        f"recording FCV={low_example['pair']['recording_fcv']:.3f}"
    )
    print(f"Summary neurons={len(neurons)}, E points={len(recording_points)}, D neurons={len(scatter_neurons)}")


if __name__ == "__main__":
    main()
