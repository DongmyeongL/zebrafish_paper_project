"""
Horizontal C. elegans Figure 14 summary.

Panels:
  A-C: clustered SC, FC, and FCV matrices
  D: FCV_z versus Post-DCA
  E-F: cluster-wise FCV_z and Post-DCA box/strip plots
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, pearsonr
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "figure14_celegans"
sys.path.insert(0, str(PROJECT_ROOT / "figures"))

import figure_style as fs


ORDER = BASE / "matrices" / "figure14_celegans_w60_nozero_matrix_neuron_order_nrec3_spontaneous.csv"
CLUSTERS = BASE / "matrices" / "figure14_celegans_w60_nozero_SC_modularity_sohn2011_clusters_k5_nrec3_spontaneous.csv"
SC_MATRIX = BASE / "matrices" / "figure14_celegans_w60_nozero_SC_matrix_nrec3_spontaneous.csv"
FC_MATRIX = BASE / "matrices" / "figure14_celegans_w60_nozero_FC_matrix.csv"
FCV_MATRIX = BASE / "matrices" / "figure14_celegans_w60_nozero_FCV_matrix.csv"
FC_MATRIX_SPONT = BASE / "matrices" / "figure14_celegans_w60_nozero_FC_matrix_nrec3_spontaneous.csv"
FCV_MATRIX_SPONT = BASE / "matrices" / "figure14_celegans_w60_nozero_FCV_matrix_nrec3_spontaneous.csv"
RECORDING_POINTS_SPLIT = BASE / "results" / "figure14_celegans_w60_modularity_sohn2011_recording_points_k5_spont_heat_nrec3_spontaneous.csv"
POSTDCA_FULL297_SUBSET = BASE / "results" / "figure14_celegans_sc_cell_measures_full297_subset122.csv"
FC_SPONT_SUMMARY = BASE / "results" / "figure14_celegans_fc_spontaneous_5measure_summary.csv"
FC_SPONT_RECORDING = BASE / "results" / "figure14_celegans_fc_spontaneous_basic_recording_level.csv"

OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure14_celegans_combined_horizontal.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure14_celegans_combined_horizontal.pdf"

POSTDCA_ZERO_TOL = 1e-12
CLUSTER_ORDER = ["M1", "M2", "M5", "M3", "M4"]
CLUSTER_SORT_ORDER = {label: i for i, label in enumerate(CLUSTER_ORDER)}
CLUSTER_COLORS = {
    "M1": "#4e79a7",
    "M2": "#f28e2b",
    "M3": "#59a14f",
    "M4": "#e15759",
    "M5": "#b07aa1",
}
CLASS_COLORS = {
    "Sensory": "#1b9e77",
    "Interneuron": "#7570b3",
    "Motorneuron": "#d95f02",
}


def computable_activity_neurons() -> set[str]:
    fc = pd.read_csv(FC_MATRIX_SPONT, index_col=0)
    fcv = pd.read_csv(FCV_MATRIX_SPONT, index_col=0)
    fc_values = fc.to_numpy(float)
    fcv_values = fcv.to_numpy(float)
    np.fill_diagonal(fc_values, np.nan)
    np.fill_diagonal(fcv_values, np.nan)

    fc_ok = np.isfinite(fc_values).any(axis=0) | np.isfinite(fc_values).any(axis=1)
    fcv_ok = np.isfinite(fcv_values).any(axis=0) | np.isfinite(fcv_values).any(axis=1)
    return set(fc.index[fc_ok & fcv_ok])


def load_panel_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_activity_neurons = computable_activity_neurons()
    order = pd.read_csv(ORDER)
    clusters = pd.read_csv(CLUSTERS)
    postdca_full = pd.read_csv(POSTDCA_FULL297_SUBSET)[["neuron", "PostDCA"]].rename(columns={"PostDCA": "PostDCA_full297"})
    fc_summary = pd.read_csv(FC_SPONT_SUMMARY)[["neuron", "FCV_z"]].rename(columns={"FCV_z": "FCV_z_spont_fc"})
    neurons = order.merge(clusters, on="neuron", how="inner")
    neurons = neurons.merge(postdca_full, on="neuron", how="left")
    neurons = neurons.merge(fc_summary, on="neuron", how="left")
    neurons["PostDCA"] = neurons["PostDCA_full297"]
    neurons["FCV_z"] = neurons["FCV_z_spont_fc"]
    neurons = neurons.drop(columns=["PostDCA_full297", "FCV_z_spont_fc"])
    neurons = neurons[neurons["neuron"].isin(valid_activity_neurons)].copy()
    neurons["modularity_cluster_k5"] = neurons["modularity_cluster_k5"].astype(int)
    neurons["cluster_label"] = neurons["modularity_cluster_k5"].map(lambda x: f"M{x}")
    neurons = neurons.replace([np.inf, -np.inf], np.nan).dropna(subset=["PostDCA", "FCV_z"])
    neurons = neurons[neurons["PostDCA"].abs() > POSTDCA_ZERO_TOL].copy()
    neurons["cluster_sort_order"] = neurons["cluster_label"].map(CLUSTER_SORT_ORDER)
    neurons = neurons.sort_values(["cluster_sort_order", "PostDCA", "neuron"]).reset_index(drop=True)

    recording_points = pd.read_csv(FC_SPONT_RECORDING).replace([np.inf, -np.inf], np.nan)
    recording_points = recording_points.merge(
        order[["neuron", "cell_class"]],
        on="neuron",
        how="left",
    )
    recording_points = recording_points.merge(
        clusters[["neuron", "modularity_cluster_k5"]],
        on="neuron",
        how="left",
    )
    recording_points["modularity_cluster_k5"] = recording_points["modularity_cluster_k5"].astype("Int64")
    recording_points["cluster_label"] = recording_points["modularity_cluster_k5"].map(lambda x: f"M{int(x)}" if pd.notna(x) else np.nan)
    recording_points = recording_points[recording_points["phase"].eq("spontaneous")].copy()
    postdca_map = dict(zip(postdca_full["neuron"], postdca_full["PostDCA_full297"]))
    recording_points["PostDCA"] = recording_points["neuron"].map(postdca_map)
    recording_points = recording_points[recording_points["neuron"].isin(valid_activity_neurons)].copy()
    recording_points = recording_points.dropna(subset=["PostDCA", "FCV_z", "cluster_label"])
    recording_points = recording_points[recording_points["PostDCA"].abs() > POSTDCA_ZERO_TOL].copy()

    scatter_neurons = (
        recording_points.groupby(["neuron", "cluster_label", "modularity_cluster_k5", "cell_class"], as_index=False)
        .agg(PostDCA=("PostDCA", "first"), FCV_z=("FCV_z", "mean"), n_recordings=("uid", "nunique"))
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["PostDCA", "FCV_z"])
    )
    return neurons, scatter_neurons, recording_points


def ordered_matrix(path: Path, neuron_order: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    return matrix.loc[neuron_order, neuron_order]


def cluster_boundaries(neurons: pd.DataFrame) -> list[float]:
    boundaries = []
    last = None
    for i, label in enumerate(neurons["cluster_label"]):
        if last is not None and label != last:
            boundaries.append(i - 0.5)
        last = label
    return boundaries


def cluster_tick_positions(neurons: pd.DataFrame) -> tuple[list[float], list[str]]:
    ticks = []
    labels = []
    for label, group in neurons.groupby("cluster_label", sort=False):
        ticks.append((group.index.min() + group.index.max()) / 2)
        labels.append(label)
    return ticks, labels


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.08) -> None:
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


def add_matrix_panel(
    fig,
    ax,
    values: np.ndarray,
    cmap: str,
    label: str,
    title: str,
    boundaries: list[float],
    cluster_ticks: list[float],
    cluster_labels: list[str],
    vmin=None,
    vmax=None,
    boundary_lw: float = 0.35,
    boundary_bg_lw: float = 0.9,
    boundary_color: str = "#202020",
) -> object:
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    ax.set_title(title, pad=4, fontsize=8)
    ax.set_xticks(cluster_ticks)
    ax.set_yticks(cluster_ticks)
    ax.set_xticklabels(cluster_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(cluster_labels, fontsize=7)
    #ax.set_xlabel("Cluster name", labelpad=2, fontsize=8)
    #ax.set_ylabel("Cluster name", labelpad=2, fontsize=8)
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=2.4, width=0.7)
    for boundary in boundaries:
        ax.axhline(boundary, color="white", lw=boundary_bg_lw, alpha=0.8)
        ax.axvline(boundary, color="white", lw=boundary_bg_lw, alpha=0.8)
        ax.axhline(boundary, color=boundary_color, lw=boundary_lw, alpha=0.95)
        ax.axvline(boundary, color=boundary_color, lw=boundary_lw, alpha=0.95)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            values.shape[1],
            values.shape[0],
            fill=False,
            edgecolor="black",
            linewidth=1.05,
            zorder=10,
            clip_on=False,
        )
    )
    panel_label(ax, label)
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.055)
    ax_pos = ax.get_position()
    cbar_pos = cbar.ax.get_position()
    #cbar.ax.set_position([cbar_pos.x0 + 0.006, ax_pos.y0, cbar_pos.width, ax_pos.height])
    vmin_eff = np.nanmin(values) if vmin is None else vmin
    vmax_eff = np.nanmax(values) if vmax is None else vmax
    if np.isfinite(vmin_eff) and np.isfinite(vmax_eff):
        cbar.set_ticks(np.linspace(vmin_eff, vmax_eff, 4))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    return cbar


def add_sig_bars(ax, df: pd.DataFrame, metric: str) -> None:
    groups = [df.loc[df["cluster_label"] == label, metric].dropna().values for label in CLUSTER_ORDER]
    pairs = [(i, j) for i in range(len(CLUSTER_ORDER)) for j in range(i + 1, len(CLUSTER_ORDER))]
    valid = [(i, j) for i, j in pairs if len(groups[i]) > 0 and len(groups[j]) > 0]
    if not valid:
        return

    pvals = [mannwhitneyu(groups[i], groups[j], alternative="two-sided")[1] for i, j in valid]
    reject, corrected, _, _ = multipletests(pvals, method="holm")
    sig = sorted(
        [(valid[k], corrected[k]) for k in range(len(valid)) if reject[k]],
        key=lambda item: (item[0][1] - item[0][0], item[0][0]),
    )
    if not sig:
        return

    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    step = yr * 0.060
    bar_h = yr * 0.014
    for lvl, ((i, j), pval) in enumerate(sig):
        y = y_max + yr * 0.025 + lvl * step
        star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
        ax.plot([i, i, j, j], [y, y + bar_h, y + bar_h, y], lw=0.6, c="#333333", clip_on=False)
        ax.text((i + j) / 2, y + bar_h, star, ha="center", va="center", fontsize=7, clip_on=False)
    ax.set_ylim(y_min, y_max)


def add_box_panel(ax, df: pd.DataFrame, metric: str, ylabel: str, label: str, rasterized: bool = False) -> None:
    palette = [CLUSTER_COLORS[label] for label in CLUSTER_ORDER]
    sns.boxplot(
        data=df,
        x="cluster_label",
        y=metric,
        order=CLUSTER_ORDER,
        hue="cluster_label",
        hue_order=CLUSTER_ORDER,
        palette=palette,
        legend=False,
        showfliers=False,
        width=0.45,
        linewidth=1.0,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="cluster_label",
        y=metric,
        order=CLUSTER_ORDER,
        color="black",
        size=3.0,
        alpha=0.10 if rasterized else 0.2,
        jitter=True,
        ax=ax,
        rasterized=rasterized,
    )
    ax.axhline(0, color="#777777", lw=0.65, alpha=0.45, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=4, width=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    add_sig_bars(ax, df, metric)
    panel_label(ax, label, x=-0.33)


def add_scatter_panel(ax, neurons: pd.DataFrame) -> None:
    colors = neurons["cluster_label"].map(CLUSTER_COLORS).fillna("#8c8c8c")
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
    ax.set_ylabel("FCV_z")
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=3.2, width=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "D")


def match_panel_heights(reference_ax, target_axes: list) -> None:
    ref_pos = reference_ax.get_position()
    for ax in target_axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, ref_pos.y0, pos.width, ref_pos.height])


def shrink_panel_widths(target_axes: list, scale: float = 0.8) -> None:
    for ax in target_axes:
        pos = ax.get_position()
        new_width = pos.width * scale
        ax.set_position([pos.x0 + (pos.width - new_width) / 2, pos.y0, new_width, pos.height])


def main() -> None:
    fs.set_paper_style()
    plt.rcParams.update(
        {
            "font.size": fs.TICK_FS_1COL,
            "axes.labelsize": fs.AXIS_LABEL_FS_1COL,
            "xtick.labelsize": fs.TICK_FS_1COL,
            "ytick.labelsize": fs.TICK_FS_1COL,
        }
    )

    neurons, scatter_neurons, recording_points = load_panel_data()
    neuron_order = neurons["neuron"].tolist()
    boundaries = cluster_boundaries(neurons)
    cluster_ticks, cluster_labels = cluster_tick_positions(neurons)

    sc = np.log1p(ordered_matrix(SC_MATRIX, neuron_order).to_numpy())
    fc = ordered_matrix(FC_MATRIX_SPONT, neuron_order).to_numpy()
    fcv = ordered_matrix(FCV_MATRIX_SPONT, neuron_order).to_numpy()
    np.fill_diagonal(fc, 0.0)
    np.fill_diagonal(fcv, 0.0)
    fc_lim = np.nanpercentile(np.abs(fc), 98)

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(16, 4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.18, 1.18, 1.18], "wspace": 0.42},
    )

    add_matrix_panel(fig, axes[0], sc, "magma", "A", "SC", boundaries, cluster_ticks, cluster_labels)
    add_matrix_panel(
        fig,
        axes[1],
        fc,
        "coolwarm",
        "B",
        "FC spont.",
        boundaries,
        cluster_ticks,
        cluster_labels,
        vmin=-fc_lim,
        vmax=fc_lim,
        boundary_lw=0.55,
        boundary_bg_lw=1.2,
        boundary_color="#00843d",
    )
    add_matrix_panel(
        fig,
        axes[2],
        fcv,
        "coolwarm",
        "C",
        "FCV spont.",
        boundaries,
        cluster_ticks,
        cluster_labels,
        boundary_lw=0.55,
        boundary_bg_lw=1.2,
        boundary_color="#00843d",
    )
    add_scatter_panel(axes[3], scatter_neurons)
    add_box_panel(axes[4], recording_points, "FCV_z", "FCV_z", "E", rasterized=True)
    axes[4].set_ylim(top=5.0)
    add_box_panel(axes[5], neurons, "PostDCA", "Post-DCA", "F")
    fig.canvas.draw()
    match_panel_heights(axes[2], [axes[3], axes[4], axes[5]])
    shrink_panel_widths([axes[3], axes[4], axes[5]], scale=0.8)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
