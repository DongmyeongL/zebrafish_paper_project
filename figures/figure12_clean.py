import os
import pickle
import warnings
import itertools

# Figure 12 workflow:
# 1. Load data and prepare derived quantities for plotting.
# 2. Prepare the figure layout and axes.
# 3. Draw each panel.
# 4. Adjust panel positions and add panel labels.
# 5. Save figure files and statistics.

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import figure_style as fs
import figure_supply_sc_heatmap as hm

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
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure12_final.png")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure12_final.pdf")
STATS_DIR = os.path.join(PROJECT_ROOT, "output", "stats")
STATS_CSV = os.path.join(STATS_DIR, "figure12_stats.csv")
SC_NETWORK_CACHE = os.path.join(DATA, "zebrafish_heatmap_matched_region_sc_network_data.npz")
DIVISION_COLUMNS = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}
VALID_DIVISIONS = [2, 1, 3, 0]
STATS_ROWS = []


# ============================================================
# 1. Data loading and plotting calculations
# ============================================================
def _load_panel_a_feature_matrix():
    with open(os.path.join(DATA, "sc_original_per_area_network_metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)

    dac_data = np.load(os.path.join(DATA, "total_selected_region_dac_data.npz"), allow_pickle=True)
    pre_dca = dac_data["arr_1"]
    post_dca = dac_data["arr_2"]
    degree_by_region = hm._load_degree_feature_by_region()

    if os.path.exists(SC_NETWORK_CACHE):
        cached = np.load(SC_NETWORK_CACHE, allow_pickle=True)
        selected_regions = cached["heatmap_regions"].astype(str).tolist()
        selected_divisions = cached["heatmap_divisions"].astype(str).tolist()
    else:
        allowed = pd.read_csv(os.path.join(DATA, "fig1_prism_D_FCS_FCV_bar.csv"))["Region"].tolist()
        selected_regions = [
            region_name
            for region_name in allowed
            if fs.brain_division_list[fs.region.index(region_name)] in hm.DIVISION_LABELS
        ]
        selected_divisions = [
            hm.DIVISION_LABELS[int(fs.brain_division_list[fs.region.index(region_name)])]
            for region_name in selected_regions
        ]

    feature_sources = [
        ("Clustering", metrics["clustering_data"]),
        ("Modularity\nQ", metrics["q_data"]),
        ("Global\nEfficiency", metrics["Eglob_data"]),
        (r"$\mathbf{DCA}_{\mathbf{post}}$", post_dca),
        (r"$\mathbf{DCA}_{\mathbf{pre}}$", pre_dca),
        ("log10\n(Out/In-deg)", degree_by_region),
    ]
    region_to_idx = {region_name: idx for idx, region_name in enumerate(fs.region[:72])}
    feature_rows = [[] for _ in feature_sources]

    for region_name in selected_regions:
        region_idx = region_to_idx[region_name]
        for row, (_, source) in zip(feature_rows, feature_sources):
            if isinstance(source, dict):
                value = source.get(region_name, np.nan)
            else:
                value = hm._clean_mean(source[region_idx])
            row.append(value)

    raw_matrix = np.asarray(feature_rows, dtype=float)
    z_matrix = np.asarray([
        hm._fill_nan_with_mean(hm._zscore(row))
        for row in raw_matrix
    ], dtype=float)
    feature_labels = [label for label, _ in feature_sources]
    return (
        z_matrix,
        feature_labels,
        np.asarray(selected_regions),
        np.asarray(selected_divisions),
    )


def _load_network_post_dca_axis(valid_regions):
    dac = np.load(os.path.join(DATA, "total_selected_region_dac_data.npz"), allow_pickle=True)
    post_dca = np.nanmean(dac["arr_2"], axis=1)
    finite_fill = np.nanmedian(post_dca[np.isfinite(post_dca)])
    post_dca = np.where(np.isfinite(post_dca), post_dca, finite_fill)
    post_max = float(np.nanmax(post_dca[valid_regions]))
    post_min = float(np.nanmin(post_dca[valid_regions]))
    post_denom = post_max - post_min if post_max > post_min else 1.0
    return post_dca, post_max, post_denom


def _network_positions(nodes, post_max, post_denom):
    positions = {}
    for row_idx, division in enumerate(DIVISION_COLUMNS):
        sub = sorted(
            [node for node in nodes if node["division"] == division],
            key=lambda node: (-node["post_dca"], node["region"]),
        )
        if not sub:
            continue
        jitter = np.linspace(-0.32, 0.32, len(sub)) if len(sub) > 1 else np.array([0.0])
        for offset, node in zip(jitter, sub):
            x = (post_max - node["post_dca"]) / post_denom
            y = len(DIVISION_COLUMNS) - 1 - row_idx + offset
            positions[node["idx"]] = (x, y)
    return positions


def _draw_sc_network_panel(ax):
    if not os.path.exists(SC_NETWORK_CACHE):
        raise FileNotFoundError(
            f"Missing cached SC network data: {SC_NETWORK_CACHE}. "
            "Run figures/plot_zebrafish_heatmap_matched_sc_network.py first."
        )

    cached = np.load(SC_NETWORK_CACHE, allow_pickle=True)
    weights = cached["weights"]
    valid_regions = cached["valid_regions"].astype(int)
    heatmap_regions = cached["heatmap_regions"].astype(str)
    heatmap_divisions = cached["heatmap_divisions"].astype(str)

    post_dca, post_max, post_denom = _load_network_post_dca_axis(valid_regions)
    nodes = [
        {
            "idx": int(idx),
            "region": str(region),
            "division": str(division),
            "post_dca": float(post_dca[int(idx)]),
        }
        for idx, region, division in zip(valid_regions, heatmap_regions, heatmap_divisions)
    ]
    positions = _network_positions(nodes, post_max, post_denom)
    max_weight = float(np.nanmax(weights)) if np.isfinite(weights).any() else 1.0
    edge_count = int((weights > 0).sum())

    for source in valid_regions:
        for target in valid_regions:
            weight = float(weights[source, target])
            if weight <= 0:
                continue
            x0, y0 = positions[int(source)]
            x1, y1 = positions[int(target)]
            ax.add_patch(
                mpatches.FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=6.2,
                    shrinkA=2.8,
                    shrinkB=2.8,
                    lw=0.10 + 0.32 * np.log1p(weight) / np.log1p(max_weight),
                    color="#444444",
                    alpha=0.105,
                    zorder=1,
                )
            )

    for division in DIVISION_COLUMNS:
        sub = [node for node in nodes if node["division"] == division]
        if not sub:
            continue
        xy = np.asarray([positions[node["idx"]] for node in sub], dtype=float)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=34,
            c=DIVISION_COLORS[division],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )

    y_ticks = [len(DIVISION_COLUMNS) - 1 - i for i in range(len(DIVISION_COLUMNS))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(DIVISION_COLUMNS, fontsize=fs.TICK_FS_2COL, fontweight="bold")
    for tick_label, division in zip(ax.get_yticklabels(), DIVISION_COLUMNS):
        tick_label.set_color(DIVISION_COLORS[division])

    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.72, len(DIVISION_COLUMNS) - 0.28)
    arrow_y = -0.5
    x0, x1, x2 = 0.08, 0.86, 0.94
    left_half, right_half, head_half = 0.042, 0.010, 0.034
    arrow_vertices = [
        (x0, arrow_y - left_half),
        (x1, arrow_y - right_half),
        (x1, arrow_y - head_half),
        (x2, arrow_y),
        (x1, arrow_y + head_half),
        (x1, arrow_y + right_half),
        (x0, arrow_y + left_half),
    ]
    ax.add_patch(
        mpatches.Polygon(
            arrow_vertices,
            closed=True,
            facecolor="#5b5b5b",
            edgecolor="none",
            alpha=0.82,
            clip_on=False,
            zorder=6,
        )
    )
    
    ax.text(
        0.02,
        arrow_y - 0.15,
        "high",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
        #fontweight="bold",
    )
    
    ax.text(
        0.92,
        arrow_y - 0.15,
        "low",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
       # fontweight="bold",
    )
    
    ax.text(
        0.42,
        arrow_y - 0.24,
        r"$\mathbf{DCA}_{\mathbf{post}}$",
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        fontweight="bold",
    )
    
    
    #ax.text(0.08, arrow_y - 0.18, "Post-DCA high", ha="left", va="top",
    #        fontsize=fs.TICK_FS_2COL, color="#222222", fontweight="bold")
    #ax.text(0.92, arrow_y - 0.18, "low", ha="right", va="top",
    #        fontsize=fs.TICK_FS_2COL - 1, color="#9a9a9a")
    
    
    #ax.set_title(f"Directed SC network\nn={len(valid_regions)}, edges={edge_count}",
    #             fontsize=fs.AXIS_LABEL_FS_2COL)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _load_sc_division_metrics():
    with open(os.path.join(DATA, "sc_original_per_area_network_metrics.pkl"), "rb") as f:
        data = pickle.load(f)

    metrics = {
        "Clustering": data["clustering_data"],
        "Global Efficiency": data["Eglob_data"],
        "Modularity Q": data["q_data"],
    }

    clean_metrics = {}
    for name, metric_data in metrics.items():
        cleaned = []
        for vals in metric_data:
            arr = np.asarray(vals, dtype=float)
            cleaned.append(arr[np.isfinite(arr)].tolist())
        clean_metrics[name] = cleaned

    division_data = {name: [[] for _ in range(4)] for name in metrics}
    for region_idx in range(72):
        div = fs.brain_division_list[region_idx]
        if div > 3:
            continue
        for name in metrics:
            division_data[name][div].extend(clean_metrics[name][region_idx])

    return division_data


def _division_lists_to_df(division_lists):
    mapping = {
        "Tel": division_lists[2],
        "Di": division_lists[1],
        "Mes": division_lists[3],
        "Hind": division_lists[0],
    }
    max_len = max((len(v) for v in mapping.values()), default=0)
    return pd.DataFrame({
        key: pd.Series(vals, dtype=float).reindex(range(max_len))
        for key, vals in mapping.items()
    })


def _load_bottom_panel_data():
    sc_division_data = _load_sc_division_metrics()
    return {
        "Clustering": _division_lists_to_df(sc_division_data["Clustering"]),
        "Modularity Q": _division_lists_to_df(sc_division_data["Modularity Q"]),
        "Global Efficiency": _division_lists_to_df(sc_division_data["Global Efficiency"]),
        "Post-DCA": pd.read_csv(os.path.join(DATA, "fig3_prism_A_PostDCA.csv")),
        "Pre-DCA": pd.read_csv(os.path.join(DATA, "fig3_prism_B_PreDCA.csv")),
    }


# ============================================================
# Plotting helpers
# ============================================================
def _record_division_stats(panel, df, order):
    groups = [df[k].dropna().values for k in order]
    pairs = list(itertools.combinations(range(len(order)), 2))
    pvals = [mannwhitneyu(groups[i], groups[j], alternative="two-sided")[1] for i, j in pairs]
    reject, corr, _, _ = multipletests(pvals, method="holm")
    for idx, (i, j) in enumerate(pairs):
        STATS_ROWS.append({
            "figure": "figure12",
            "panel": panel,
            "test": "Mann-Whitney U",
            "alternative": "two-sided",
            "group_1": order[i],
            "group_2": order[j],
            "n_group_1": len(groups[i]),
            "n_group_2": len(groups[j]),
            "p_uncorrected": pvals[idx],
            "p_holm": corr[idx],
            "reject_holm_0.05": bool(reject[idx]),
        })
    return pairs, reject, corr


def _add_sig_bars(ax, df, order, panel):
    pairs, reject, corr = _record_division_stats(panel, df, order)
    sig = sorted(
        [(pairs[k], corr[k]) for k in range(len(pairs)) if reject[k]],
        key=lambda x: x[0][1] - x[0][0],
    )
    if not sig:
        return
    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    step = yr * 0.060
    bar_h = yr * 0.014
    for lvl, ((i, j), p) in enumerate(sig):
        y = y_max + yr * 0.025 + lvl * step
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        ax.plot(
            [i, i, j, j],
            [y, y + bar_h, y + bar_h, y],
            lw=0.6,
            c="#333",
            clip_on=False,
        )
        ax.text(
            (i + j) / 2, y + bar_h, star,
            ha="center", va="center", fontsize=fs.STAR_FS_2COL,
            clip_on=False,
        )
    ax.set_ylim(y_min, y_max)


def _boxplot_panel(ax, df, ylabel):
    vals, labels = [], []
    for col in DIVISION_COLUMNS:
        v = df[col].dropna().values
        vals.extend(v)
        labels.extend([col] * len(v))
    sns.boxplot(
        x=labels,
        y=vals,
        order=DIVISION_COLUMNS,
        hue=labels,
        hue_order=DIVISION_COLUMNS,
        legend=False,
        palette=[DIVISION_COLORS[d] for d in DIVISION_COLUMNS],
        showfliers=False,
        width=0.45,
        linewidth=1.0,
        ax=ax,
    )
    sns.stripplot(
        x=labels,
        y=vals,
        order=DIVISION_COLUMNS,
        color="black",
        size=3,
        alpha=0.2,
        jitter=True,
        ax=ax,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=4, width=1.2)
    _add_sig_bars(ax, {c: df[c] for c in DIVISION_COLUMNS}, DIVISION_COLUMNS, ylabel)


def _draw_panel_a(fig, subspec, panel_a_data):
    feature_matrix, feature_labels, regions, divisions = panel_a_data
    n_regions = feature_matrix.shape[1]
    z_linkage = linkage(feature_matrix.T, method="ward")
    z_linkage = hm._reorder_linkage_tel_first(z_linkage, divisions)
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

    subgs = GridSpecFromSubplotSpec(
        3,
        6,
        subplot_spec=subspec,
        height_ratios=[1.0, 0.2, 2.5],
        width_ratios=[1, 1, 1, 1, 0.16, 0.24],
        hspace=0.12,
        wspace=0.14,
    )
    
    ax_dendro = fig.add_subplot(subgs[0, 0:4])
    ax_divbar = fig.add_subplot(subgs[1, 0:4])
    ax_heat = fig.add_subplot(subgs[2, 0:4])
    ax_cbar = fig.add_subplot(subgs[2, 5])
    
    
    ax_dendro.text(-0.06, 1.25, "A", transform=ax_dendro.transAxes,
                   fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom", ha="right")

    dendrogram(z_linkage, ax=ax_dendro, no_labels=True, color_threshold=0, above_threshold_color="#333333")
    dend_max_h = max(max(coords) for coords in dend_info["dcoord"])
    ax_dendro.set_xlim(0, n_regions * 10)
    ax_dendro.set_ylim(0, dend_max_h * 1.05)
    ax_dendro.axis("off")

    for i, division in enumerate(divisions_ordered):
        ax_divbar.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1, color=hm.DIVISION_COLORS[division], linewidth=0))
    ax_divbar.set_xlim(-0.5, n_regions - 0.5)
    ax_divbar.set_ylim(0, 1)
    ax_divbar.set_xticks([])
    ax_divbar.set_yticks([])
    for spine in ax_divbar.spines.values():
        spine.set_visible(False)
    for bnd in cluster_boundaries:
        ax_divbar.axvline(x=bnd, color="black", linewidth=1.5, zorder=5)

    im = ax_heat.imshow(matrix_ordered, aspect="auto", cmap="PiYG_r", vmax=2, vmin=-2, interpolation="nearest")
    ax_heat.set_xlim(-0.5, n_regions - 0.5)
    ax_heat.set_yticks(range(len(feature_labels)))
    ax_heat.set_yticklabels(feature_labels, fontsize=fs.TICK_FS_2COL, fontweight="bold")
    ax_heat.tick_params(axis="y", length=0, labelright=True, labelleft=False)
    ax_heat.set_xticks(range(n_regions))
    ax_heat.set_xticklabels(regions_ordered, rotation=90, fontsize=fs.TICK_FS_2COL - 2, ha="center", fontweight="bold")
    ax_heat.tick_params(axis="x", length=2, width=0.5)
    for tick_label, div in zip(ax_heat.get_xticklabels(), divisions_ordered):
        tick_label.set_color(hm.DIVISION_COLORS[div])
    for y in np.arange(-0.5, len(feature_labels), 1):
        ax_heat.axhline(y=y, color="white", linewidth=0.4)
    for bnd in cluster_boundaries:
        ax_heat.axvline(x=bnd, color="black", linewidth=1.5, zorder=5)
    for spine in ax_heat.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax_heat.add_patch(
        mpatches.Rectangle(
            (-0.5, -0.5), n_regions, len(feature_labels),
            fill=False, edgecolor="black", linewidth=1.4,
            zorder=10, clip_on=False,
        )
    )
    
    cbar = plt.colorbar(im, cax=ax_cbar)
    ax_cbar.set_position([0.5235,0.585, 0.010, 0.225])
    cbar.ax.set_title("Z-score", fontsize=fs.AXIS_LABEL_FS_2COL, fontweight="bold")
    cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)
    cbar.set_ticks([-2, -1, 0, 1, 2])

    div_handles = [mpatches.Patch(color=hm.DIVISION_COLORS[d], label=d) for d in ["Tel", "Di", "Mes", "Hind"]]
    leg = ax_heat.legend(handles=div_handles, loc="upper left", bbox_to_anchor=(0.41, 1.75),
                         ncol=4, fontsize=fs.TICK_FS_2COL, frameon=False, framealpha=0.95,
                         title_fontsize=fs.AXIS_LABEL_FS_2COL)
    leg.get_title().set_fontweight("bold")
    for text in leg.get_texts():
        text.set_fontweight("bold")


def _draw_panels_b_c(fig, subspec):
    container = fig.add_subplot(subspec)
    container.axis("off")

    container.text(0.08, 1.07, "B", transform=container.transAxes,
                   fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom", ha="right")

    container.text(0.52, 1.07, "C", transform=container.transAxes,
                   fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom", ha="right")

    subgs = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=subspec,
        width_ratios=[1.05, 1.0],
        wspace=0.12,
    )
    ax_network = fig.add_subplot(subgs[0, 0])
    _draw_sc_network_panel(ax_network)

    c_gs = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subgs[0, 1],
        hspace=0.02,
        wspace=0.02,
    )

    files = [
        "network_rP_rimRF_network_diagarm.png",
        "network_rP_rMOS4_network_diagarm.png",
        "network_rRa_rpRF_network_diagarm.png",
        "network_rRa_rimRF_network_diagarm.png",
    ]
    bottom_labels = [
        ("P", "imRF"),
        ("P", "MOS4"),
        ("Ra", "pRF"),
        ("Ra", "imRF"),
    ]
    c_axes = []
    for i, (fname, (top_label, bottom_label)) in enumerate(zip(files, bottom_labels)):
        ax = fig.add_subplot(c_gs[i // 2, i % 2])
        ax.imshow(plt.imread(os.path.join(DATA, fname)))

        pos = ax.get_position()
        ax.set_position([
            pos.x0 + pos.width * 0.1,
            pos.y0 + pos.height * 0.1,
            pos.width * 0.8,
            pos.height * 0.8,
        ])

        ax.text(0.50, 0.90, top_label, transform=ax.transAxes,
                ha="center", va="center", fontsize=fs.AXIS_LABEL_FS_2COL,
                fontweight="bold", color="#333333")
        ax.text(0.50, 0.10, bottom_label, transform=ax.transAxes,
                ha="center", va="center", fontsize=fs.AXIS_LABEL_FS_2COL,
                fontweight="bold", color="#333333")
        ax.axis("off")
        c_axes.append(ax)

    c_positions = [ax.get_position() for ax in c_axes]
    c_x0 = min(pos.x0 for pos in c_positions)
    c_x1 = max(pos.x1 for pos in c_positions)
    c_y0 = min(pos.y0 for pos in c_positions)
    cbar_ax = fig.add_axes([c_x0 + 0.18 * (c_x1 - c_x0), c_y0 - 0.030, 0.36 * (c_x1 - c_x0), 0.010])
    cbar_mappable = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap="jet")
    cbar_mappable.set_array([])
    cbar = fig.colorbar(cbar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL - 1, length=2, pad=1)
    cbar.set_label(
         r"$\mathrm{DCA}$",
        fontsize=fs.TICK_FS_2COL,
        labelpad=0.1,
    )
    cbar_ax.annotate(
        "",
        xy=(1.18, 1.85),
        xytext=(1.18, -0.85),
        xycoords=cbar_ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#e5a247", lw=2.4, mutation_scale=12),
        clip_on=False,
    )
    cbar_ax.annotate(
        "",
        xy=(1.34, -1.85+0.33),
        xytext=(1.34, 0.85+0.33),
        xycoords=cbar_ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color="#8bcef4", lw=2.4, mutation_scale=12),
        clip_on=False,
    )

    pos = ax_network.get_position()
    ax_network.set_position([pos.x0 + 0.06, pos.y0, pos.width * 0.82, pos.height])


def _adjust_bottom_panel_positions(axes):
    for ax in axes:
        pos = ax.get_position()
        center_x = pos.x0 + pos.width / 2
        center_y = pos.y0 + pos.height / 2
        new_width = pos.width * 0.8
        new_height = pos.height * 0.45
        ax.set_position([
            center_x - new_width / 2,
            center_y - new_height / 2,
            new_width,
            new_height,
        ])


def _add_bottom_panel_labels(axes):
    for label, ax in zip(["D", "E", "F", "G", "H"], axes):
        ax.text(
            -0.36, 1.02, label,
            transform=ax.transAxes,
            fontsize=fs.PANEL_LABEL_FS_2COL,
            fontweight="bold",
            va="bottom",
        )


def main():
    STATS_ROWS.clear()

    # ============================================================
    # 1. Data loading and plotting calculations
    # ============================================================
    panel_a_data = _load_panel_a_feature_matrix()
    bottom_panel_data = _load_bottom_panel_data()

    # ============================================================
    # 2. Layout preparation
    # ============================================================
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(
        5, 10, figure=fig,
        height_ratios=[1.0, 0.2, 2.5, 2.0, 2.0],
        width_ratios=[1] * 10,
        left=0.08, right=0.97, top=0.94, bottom=0.20,
        hspace=0.20, wspace=0.30,
    )

    ax_c = fig.add_subplot(gs[3:5, 0:2])
    ax_d = fig.add_subplot(gs[3:5, 2:4])
    ax_e = fig.add_subplot(gs[3:5, 4:6])
    ax_f = fig.add_subplot(gs[3:5, 6:8])
    ax_g = fig.add_subplot(gs[3:5, 8:10])
    bottom_axes = [ax_c, ax_d, ax_e, ax_f, ax_g]

    # ============================================================
    # 3. Draw each panel
    # ============================================================
    _draw_panel_a(fig, gs[0:3, 0:5], panel_a_data)
    _draw_panels_b_c(fig, gs[0:3, 5:10])
    _boxplot_panel(ax_c, bottom_panel_data["Clustering"], "Clustering")
    ax_c.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value * 1e2:g}"))
    ax_c.text(
        -0.22, 1.02, r"$\times 10^{-2}$",
        transform=ax_c.transAxes,
        fontsize=fs.TICK_FS_2COL,
        ha="left",
        va="bottom",
    )
    _boxplot_panel(ax_d, bottom_panel_data["Modularity Q"], "Modularity Q")
    _boxplot_panel(ax_e, bottom_panel_data["Global Efficiency"], "Global Efficiency")
    _boxplot_panel(ax_f, bottom_panel_data["Post-DCA"], r"$\mathrm{DCA}_{\mathrm{post}}$")
    _boxplot_panel(ax_g, bottom_panel_data["Pre-DCA"], r"$\mathrm{DCA}_{\mathrm{pre}}$")

    # ============================================================
    # 4. Panel position adjustment and panel labels
    # ============================================================
    _adjust_bottom_panel_positions(bottom_axes)
    _add_bottom_panel_labels(bottom_axes)

    # ============================================================
    # 5. Save figure and statistics
    # ============================================================
    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight",transparent=True)
    #fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    os.makedirs(STATS_DIR, exist_ok=True)
    pd.DataFrame(STATS_ROWS).to_csv(STATS_CSV, index=False)
    print(f"Saved {STATS_CSV}")


if __name__ == "__main__":
    main()
