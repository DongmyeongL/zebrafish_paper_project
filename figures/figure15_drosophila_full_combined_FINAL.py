"""
Full Drosophila Figure 15 combined panel.

Final working result:
  - FlyWire783 cell-level weighted directed SC.
  - Branson999 full ROI FC/FCV, window=30 frames, step=8 frames.
  - Main association: mean Post-DCA+ versus FCV_z excluding same side-aware block.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, Patch, Polygon
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import butter, sosfiltfilt
from scipy.signal import detrend as scipy_detrend
from scipy.stats import kruskal, mannwhitneyu, pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "figure15_drosophila"
STANDARD = BASE / "final_results" / "SC_FC_FCV_current_standard"
BRANSON = BASE / "final_results" / "Branson999_full_FC_FCV"
FIGS = PROJECT_ROOT / "output" / "png"
PDFS = PROJECT_ROOT / "output" / "pdf"

SC_ITO48 = STANDARD / "SC_flywire783_ito_R_then_L_matrix_FINAL.csv"
POSTDCA_BLOCK = STANDARD / "sideaware_cell_PostDCA_block_scores_FINAL.csv"
ITO_ORDER = STANDARD / "ito_R_then_L_region_order_FINAL.csv"
DCA_REGION = STANDARD / "DCA_PostDCA_ito48_FINAL.csv"
ORDER_999 = BRANSON / "Branson999_full_roi_order_FINAL.csv"
FC_999 = BRANSON / "Branson999_full_FC_matrix_w15_step5_hp030_FINAL.csv"
FCV_999 = BRANSON / "Branson999_full_FCV_matrix_w15_step5_hp030_FINAL.csv"
SCATTER_SCORES = BRANSON / "Branson999_w15_step5_hp030_weighted_vs_binary_PostDCApositive_scores_FINAL.csv"
FC5_RECORDING = BRANSON / "Branson999_full_ROI_5measure_recording_level_w15_step5_hp030_FINAL.csv"
FC5_ROI_SUMMARY = BRANSON / "Branson999_full_ROI_5measure_summary_w15_step5_hp030_FINAL.csv"
METHOD_STATS = BRANSON / "Branson999_w30_PostDCA_vs_PostDCApositive_method_check_stats_FINAL.csv"
WEIGHT_BINARY_STATS = BRANSON / "Branson999_w15_step5_hp030_weighted_vs_binary_PostDCApositive_stats_FINAL.csv"
BRANSON_RESP = BASE / "raw" / "turner_mann_clandinin" / "data" / "branson_responses"
REC_FCV_POINTS = BRANSON / "Branson999_w30_matched41_region_FCVz_by_recording_FINAL.csv"
FC5_SIDEKEY = BRANSON / "Branson999_full_ROI_5measure_sidekey_summary_w15_step5_hp030_FINAL.csv"
SC_MEASURE_RESULTS = BASE / "results" / "drosophila_flywire783_matched41_sc_cell_measures.csv"

OUT_PNG = FIGS / "figure15_drosophila_full_combined_FINAL.png"
OUT_PDF = PDFS / "figure15_drosophila_full_combined_FINAL.pdf"
FINAL_COPY = OUT_PNG

GROUP_ORDER = [
    "olfactory system",
    "mushroom body",
    "central complex",
    "optic/lateral protocerebrum",
    "superior protocerebrum",
    "inferior/ventrolateral protocerebrum",
]

GROUP_COLORS = {
    "olfactory system": "#54a24b",
    "mushroom body": "#b279a2",
    "central complex": "#4c78a8",
    "optic/lateral protocerebrum": "#f58518",
    "superior protocerebrum": "#72b7b2",
    "inferior/ventrolateral protocerebrum": "#e45756",
}

SHORT_LABELS = {
    "olfactory system": "Olfactory",
    "mushroom body": "MB",
    "central complex": "CX",
    "optic/lateral protocerebrum": "Optic/LP",
    "superior protocerebrum": "Superior",
    "inferior/ventrolateral protocerebrum": "Inferior/VL",
}

SC_MEASURE_COLS = [
    "mean_PostDCA_positive",
    "mean_PreDCA_positive",
    "mean_Log10_OutInput_degree",
    "mean_OO_fraction",
]
SC_MEASURE_LABELS = [
    "Post-DCA+",
    "Pre-DCA+",
    "log10 out/in degree",
    "Output-output motif",
]

FC_MEASURE_COLS = ["FCS_z", "FCV_z", "Metastability", "NetTE_z", "NeighborNetTE_z"]
FC_MEASURE_LABELS = ["z-FCS", "z-FCV", "Metastability", "Net TE", "Neighbor Net TE"]
TRACE_EXAMPLE_DATE = "2018-11-03"
TRACE_HIGH_RECORDING = "2018-11-03_3"
TRACE_LOW_RECORDING = "2018-11-03_4"
TRACE_HIGH_PAIR = ("MB_R_727", "SLP_R_870")
TRACE_LOW_PAIR = ("CRE_L_136", "CRE_L_140")
TRACE_SAMPLING_RATE_HZ = 1.2
TRACE_HIGHPASS_HZ = 0.03
TRACE_FCV_WINDOW = 15
TRACE_FCV_STEP = 5
TRACE_DISPLAY_SECONDS = 300
TRACE_PAIR_COLORS = {
    "high": ("#4e79a7", "#59a14f"),
    "low": ("#59a14f", "#e15759"),
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def label(ax: plt.Axes, text: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(x, y, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold")


def style_open_axes(ax: plt.Axes, tick_length: float = 3.2, tick_width: float = 1.0) -> None:
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=tick_length, width=tick_width)
    ax.spines[["top", "right"]].set_visible(False)


def split_side(name: str) -> tuple[str, str]:
    text = str(name)
    if text.endswith("_L") or text.endswith("_R"):
        return text[:-2], text[-1]
    return text, "midline"


def add_matrix_boundaries(ax: plt.Axes, labels: list[str], groups: list[str] | None = None, lw_scale: float = 1.0) -> None:
    sides = [split_side(x)[1] for x in labels]
    for idx in range(1, len(labels)):
        if sides[idx] != sides[idx - 1]:
            ax.axhline(idx - 0.5, color="black", lw=1.05 * lw_scale)
            ax.axvline(idx - 0.5, color="black", lw=1.05 * lw_scale)
        elif groups is not None and groups[idx] != groups[idx - 1]:
            ax.axhline(idx - 0.5, color=GROUP_COLORS.get(groups[idx], "white"), lw=0.35 * lw_scale, alpha=0.9)
            ax.axvline(idx - 0.5, color=GROUP_COLORS.get(groups[idx], "white"), lw=0.35 * lw_scale, alpha=0.9)


def add_roi_boundaries(ax: plt.Axes, order: pd.DataFrame) -> None:
    sides = order["side"].astype(str).tolist()
    bases = order["base_region"].astype(str).tolist()
    for idx in range(1, len(order)):
        if sides[idx] != sides[idx - 1]:
            color, lw, alpha = "black", 0.9, 0.85
        elif bases[idx] != bases[idx - 1]:
            color, lw, alpha = "white", 0.18, 0.45
        else:
            continue
        ax.axhline(idx - 0.5, color=color, lw=lw, alpha=alpha)
        ax.axvline(idx - 0.5, color=color, lw=lw, alpha=alpha)


def group_tick_positions(nodes: pd.DataFrame) -> tuple[list[float], list[str]]:
    ticks = []
    labels = []
    for group, sub in nodes.reset_index().groupby("big_group", sort=False):
        ticks.append(float((sub["index"].min() + sub["index"].max()) / 2))
        labels.append(SHORT_LABELS.get(group, str(group)))
    return ticks, labels


def add_matrix_outline(ax: plt.Axes, shape: tuple[int, int]) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.add_patch(
        plt.Rectangle(
            (-0.5, -0.5),
            shape[1],
            shape[0],
            fill=False,
            edgecolor="black",
            linewidth=1.05,
            zorder=10,
            clip_on=False,
        )
    )


def add_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    cmap: str,
    panel: str,
    vmin=None,
    vmax=None,
    tick_positions: list[float] | None = None,
    tick_labels: list[str] | None = None,
) -> object:
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=4, fontsize=8)
    if tick_positions is None or tick_labels is None:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6.5)
        ax.set_yticklabels(tick_labels, fontsize=6.5)
    ax.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=2.4, width=0.7)
    add_matrix_outline(ax, matrix.shape)
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.055)
    vmin_eff = np.nanmin(matrix) if vmin is None else vmin
    vmax_eff = np.nanmax(matrix) if vmax is None else vmax
    if np.isfinite(vmin_eff) and np.isfinite(vmax_eff):
        cbar.set_ticks(np.linspace(vmin_eff, vmax_eff, 4))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    label(ax, panel)
    return cbar


def zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def detrended_zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return zscore_1d(x)
    filled = x.copy()
    if not finite.all():
        idx = np.arange(len(filled))
        filled[~finite] = np.interp(idx[~finite], idx[finite], filled[finite])
    detrended = scipy_detrend(filled, type="linear")
    if len(detrended) < 20:
        return zscore_1d(detrended)
    cutoff = TRACE_HIGHPASS_HZ / (0.5 * TRACE_SAMPLING_RATE_HZ)
    sos = butter(2, cutoff, btype="highpass", output="sos")
    filtered = sosfiltfilt(sos, detrended)
    return zscore_1d(filtered)


def zscore_rows_for_plot(matrix: np.ndarray) -> np.ndarray:
    out = matrix.astype(float).copy()
    for i in range(out.shape[0]):
        row = out[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        sd = np.nanstd(row[mask])
        if sd > 0:
            out[i, mask] = (row[mask] - np.nanmean(row[mask])) / sd
    return np.clip(out, -2.0, 2.0)


def orient_linkage_high_score_left(z_linkage: np.ndarray, scores: np.ndarray) -> np.ndarray:
    oriented = z_linkage.copy()
    n = len(scores)
    score_sum: dict[int, float] = {i: float(scores[i]) if np.isfinite(scores[i]) else 0.0 for i in range(n)}
    counts: dict[int, int] = {i: int(np.isfinite(scores[i])) for i in range(n)}
    for row_idx in range(oriented.shape[0]):
        node = n + row_idx
        left = int(oriented[row_idx, 0])
        right = int(oriented[row_idx, 1])
        left_mean = score_sum[left] / max(counts[left], 1)
        right_mean = score_sum[right] / max(counts[right], 1)
        if right_mean > left_mean:
            oriented[row_idx, 0], oriented[row_idx, 1] = oriented[row_idx, 1], oriented[row_idx, 0]
            left, right = right, left
        score_sum[node] = score_sum[left] + score_sum[right]
        counts[node] = counts[left] + counts[right]
    return oriented


def pair_window_corr(a: np.ndarray, b: np.ndarray, window: int = TRACE_FCV_WINDOW, step: int = TRACE_FCV_STEP) -> tuple[np.ndarray, np.ndarray]:
    a = detrended_zscore_1d(a)
    b = detrended_zscore_1d(b)
    vals = []
    centers = []
    for start in range(0, len(a) - window + 1, step):
        aa = zscore_1d(a[start : start + window])
        bb = zscore_1d(b[start : start + window])
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
        centers.append(start + window / 2)
    return np.asarray(centers), np.asarray(vals)


def matched_node_table() -> pd.DataFrame:
    scores = pd.read_csv(SCATTER_SCORES)
    order = pd.read_csv(ORDER_999)[["roi", "label", "atlas_region", "side"]]
    merged = scores.merge(order, on=["roi", "label"], how="left", suffixes=("", "_order"))
    table = (
        merged.dropna(subset=["atlas_region"])
        .groupby("atlas_region", as_index=False)
        .agg(
            n_rois=("roi", "size"),
            big_group=("big_group", lambda x: x.mode().iat[0]),
            side=("side_order", lambda x: x.mode().iat[0] if "side_order" in merged.columns and len(x.mode()) else ""),
        )
    )
    if "side" not in table or table["side"].isna().all():
        table["side"] = table["atlas_region"].map(lambda x: split_side(x)[1])
    table = table[table["big_group"].isin(GROUP_ORDER)].copy()
    table["side_order"] = table["side"].map({"R": 0, "L": 1, "midline": 2}).fillna(3).astype(int)
    table["group_order"] = table["big_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    table = table.sort_values(["group_order", "side_order", "atlas_region"]).reset_index(drop=True)
    return table


def holm_significant_pairs(groups: list[np.ndarray]) -> list[tuple[tuple[int, int], float]]:
    pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups)) if len(groups[i]) and len(groups[j])]
    if not pairs:
        return []
    raw = np.array([mannwhitneyu(groups[i], groups[j], alternative="two-sided").pvalue for i, j in pairs])
    order = np.argsort(raw)
    corrected = np.empty_like(raw)
    running = 0.0
    m = len(raw)
    for rank, idx in enumerate(order):
        val = min(raw[idx] * (m - rank), 1.0)
        running = max(running, val)
        corrected[idx] = running
    return [(pairs[i], float(corrected[i])) for i in range(len(pairs)) if corrected[i] < 0.05]


def p_to_stars(pval: float) -> str:
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "n.s."


def add_sig_bars(ax: plt.Axes, groups: list[np.ndarray], start_y: float | None = None) -> None:
    sig = sorted(holm_significant_pairs(groups), key=lambda item: (item[0][1] - item[0][0], item[0][0]))
    if not sig:
        return
    y_min, y_max = ax.get_ylim()
    yr = y_max - y_min
    step = yr * 0.040
    bar_h = yr * 0.010
    first_y = y_max + yr * 0.002 if start_y is None else start_y
    for lvl, ((i, j), pval) in enumerate(sig):
        y = first_y + lvl * step
        star = p_to_stars(pval).replace("n.s.", "ns")
        ax.plot([i, i, j, j], [y, y + bar_h, y + bar_h, y], lw=0.6, c="#333333", clip_on=False)
        ax.text((i + j) / 2, y + bar_h, star, ha="center", va="center", fontsize=7, clip_on=False)
    ax.set_ylim(y_min, y_max)


def collapse_ito_region(region: str) -> str:
    text = str(region)
    if text.startswith("MB_") and (text.endswith("_R") or text.endswith("_L")):
        return "MB_" + text[-1]
    if text.startswith("AOTU_"):
        return "OTU_" + text[-1]
    return text


def aggregate_sc_to_matched_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    sc = pd.read_csv(SC_ITO48, index_col=0)
    row_groups = pd.Index([collapse_ito_region(x) for x in sc.index], name="region")
    col_groups = pd.Index([collapse_ito_region(x) for x in sc.columns], name="region")
    agg = sc.copy()
    agg.index = row_groups
    agg.columns = col_groups
    agg = agg.groupby(level=0).sum().T.groupby(level=0).sum().T
    labels = nodes["atlas_region"].tolist()
    agg = agg.reindex(index=labels, columns=labels, fill_value=0.0)
    agg_values = agg.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(agg_values, 0.0)
    return pd.DataFrame(agg_values, index=labels, columns=labels)


def aggregate_999_matrix_to_matched_nodes(matrix_path: Path, nodes: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, index_col=0)
    order = pd.read_csv(ORDER_999)
    labels_by_node = {
        node: order.loc[order["atlas_region"] == node, "label"].astype(str).tolist()
        for node in nodes["atlas_region"]
    }
    labels = nodes["atlas_region"].tolist()
    out = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for src in labels:
        src_labels = [x for x in labels_by_node[src] if x in matrix.index]
        if not src_labels:
            continue
        for dst in labels:
            dst_labels = [x for x in labels_by_node[dst] if x in matrix.columns]
            if not dst_labels:
                continue
            block = matrix.loc[src_labels, dst_labels].to_numpy(dtype=float)
            out.loc[src, dst] = float(np.nanmean(block))
    out_values = out.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(out_values, 0.0)
    return pd.DataFrame(out_values, index=labels, columns=labels)


def recording_node_timeseries(rec: pd.DataFrame, nodes: pd.DataFrame, order: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    roi_to_node = order.set_index("roi")["atlas_region"].astype(str).to_dict()
    rows = []
    labels = []
    for node in nodes["atlas_region"].astype(str):
        rois = [int(x) for x in rec.index if roi_to_node.get(int(x)) == node]
        if not rois:
            continue
        traces = np.vstack([zscore_1d(rec.loc[roi].to_numpy(dtype=float)) for roi in rois])
        rows.append(np.nanmean(traces, axis=0))
        labels.append(node)
    return labels, np.vstack(rows) if rows else np.empty((0, rec.shape[1]))


def corr_fcv_matrix_from_timeseries(x: np.ndarray, window: int = 30, step: int = 8) -> np.ndarray:
    n, nt = x.shape
    fc_sum = np.zeros((n, n), dtype=float)
    fc2_sum = np.zeros((n, n), dtype=float)
    nwin = 0
    for start in range(0, nt - window + 1, step):
        win = x[:, start : start + window]
        win = np.vstack([zscore_1d(row) for row in win])
        corr = (win @ win.T) / win.shape[1]
        corr = np.clip(corr, -1.0, 1.0)
        fc_sum += corr
        fc2_sum += corr * corr
        nwin += 1
    fc = fc_sum / max(nwin, 1)
    fcv = np.sqrt(np.maximum(fc2_sum / max(nwin, 1) - fc * fc, 0.0))
    np.fill_diagonal(fcv, np.nan)
    return fcv


def load_recording_fcv_points(nodes: pd.DataFrame) -> pd.DataFrame:
    if REC_FCV_POINTS.exists():
        return pd.read_csv(REC_FCV_POINTS)
    order = pd.read_csv(ORDER_999)
    paths = sorted(BRANSON_RESP.glob("branson_*.pkl"))
    rows = []
    for path in paths:
        rec = pd.read_pickle(path)
        labels, x = recording_node_timeseries(rec, nodes, order)
        if len(labels) < 4:
            continue
        fcv = corr_fcv_matrix_from_timeseries(x)
        node_to_group = nodes.set_index("atlas_region")["big_group"].to_dict()
        values = []
        for i, node in enumerate(labels):
            vals = fcv[i, :]
            values.append(float(np.nanmean(vals)))
        z = zscore_1d(np.asarray(values, dtype=float))
        for node, val, zval in zip(labels, values, z):
            rows.append(
                {
                    "recording_id": path.stem.replace("branson_", ""),
                    "atlas_region": node,
                    "big_group": node_to_group.get(node, ""),
                    "region_FCV": val,
                    "region_FCV_z": float(zval),
                }
            )
    out = pd.DataFrame(rows)
    REC_FCV_POINTS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REC_FCV_POINTS, index=False)
    return out


def add_matched_boundaries(ax: plt.Axes, nodes: pd.DataFrame, lw_scale: float = 1.0) -> None:
    sides = nodes["side"].astype(str).tolist()
    groups = nodes["big_group"].astype(str).tolist()
    for idx in range(1, len(nodes)):
        if sides[idx] != sides[idx - 1]:
            color, lw, alpha = "black", 1.0 * lw_scale, 0.9
        elif groups[idx] != groups[idx - 1]:
            color, lw, alpha = GROUP_COLORS.get(groups[idx], "white"), 0.42 * lw_scale, 0.95
        else:
            continue
        ax.axhline(idx - 0.5, color=color, lw=lw, alpha=alpha)
        ax.axvline(idx - 0.5, color=color, lw=lw, alpha=alpha)


def region_group_lookup() -> dict[str, str]:
    scores = pd.read_csv(SCATTER_SCORES)
    order = pd.read_csv(ORDER_999)[["roi", "label", "atlas_region"]]
    merged = scores.merge(order, on=["roi", "label"], how="left")
    return merged.groupby("atlas_region")["big_group"].agg(lambda x: x.mode().iat[0]).to_dict()


def choose_trace_examples() -> tuple[dict, dict]:
    fcv = pd.read_csv(FCV_999, index_col=0)
    fc = pd.read_csv(FC_999, index_col=0)
    order = pd.read_csv(ORDER_999)
    label_to_roi = order.set_index("label")["roi"].astype(int).to_dict()
    label_to_region = order.set_index("label")["atlas_region"].astype(str).to_dict()
    group_by_region = region_group_lookup()
    label_to_group = {label: group_by_region.get(region, "") for label, region in label_to_region.items()}
    paths = sorted(BRANSON_RESP.glob("branson_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No Branson response pkl files in {BRANSON_RESP}")

    def recording_name(path: Path) -> str:
        return path.stem.replace("branson_", "")

    def recording_date(path: Path) -> str:
        return recording_name(path).split("_", 1)[0]

    def make_example(path: Path, rec: pd.DataFrame, lab_a: str, lab_b: str, tag: str) -> dict:
        roi_a, roi_b = label_to_roi[lab_a], label_to_roi[lab_b]
        a = rec.loc[roi_a].to_numpy(dtype=float)
        b = rec.loc[roi_b].to_numpy(dtype=float)
        centers, corr = pair_window_corr(a, b)
        n_display = min(len(a), int(round(TRACE_DISPLAY_SECONDS * TRACE_SAMPLING_RATE_HZ)))
        max_start = max(len(a) - n_display, 0)
        seed_text = f"{recording_name(path)}:{lab_a}:{lab_b}"
        seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        display_start = 0 if max_start == 0 else int(np.random.default_rng(seed).integers(0, max_start + 1))
        return {
            "recording": recording_name(path),
            "tag": tag,
            "label_a": lab_a,
            "label_b": lab_b,
            "roi_a": roi_a,
            "roi_b": roi_b,
            "trace_a": detrended_zscore_1d(a),
            "trace_b": detrended_zscore_1d(b),
            "centers": centers,
            "corr": corr,
            "global_fc": float(fc.loc[lab_a, lab_b]),
            "global_fcv": float(fcv.loc[lab_a, lab_b]),
            "display_start": display_start,
            "display_stop": display_start + n_display,
        }

    path_by_recording = {recording_name(path): path for path in paths}
    if TRACE_HIGH_RECORDING in path_by_recording and TRACE_LOW_RECORDING in path_by_recording:
        high_path = path_by_recording[TRACE_HIGH_RECORDING]
        low_path = path_by_recording[TRACE_LOW_RECORDING]
        high_rec = pd.read_pickle(high_path)
        low_rec = pd.read_pickle(low_path)
        if all(label_to_roi[label] in high_rec.index for label in TRACE_HIGH_PAIR) and all(
            label_to_roi[label] in low_rec.index for label in TRACE_LOW_PAIR
        ):
            return (
                make_example(high_path, high_rec, TRACE_HIGH_PAIR[0], TRACE_HIGH_PAIR[1], "High FCV: MB-Superior"),
                make_example(low_path, low_rec, TRACE_LOW_PAIR[0], TRACE_LOW_PAIR[1], "Low FCV: Inferior/VL"),
            )

    candidate_paths = [path for path in paths if recording_date(path) == TRACE_EXAMPLE_DATE]
    if len(candidate_paths) < 2:
        candidate_paths = paths

    for path in candidate_paths:
        same_date_paths = [other for other in paths if other != path and recording_date(other) == recording_date(path)]
        if not same_date_paths:
            continue
        rec = pd.read_pickle(path)
        present_rois = set(int(x) for x in rec.index)
        present_labels = [lab for lab, roi in label_to_roi.items() if roi in present_rois and lab in fcv.index]
        if len(present_labels) < 20:
            continue
        high_candidates = [
            lab
            for lab in present_labels
            if label_to_group.get(lab) in {"mushroom body", "superior protocerebrum"}
        ]
        low_candidates = [
            lab for lab in present_labels if label_to_group.get(lab) == "inferior/ventrolateral protocerebrum"
        ]
        if len(high_candidates) < 4 or len(low_candidates) < 3:
            continue
        pairs: list[tuple[str, str, float]] = []
        for i, lab_a in enumerate(high_candidates):
            for lab_b in high_candidates[i + 1 :]:
                groups = {label_to_group.get(lab_a), label_to_group.get(lab_b)}
                if groups != {"mushroom body", "superior protocerebrum"}:
                    continue
                val = float(fcv.loc[lab_a, lab_b])
                if np.isfinite(val):
                    pairs.append((lab_a, lab_b, val))
        low_pairs: list[tuple[str, str, float]] = []
        for i, lab_a in enumerate(low_candidates):
            for lab_b in low_candidates[i + 1 :]:
                val = float(fcv.loc[lab_a, lab_b])
                if np.isfinite(val):
                    low_pairs.append((lab_a, lab_b, val))
        if not pairs or not low_pairs:
            continue
        high_lab_a, high_lab_b, _ = max(pairs, key=lambda x: x[2])
        high_example = make_example(path, rec, high_lab_a, high_lab_b, "High FCV: MB-Superior")
        for low_path in same_date_paths:
            low_rec = pd.read_pickle(low_path)
            present_rois = set(int(x) for x in low_rec.index)
            present_labels = [lab for lab, roi in label_to_roi.items() if roi in present_rois and lab in fcv.index]
            if len(present_labels) < 20:
                continue
            low_candidates = [
                lab for lab in present_labels if label_to_group.get(lab) == "inferior/ventrolateral protocerebrum"
            ]
            if len(low_candidates) < 3:
                continue
            low_pairs: list[tuple[str, str, float]] = []
            for i, lab_a in enumerate(low_candidates):
                for lab_b in low_candidates[i + 1 :]:
                    val = float(fcv.loc[lab_a, lab_b])
                    if np.isfinite(val):
                        low_pairs.append((lab_a, lab_b, val))
            if not low_pairs:
                continue
            low_lab_a, low_lab_b, _ = min(low_pairs, key=lambda x: x[2])
            low_example = make_example(low_path, low_rec, low_lab_a, low_lab_b, "Low FCV: Inferior/VL")
            return high_example, low_example

    raise RuntimeError("Could not find high and low Branson999 examples from separate recordings on the same date.")


def draw_sc_matrix(fig: plt.Figure, ax: plt.Axes, panel: str = "A") -> None:
    nodes = matched_node_table()
    sc = aggregate_sc_to_matched_nodes(nodes)
    arr = np.log1p(sc.to_numpy(dtype=float))
    return add_heatmap(fig, ax, arr, "SC", "magma", panel)


def draw_postdca_bar(ax: plt.Axes) -> None:
    df = pd.read_csv(POSTDCA_BLOCK)
    df = df[df["big_group"].isin(GROUP_ORDER)].copy()
    df["side_order"] = df["side"].map({"R": 0, "L": 1, "midline": 2}).fillna(3)
    df["group_order"] = df["big_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    df = df.sort_values(["side_order", "group_order", "side_key"]).reset_index(drop=True)
    x = np.arange(len(df))
    colors = [GROUP_COLORS[g] for g in df["big_group"]]
    ax.bar(x, df["mean_PostDCA_positive"], color=colors, alpha=0.9, width=0.82)
    ax.axhline(0, color="0.35", lw=0.7)
    for idx in range(1, len(df)):
        if df.loc[idx, "side"] != df.loc[idx - 1, "side"]:
            ax.axvline(idx - 0.5, color="black", lw=1.0)
    ax.set_ylabel("Mean Post-DCA+")
    ax.set_xlabel("FlyWire side-aware blocks")
    ax.set_title("Weighted cell-level Post-DCA+", fontsize=8, pad=4)
    ax.set_xticks(x[::2], df["side_key"].iloc[::2], rotation=90, fontsize=5.6)
    style_open_axes(ax)
    label(ax, "B", x=-0.08)


def draw_ito48_network(ax: plt.Axes) -> None:
    nodes = matched_node_table()
    sc = aggregate_sc_to_matched_nodes(nodes)
    post_mean = pd.read_csv(FC5_SIDEKEY).set_index("side_key")["weighted_mean_PostDCA_positive"]
    nodes["PostDCA_positive"] = nodes["atlas_region"].map(post_mean)
    nodes = nodes.dropna(subset=["PostDCA_positive"]).copy()
    nodes["group_order"] = nodes["big_group"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    nodes = nodes.sort_values(["group_order", "PostDCA_positive", "atlas_region"], ascending=[True, False, True]).reset_index(drop=True)
    regions = nodes["atlas_region"].astype(str).tolist()
    sc = sc.loc[regions, regions].copy()

    x_values = nodes["PostDCA_positive"].to_numpy(dtype=float)
    x_min, x_max = float(np.nanmin(x_values)), float(np.nanmax(x_values))
    x_scaled = 0.06 + 0.96 * (x_max - x_values) / max(x_max - x_min, 1e-12)
    group_offsets = {g: i for i, g in enumerate(GROUP_ORDER)}
    y_base = np.array([len(GROUP_ORDER) - 1 - group_offsets[g] for g in nodes["big_group"]], dtype=float)
    y_jitter = ((np.arange(len(nodes)) % 7) - 3) * 0.035
    y_scaled = y_base + y_jitter
    x_pos = pd.Series(x_scaled, index=regions)
    y_pos = pd.Series(y_scaled, index=regions)

    edge_rows = []
    arr = sc.to_numpy(dtype=float)
    for i, src in enumerate(regions):
        row = arr[i].copy()
        row[i] = 0
        for j in np.where(row > 0)[0]:
            edge_rows.append((src, regions[j], row[j]))
    if edge_rows:
        max_edge = max(x[2] for x in edge_rows)
    else:
        max_edge = 1.0

    for src, dst, weight in edge_rows:
        x1, y1 = float(x_pos[src]), float(y_pos[src])
        x2, y2 = float(x_pos[dst]), float(y_pos[dst])
        if abs(x1 - x2) < 0.02 and abs(y1 - y2) < 0.02:
            continue
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=4.8,
            linewidth=0.16 + 0.34 * np.log1p(weight) / np.log1p(max_edge),
            color="#3f3f3f",
            alpha=0.18,
            shrinkA=3.2,
            shrinkB=3.2,
            zorder=1,
        )
        ax.add_patch(arrow)

    for group in GROUP_ORDER:
        sub = nodes[nodes["big_group"] == group]
        if sub.empty:
            continue
        xy = np.asarray([[x_pos[region], y_pos[region]] for region in sub["atlas_region"]])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=24,
            c=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
            label=SHORT_LABELS[group],
        )

    y_ticks = [len(GROUP_ORDER) - 1 - i for i in range(len(GROUP_ORDER))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([SHORT_LABELS[g] for g in GROUP_ORDER], fontsize=8.5)
    for tick, group in zip(ax.get_yticklabels(), GROUP_ORDER):
        tick.set_color(GROUP_COLORS[group])
        tick.set_fontweight("bold")

    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.72, len(GROUP_ORDER) - 0.28)

    arrow_y = -0.50
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
        Polygon(
            arrow_vertices,
            closed=True,
            facecolor="#5b5b5b",
            edgecolor="none",
            alpha=0.82,
            clip_on=False,
            zorder=6,
        )
    )
    ax.text(0.02, arrow_y - 0.15, "high", ha="left", va="top", fontsize=8, color="#222222")
    ax.text(0.92, arrow_y - 0.15, "low", ha="left", va="top", fontsize=8, color="#222222")
    ax.text(
        0.42,
        arrow_y - 0.24,
        "Post-DCA+",
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        fontweight="bold",
    )
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    label(ax, "A", x=-0.04, y=1.02)


def draw_trace_panel(ax: plt.Axes, ex: dict, panel: str, kind: str) -> None:
    t = np.arange(len(ex["trace_a"])) / TRACE_SAMPLING_RATE_HZ
    start = int(ex.get("display_start", 0))
    stop = int(ex.get("display_stop", min(len(t), 900)))
    tt = t[start:stop] - t[start]
    colors = TRACE_PAIR_COLORS[kind]
    offsets = [1.45, -1.45]
    labels = [ex["label_a"], ex["label_b"]]
    traces = [ex["trace_a"][start:stop], ex["trace_b"][start:stop]]
    for name, color, offset, trace in zip(labels, colors, offsets, traces):
        ax.plot(tt, trace + offset, color=color, lw=0.75)
        ax.text(tt[-1] + 8, offset, name, color=color, va="center", fontsize=8)
    ax.set_title(ex["tag"], fontsize=9, pad=2)
    ax.axis("off")
    ax.axvline(tt[-1], color="#999999", lw=0.7, ls=":")
    ax.set_yticks([])
    ax.set_ylabel("Calcium\nz-score", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xlim(tt[0], tt[-1] + 55)
    label(ax, panel, x=-0.12, y=1.08)


def draw_corr_panel(ax: plt.Axes, ex: dict, panel: str | None = None) -> None:
    start = int(ex.get("display_start", 0))
    stop = int(ex.get("display_stop", len(ex["trace_a"])))
    centers = np.asarray(ex["centers"], dtype=float)
    corr = np.asarray(ex["corr"], dtype=float)
    keep = (centers >= start) & (centers <= stop)
    t = centers[keep] / TRACE_SAMPLING_RATE_HZ - start / TRACE_SAMPLING_RATE_HZ
    ax.plot(t, corr[keep], color="#222222", lw=1.0)
    ax.axhline(0, color="#777777", lw=0.6, alpha=0.45)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, (stop - start) / TRACE_SAMPLING_RATE_HZ + 55)
    ax.set_xlabel("Time(s)", fontsize=8)
    ax.set_ylabel("FC", fontsize=8)
    ax.text(
        0.04,
        0.08,
        f"FCV={ex['global_fcv']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    ax.tick_params(axis="both", labelsize=7, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    if panel is not None:
        label(ax, panel, x=-0.12, y=1.08)


def draw_representation_row(fig: plt.Figure, subspec) -> None:
    high, low = choose_trace_examples()
    gs = subspec.subgridspec(
        1,
        3,
        width_ratios=[1.55, 1.0, 1.0],
        wspace=0.38,
    )
    ax_network = fig.add_subplot(gs[0, 0])
    high_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[1.35, 0.8], hspace=0.18)
    low_gs = gs[0, 2].subgridspec(2, 1, height_ratios=[1.35, 0.8], hspace=0.18)
    ax_high_trace = fig.add_subplot(high_gs[0, 0])
    ax_high_corr = fig.add_subplot(high_gs[1, 0])
    ax_low_trace = fig.add_subplot(low_gs[0, 0])
    ax_low_corr = fig.add_subplot(low_gs[1, 0])
    draw_ito48_network(ax_network)
    draw_trace_panel(ax_high_trace, high, "B", "high")
    draw_trace_panel(ax_low_trace, low, "C", "low")
    draw_corr_panel(ax_high_corr, high)
    draw_corr_panel(ax_low_corr, low)


def draw_fc_matrix(fig: plt.Figure, ax: plt.Axes, order: pd.DataFrame, panel: str = "C") -> None:
    nodes = matched_node_table()
    fc_df = aggregate_999_matrix_to_matched_nodes(FC_999, nodes)
    fc = fc_df.to_numpy(dtype=float)
    lim = np.nanpercentile(np.abs(fc), 99)
    return add_heatmap(fig, ax, fc, "FC", "coolwarm", panel, vmin=-lim, vmax=lim)


def draw_fcv_matrix(fig: plt.Figure, ax: plt.Axes, order: pd.DataFrame, panel: str = "D") -> None:
    nodes = matched_node_table()
    fcv_df = aggregate_999_matrix_to_matched_nodes(FCV_999, nodes)
    fcv = fcv_df.to_numpy(dtype=float)
    return add_heatmap(fig, ax, fcv, "FCV", "coolwarm", panel, vmin=0.2, vmax=0.3)


def draw_best_scatter(ax: plt.Axes) -> None:
    y_col = "mean_FCV_excl_same_side_key_z"
    df = pd.read_csv(SCATTER_SCORES).dropna(subset=["weighted_mean_PostDCA_positive", y_col])
    for group in GROUP_ORDER:
        sub = df[df["big_group"] == group]
        if sub.empty:
            continue
        ax.scatter(
            sub["weighted_mean_PostDCA_positive"],
            sub[y_col],
            s=16,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.25,
            alpha=0.66,
            label=SHORT_LABELS[group],
        )
    x = df["weighted_mean_PostDCA_positive"].to_numpy()
    y = df[y_col].to_numpy()
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.0)
    sp = spearmanr(x, y)
    pr = pearsonr(x, y)
    ax.text(
        0.4,
        0.2,
        f"rho={sp.statistic:.2f}, p={sp.pvalue:.2g}\n"
        f"r={pr.statistic:.2f}, p={pr.pvalue:.2g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.axhline(0, color="#777777", lw=0.6, alpha=0.35, zorder=0)
    ax.axvline(0, color="#777777", lw=0.6, alpha=0.35, zorder=0)
    ax.set_xlabel("Weighted mean Post-DCA+")
    ax.set_ylabel("zFCV")
    style_open_axes(ax)
    label(ax, "G", x=-0.10)


def draw_group_box_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    panel: str,
    point_size: float = 11.0,
    point_alpha: float = 0.72,
    jitter_width: float = 0.18,
    sig_start_y: float | None = None,
) -> None:
    data = df[df["big_group"].isin(GROUP_ORDER)].copy()
    positions = np.arange(len(GROUP_ORDER))
    box_data = [data.loc[data["big_group"] == g, value_col].dropna().to_numpy(dtype=float) for g in GROUP_ORDER]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "lw": 1.0},
        whiskerprops={"color": "0.35", "lw": 1.0},
        capprops={"color": "0.35", "lw": 1.0},
    )
    for patch, group in zip(bp["boxes"], GROUP_ORDER):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(1.0)
        patch.set_edgecolor("0.25")
        patch.set_linewidth(1.0)
    rng = np.random.default_rng(15)
    for i, group in enumerate(GROUP_ORDER):
        vals = box_data[i]
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-jitter_width, jitter_width, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=point_size,
            color="black",
            edgecolor="none",
            linewidth=0.0,
            alpha=point_alpha,
            rasterized=True,
        )
    ax.axhline(0, color="#777777", lw=0.65, alpha=0.45, zorder=0)
    ax.set_xticks(positions, [SHORT_LABELS[g] for g in GROUP_ORDER], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    style_open_axes(ax, tick_length=4.0, tick_width=1.2)
    add_sig_bars(ax, box_data, start_y=sig_start_y)
    label(ax, panel, x=-0.33)


def load_recording_fcv_points_for_j() -> pd.DataFrame:
    rec = pd.read_csv(FC5_RECORDING, usecols=["recording_id", "roi", "label", "FCV_z"])
    groups = pd.read_csv(SCATTER_SCORES, usecols=["roi", "label", "big_group"])
    out = rec.merge(groups, on=["roi", "label"], how="inner")
    return out.dropna(subset=["FCV_z", "big_group"]).copy()


def match_panel_heights(reference: plt.Axes, targets: list[plt.Axes]) -> None:
    ref_pos = reference.get_position()
    ref_center = ref_pos.y0 + ref_pos.height / 2
    for ax in targets:
        pos = ax.get_position()
        new_y0 = ref_center - ref_pos.height / 2
        ax.set_position([pos.x0, new_y0, pos.width, ref_pos.height])


def draw_summary_row(fig: plt.Figure, subspec, order: pd.DataFrame) -> None:
    gs = subspec.subgridspec(
        1,
        6,
        width_ratios=[1.0, 1.0, 1.0, 1.18, 1.18, 1.18],
        wspace=0.42,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    matrix_cbars = [
        draw_sc_matrix(fig, axes[0], "D"),
        draw_fc_matrix(fig, axes[1], order, "E"),
        draw_fcv_matrix(fig, axes[2], order, "F"),
    ]
    draw_best_scatter(axes[3])
    j_points = load_recording_fcv_points_for_j()
    k_points = pd.read_csv(SCATTER_SCORES).dropna(subset=["weighted_mean_PostDCA_positive", "big_group"])
    draw_group_box_panel(
        axes[4],
        j_points,
        "FCV_z",
        "FCV z",
        "H",
        point_size=3.0,
        point_alpha=0.10,
        jitter_width=0.22,
        sig_start_y=4.0,
    )
    axes[4].set_ylim(top=5.0)
    draw_group_box_panel(
        axes[5],
        k_points,
        "weighted_mean_PostDCA_positive",
        "Post-DCA+",
        "I",
        point_size=3.0,
        point_alpha=0.20,
        jitter_width=0.22,
    )
    fig.canvas.draw()
    for ax, cbar in zip(axes[:3], matrix_cbars):
        ax_pos = ax.get_position()
        cbar_pos = cbar.ax.get_position()
        cbar.ax.set_position([cbar_pos.x0, ax_pos.y0, cbar_pos.width, ax_pos.height])
    match_panel_heights(axes[2], axes[3:])


def measure_cluster_order(df: pd.DataFrame, measure_cols: list[str], score_col: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    values = df[measure_cols].to_numpy(float).T
    z_values = zscore_rows_for_plot(values)
    features = z_values.T
    finite = np.where(np.isfinite(features), features, np.nanmean(features, axis=0))
    finite = np.nan_to_num(finite, nan=0.0)
    z_linkage = linkage(finite, method="ward", optimal_ordering=True)
    scores = df[score_col].to_numpy(float)
    z_linkage = orient_linkage_high_score_left(z_linkage, scores)
    leaves = dendrogram(z_linkage, no_plot=True)["leaves"]
    left = np.nanmean(scores[leaves[: max(1, len(leaves) // 2)]])
    right = np.nanmean(scores[leaves[max(1, len(leaves) // 2) :]])
    if right > left:
        leaves = list(reversed(leaves))
    return leaves, z_linkage, z_values


def draw_measure_heatmap_panel(
    fig: plt.Figure,
    subspec,
    df: pd.DataFrame,
    measure_cols: list[str],
    measure_labels: list[str],
    name_col: str,
    score_col: str,
    panel: str,
    title: str,
    xlabel: str,
) -> None:
    plot = df.dropna(subset=measure_cols, how="all").copy()
    plot = plot[plot["big_group"].isin(GROUP_ORDER)].reset_index(drop=True)
    leaves, z_linkage, z_values = measure_cluster_order(plot, measure_cols, score_col)
    ordered = plot.iloc[leaves].reset_index(drop=True)
    plot_values = z_values[:, leaves]

    gs = GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=subspec,
        height_ratios=[0.30, 0.085, 1.0],
        width_ratios=[1.0, 0.030],
        hspace=0.04,
        wspace=0.035,
    )
    ax_d = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[2, 0])
    cax = fig.add_subplot(gs[2, 1])

    dend = dendrogram(z_linkage, no_plot=True)
    desired_pos = {leaf: pos for pos, leaf in enumerate(leaves)}
    for icoord, dcoord in zip(dend["icoord"], dend["dcoord"]):
        xs = []
        for x in icoord:
            old_leaf_pos = int(round((x - 5.0) / 10.0))
            leaf = dend["leaves"][old_leaf_pos]
            xs.append(desired_pos[leaf])
        ax_d.plot(xs, dcoord, color="black", lw=0.7)
    ax_d.set_xlim(-0.5, len(ordered) - 0.5)
    ax_d.axis("off")
    ax_d.set_title(title, fontsize=9, pad=2)
    label(ax_d, panel, x=-0.035, y=0.95)

    colors = [GROUP_COLORS.get(g, "#999999") for g in ordered["big_group"]]
    ax_bar.imshow(np.array([range(len(ordered))]), aspect="auto", cmap="gray", alpha=0)
    for i, color in enumerate(colors):
        ax_bar.add_patch(plt.Rectangle((i - 0.5, -0.5), 1.0, 1.0, facecolor=color, edgecolor="none"))
    ax_bar.set_xlim(-0.5, len(ordered) - 0.5)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.axis("off")

    im = ax.imshow(plot_values, cmap="PiYG_r", vmin=-2, vmax=2, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(measure_labels)), measure_labels)
    ax.set_xticks(np.arange(len(ordered)), ordered[name_col].astype(str), rotation=90, fontsize=4.9)
    ax.tick_params(axis="x", length=0, pad=1.2)
    ax.axhline(-0.5, color="black", lw=0.8)
    ax.axhline(len(measure_labels) - 0.5, color="black", lw=0.8)
    ax.axvline(-0.5, color="black", lw=0.8)
    ax.axvline(len(ordered) - 0.5, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("row z-score", fontsize=6.5)
    cb.set_ticks([-2, -1, 0, 1, 2])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cb.ax.tick_params(labelsize=6.5, length=2.5, width=0.6)


def draw_heatmap_row(fig: plt.Figure, subspec) -> None:
    gs = subspec.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.15)
    sc = pd.read_csv(SC_MEASURE_RESULTS)
    fc = pd.read_csv(FC5_SIDEKEY)
    roi_summary = pd.read_csv(FC5_ROI_SUMMARY)
    sidekey_map = pd.read_csv(SCATTER_SCORES)[["roi", "label", "side_key"]].drop_duplicates()
    raw_meta = (
        sidekey_map.merge(roi_summary[["roi", "label", "Metastability"]], on=["roi", "label"], how="left")
        .groupby("side_key", as_index=False)
        .agg(Metastability=("Metastability", "mean"))
    )
    fc = fc.drop(columns=["Metastability"], errors="ignore").merge(raw_meta, on="side_key", how="left")
    draw_measure_heatmap_panel(
        fig,
        gs[0, 0],
        sc,
        SC_MEASURE_COLS,
        SC_MEASURE_LABELS,
        "region",
        "mean_PostDCA_positive",
        "J",
        "FlyWire783 SC measures",
        "Drosophila regions",
    )
    draw_measure_heatmap_panel(
        fig,
        gs[0, 1],
        fc,
        FC_MEASURE_COLS,
        FC_MEASURE_LABELS,
        "side_key",
        "FCV_z",
        "K",
        "Branson999 FC measures",
        "Drosophila side-aware blocks",
    )


def draw_method_stats(ax: plt.Axes) -> None:
    stats = pd.read_csv(METHOD_STATS)
    rows = stats[(stats["subset"] == "all") & stats["method"].isin(["mean_PostDCA", "median_PostDCA_positive", "mean_PostDCA_positive", "weighted_PostDCA_positive"])].copy()
    label_map = {
        "mean_PostDCA": "Signed mean",
        "median_PostDCA_positive": "Median +",
        "mean_PostDCA_positive": "Mean +",
        "weighted_PostDCA_positive": "Weighted +",
    }
    rows["short"] = rows["method"].map(label_map)
    rows = rows.set_index("method").loc[list(label_map)].reset_index()
    colors = ["#9a9a9a", "#6baed6", "#2171b5", "#08519c"]
    ax.bar(np.arange(len(rows)), rows["spearman_rho"], color=colors, alpha=0.92)
    ax.axhline(0, color="black", lw=0.8)
    for i, row in rows.iterrows():
        ax.text(i, row["spearman_rho"] + (0.025 if row["spearman_rho"] >= 0 else -0.045), f"p={row['spearman_p']:.1e}", ha="center", va="bottom" if row["spearman_rho"] >= 0 else "top", fontsize=6)
    ax.set_xticks(np.arange(len(rows)), rows["short"], rotation=25, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Post-DCA definition check")
    ax.spines[["top", "right"]].set_visible(False)
    label(ax, "F", x=-0.10)


def draw_weight_control(ax: plt.Axes) -> None:
    stats = pd.read_csv(WEIGHT_BINARY_STATS)
    rows = stats[(stats["subset"] == "all") & stats["method"].isin(["weighted_mean_PostDCA_positive", "binary_mean_PostDCA_positive"])].copy()
    rows = rows.set_index("method").loc[["weighted_mean_PostDCA_positive", "binary_mean_PostDCA_positive"]].reset_index()
    rows["short"] = ["Weighted SC", "Binary SC"]
    ax.bar(np.arange(len(rows)), rows["spearman_rho"], color=["#2171b5", "#bdbdbd"], alpha=0.95)
    ax.axhline(0, color="black", lw=0.8)
    for i, row in rows.iterrows():
        ax.text(i, row["spearman_rho"] + (0.025 if row["spearman_rho"] >= 0 else -0.045), f"p={row['spearman_p']:.1e}", ha="center", va="bottom" if row["spearman_rho"] >= 0 else "top", fontsize=6)
    ax.set_xticks(np.arange(len(rows)), rows["short"], rotation=15, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-0.18, 0.62)
    ax.set_title("Synapse-weight control")
    ax.spines[["top", "right"]].set_visible(False)
    label(ax, "G", x=-0.10)


def main() -> None:
    set_style()
    order = pd.read_csv(ORDER_999)

    fig = plt.figure(figsize=(16.0, 7.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.02, 1.05], hspace=0.18)
    draw_representation_row(fig, outer[0])
    draw_summary_row(fig, outer[1], order)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")
    print("Final result remembered: w15 step5, 0.03 Hz high-pass, weighted_mean_PostDCA_positive, synapse-count weighted SC, Branson999-first FC measures.")


if __name__ == "__main__":
    main()
