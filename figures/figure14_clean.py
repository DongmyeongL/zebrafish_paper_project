"""Figure 14 - Post-DCA / output-to-output motif validation.

This analysis tests whether regional Post-DCA tracks broadcast-to-broadcast
inter-regional cascade structure.  For each subject and source region, it
counts outgoing inter-regional edges whose source and target neurons are both
output-dominated within their own regional subnetworks.

By default the script uses the saved subject-level DCA values
(`subject*_cp_cont_data.npz`) with the historical pre/post swap corrected
as cp_in - cp_out.  The optional `--include-raw` flag recomputes an
additional sparse raw-DCA robustness variant.
"""

from __future__ import annotations

import os
import pickle
import argparse
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, rankdata, spearmanr, t

import figure_style as fs
import figure_fc_dynamics_final_fc_shift_summary as fc_shift_summary
import figure_regionwise_multivariate_coupling as coupling

# Workflow:
# 1. Data loading and plotting calculations.
# 2. Layout preparation.
# 3. Draw each panel.
# 4. Panel position adjustment and panel labels.
# 5. Save figure and statistics.


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG14_DATA_DIR = DATA_DIR / "figure14"
OUTPUT_DIR = PROJECT_ROOT / "output"
PNG_DIR = OUTPUT_DIR / "png"
PDF_DIR = OUTPUT_DIR / "pdf"
STATS_DIR = OUTPUT_DIR / "stats"
RAW_DIR = BASE_DIR.parents[3] / "original_raw_data"
DCA_DIR = FIG14_DATA_DIR
FCV_FILE = DATA_DIR / "fig1_prism_D_FCS_FCV_bar.csv"
REGION_DCA_FILE = DATA_DIR / "total_selected_region_dac_data.npz"
REGION_SELECTION_FILE = DATA_DIR / "figure13" / "figure1_whole_brain_fc_mean_std_data.npz"
OUT_PNG = PNG_DIR / "figure14_final.png"
OUT_PDF = PDF_DIR / "figure14_final.pdf"
OUT_SUBJECT_SUMMARY = FIG14_DATA_DIR / "figure14_postdca_oo_motif_summary.csv"
OUT_REGION_MEAN = FIG14_DATA_DIR / "figure14_postdca_oo_motif_region_mean.csv"
OUT_STATS = STATS_DIR / "figure14_postdca_oo_motif_stats.csv"
OUT_SUBJECT_STATS = STATS_DIR / "figure14_postdca_oo_motif_subject_stats.csv"
OUT_STATS_TXT = STATS_DIR / "figure14_postdca_oo_motif_stats.txt"
SUBJECTS = list(range(12, 19))
N_REGION = len(fs.region)

DIVISION_LABELS = {
    0: "Hind",
    1: "Di",
    2: "Tel",
    3: "Mes",
}
DIVISION_COLORS = {
    "Hind": fs.division_colors[0],
    "Di": fs.division_colors[1],
    "Tel": fs.division_colors[2],
    "Mes": fs.division_colors[3],
}
DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DCA_POST_LABEL = r"$\mathrm{DCA}_{\mathrm{post}}$"


fs.set_paper_style()
plt.rcParams.update({
    "font.size": fs.AXIS_LABEL_FS_2COL,
    "axes.labelsize": fs.AXIS_LABEL_FS_2COL,
    "axes.titlesize": fs.AXIS_LABEL_FS_2COL,
    "xtick.labelsize": fs.TICK_FS_2COL,
    "ytick.labelsize": fs.TICK_FS_2COL,
})


# 1. Data loading and plotting calculations

def _safe_pearson(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan, int(mask.sum())
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def _safe_spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan
    r, p = spearmanr(x[mask], y[mask])
    return float(r), float(p)


def _format_stat_text(r, p):
    if not np.isfinite(p):
        p_line = "p = nan"
    elif p < 0.001:
        p_line = "p < 0.001"
    else:
        p_line = f"p = {p:.3f}"
    return f"r = {r:.3f}\n{p_line}"


def _residualize(y, covariate):
    mask = np.isfinite(y) & np.isfinite(covariate)
    residual = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 3 or np.nanstd(covariate[mask]) == 0:
        return residual
    X = np.column_stack([np.ones(mask.sum()), covariate[mask]])
    beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    residual[mask] = y[mask] - X @ beta
    return residual


def _rank1_sparse_power(subgraph, normalize=True, n_iter=300, tol=1e-7, seed=0):
    rng = np.random.default_rng(seed)
    n = subgraph.shape[0]
    c_out = np.abs(rng.standard_normal(n)) + 1e-8
    c_in = np.abs(rng.standard_normal(n)) + 1e-8
    c_out /= np.linalg.norm(c_out) + 1e-12
    c_in /= np.linalg.norm(c_in) + 1e-12

    for _ in range(n_iter):
        next_out = np.asarray(subgraph @ c_in).ravel()
        next_out = np.maximum(next_out, 0.0)
        out_norm = np.linalg.norm(next_out)
        if out_norm > 1e-12:
            next_out /= out_norm

        next_in = np.asarray(subgraph.T @ next_out).ravel()
        next_in = np.maximum(next_in, 0.0)
        in_norm = np.linalg.norm(next_in)
        if in_norm > 1e-12:
            next_in /= in_norm

        if (
            np.linalg.norm(next_out - c_out) < tol
            and np.linalg.norm(next_in - c_in) < tol
        ):
            c_out, c_in = next_out, next_in
            break
        c_out, c_in = next_out, next_in

    if not normalize:
        sigma = float(c_out @ (subgraph @ c_in))
        scale = np.sqrt(max(sigma, 0.0))
        c_out = c_out * scale
        c_in = c_in * scale
    return c_out, c_in


def _rank_dca_within_regions(dca, neuron_region):
    dca_rank = np.full_like(dca, np.nan, dtype=float)
    for region_id in np.unique(neuron_region):
        idx = np.flatnonzero(neuron_region == region_id)
        valid = idx[np.isfinite(dca[idx])]
        if len(valid) < 2:
            continue
        ranks = rankdata(dca[valid], method="average")
        dca_rank[valid] = (ranks - 1.0) / (len(valid) - 1.0)
    return dca_rank


def _raw_dca_from_edges(edges, neuron_region, n_neurons, seed):
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(n_neurons, n_neurons),
    )
    dca = np.full(n_neurons, np.nan, dtype=float)
    for region_id in np.unique(neuron_region):
        idx = np.flatnonzero(neuron_region == region_id)
        if len(idx) < 2:
            continue
        sub = graph[idx, :][:, idx]
        if sub.nnz == 0:
            continue
        c_out, c_in = _rank1_sparse_power(
            sub, normalize=False, seed=int(seed + int(region_id))
        )
        dca[idx] = c_out - c_in
    return dca


def _motif_metrics(edges, neuron_region, dca, threshold, output_polarity="positive"):
    src = edges[:, 0]
    tgt = edges[:, 1]
    src_region = neuron_region[src]
    tgt_region = neuron_region[tgt]
    inter = src_region != tgt_region

    src_ir = src[inter]
    tgt_ir = tgt[inter]
    src_region_ir = src_region[inter]

    valid = np.isfinite(dca[src_ir]) & np.isfinite(dca[tgt_ir])
    if output_polarity == "positive":
        src_output = dca[src_ir] > threshold
        tgt_output = dca[tgt_ir] > threshold
    elif output_polarity == "negative":
        src_output = dca[src_ir] < threshold
        tgt_output = dca[tgt_ir] < threshold
    else:
        raise ValueError(f"Unknown output_polarity: {output_polarity}")

    total_out = np.bincount(src_region_ir, minlength=N_REGION)
    motif_masks = {
        "OO": valid & src_output & tgt_output,
        "ON": valid & src_output & ~tgt_output,
        "NO": valid & ~src_output & tgt_output,
        "NN": valid & ~src_output & ~tgt_output,
    }
    counts = {
        name: np.bincount(src_region_ir[mask], minlength=N_REGION).astype(float)
        for name, mask in motif_masks.items()
    }
    fractions = {}
    for name, count in counts.items():
        frac = np.full(N_REGION, np.nan, dtype=float)
        ok = total_out > 0
        frac[ok] = count[ok] / total_out[ok]
        fractions[name] = frac
    return counts, fractions, total_out.astype(float)


def _load_region_dca():
    dca = np.load(REGION_DCA_FILE, allow_pickle=True)
    # Match Figure 12 panels G/H and fig3_prism_A_PostDCA.csv:
    # arr_2 is the regional dac_in quantity labeled as Post-DCA.
    # arr_1 is the regional dac_out quantity labeled as Pre-DCA.
    pre_dca = np.asarray(dca["arr_1"], dtype=float)
    post_dca = np.asarray(dca["arr_2"], dtype=float)
    if pre_dca.shape != (N_REGION, len(SUBJECTS)):
        raise ValueError(f"Unexpected Pre-DCA shape: {pre_dca.shape}")
    if post_dca.shape != (N_REGION, len(SUBJECTS)):
        raise ValueError(f"Unexpected Post-DCA shape: {post_dca.shape}")
    return pre_dca, post_dca


def _load_region_fcv():
    fcv_df = pd.read_csv(FCV_FILE)
    return {
        str(row["Region"]): float(row["FCV"])
        for _, row in fcv_df.iterrows()
        if np.isfinite(float(row["FCV"]))
    }


def _selected_region_ids():
    load = np.load(REGION_SELECTION_FILE, allow_pickle=True)
    region_ids = np.asarray(load["new_region_array"], dtype=int)
    region_ids = np.array(
        [r for r in region_ids if r < N_REGION and fs.brain_division_list[r] < 4],
        dtype=int,
    )
    return region_ids


def _load_subject(
    subject,
    edge_convention="col0_to_col1",
    dca_convention="prepost_swapped",
):
    raw_path = RAW_DIR / f"subject_{subject}_data_cellular_synapse_sc_100_data.pkl"
    with open(raw_path, "rb") as handle:
        raw = pickle.load(handle)

    edges = np.asarray(raw["cellular_sc_list"], dtype=np.int32)
    if edge_convention == "col1_to_col0":
        edges = edges[:, ::-1].copy()
    elif edge_convention != "col0_to_col1":
        raise ValueError(f"Unknown edge convention: {edge_convention}")
    neuron_region_cluster = np.asarray(raw["neuron_region_id"], dtype=np.int32)
    root_area = np.asarray(raw["root_area"], dtype=np.int32)
    neuron_region = root_area[neuron_region_cluster]

    dca_npz = np.load(DCA_DIR / f"subject{subject}_cp_cont_data.npz")
    cp_in = np.asarray(dca_npz["arr_0"], dtype=float)
    cp_out = np.asarray(dca_npz["arr_1"], dtype=float)
    if dca_convention == "stored_labels":
        unit_dca = cp_out - cp_in
    elif dca_convention == "prepost_swapped":
        # The saved labels follow the historical scripts, but the original
        # pre/post convention was reversed during DCA generation.  This maps
        # output-dominated neurons back to positive DCA.
        unit_dca = cp_in - cp_out
    else:
        raise ValueError(f"Unknown dca_convention: {dca_convention}")

    if len(unit_dca) != len(neuron_region):
        raise ValueError(
            f"Subject {subject}: DCA length {len(unit_dca)} does not match "
            f"neuron_region length {len(neuron_region)}"
        )
    return edges, neuron_region, unit_dca


def build_validation_table(
    include_raw=True,
    edge_convention="col0_to_col1",
    dca_convention="prepost_swapped",
    output_polarity="positive",
):
    selected_regions = _selected_region_ids()
    pre_dca, post_dca = _load_region_dca()
    rows = []

    for subject_pos, subject in enumerate(SUBJECTS):
        print(f"Subject {subject}: loading raw edges and DCA")
        edges, neuron_region, unit_dca = _load_subject(
            subject,
            edge_convention=edge_convention,
            dca_convention=dca_convention,
        )
        rank_dca = _rank_dca_within_regions(unit_dca, neuron_region)

        variants = [
            ("unit_norm", unit_dca, 0.0),
            ("rank", rank_dca, 0.5),
        ]
        if include_raw:
            print(f"Subject {subject}: computing sparse raw-DCA robustness")
            raw_dca = _raw_dca_from_edges(
                edges, neuron_region, len(neuron_region), seed=subject
            )
            variants.insert(1, ("raw", raw_dca, 0.0))

        for variant_name, dca, threshold in variants:
            counts, fractions, total_out = _motif_metrics(
                edges, neuron_region, dca, threshold, output_polarity=output_polarity
            )
            for region_id in selected_regions:
                division_id = int(fs.brain_division_list[region_id])
                row = {
                    "Subject": subject,
                    "RegionID": int(region_id),
                    "Region": fs.region[region_id],
                    "Division": DIVISION_LABELS[division_id],
                    "Variant": variant_name,
                    "PreDCA": float(pre_dca[region_id, subject_pos]),
                    "PostDCA": float(post_dca[region_id, subject_pos]),
                    "InterOutDegree": float(total_out[region_id]),
                    "LogInterOutDegree": float(np.log10(total_out[region_id] + 1.0)),
                }
                for motif_name in ("OO", "ON", "NO", "NN"):
                    row[f"{motif_name}_count"] = float(counts[motif_name][region_id])
                    row[f"{motif_name}_fraction"] = float(
                        fractions[motif_name][region_id]
                    )
                rows.append(row)

    df = pd.DataFrame(rows)
    df.insert(0, "EdgeConvention", edge_convention)
    df.insert(1, "DCAConvention", dca_convention)
    df.insert(2, "OutputPolarity", output_polarity)
    FIG14_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_SUBJECT_SUMMARY, index=False)
    return df


def average_regions(df):
    """Average subject-level rows so each anatomical region contributes one point."""
    fcv_by_region = _load_region_fcv()
    id_cols = [
        "EdgeConvention",
        "DCAConvention",
        "OutputPolarity",
        "Variant",
        "RegionID",
        "Region",
        "Division",
    ]
    value_cols = [
        "PreDCA",
        "PostDCA",
        "InterOutDegree",
        "OO_count",
        "OO_fraction",
        "ON_count",
        "ON_fraction",
        "NO_count",
        "NO_fraction",
        "NN_count",
        "NN_fraction",
    ]
    out = (
        df.groupby(id_cols, as_index=False)[value_cols]
        .mean(numeric_only=True)
        .sort_values(["Variant", "RegionID"])
    )
    out["LogInterOutDegree"] = np.log10(out["InterOutDegree"] + 1.0)
    out["FCV"] = out["Region"].map(fcv_by_region)
    FIG14_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_REGION_MEAN, index=False)
    return out


def _ensure_fcv_column(df):
    if "FCV" in df.columns and df["FCV"].notna().any():
        return df
    out = df.copy()
    out["FCV"] = out["Region"].map(_load_region_fcv())
    return out


def compute_stats(df):
    rows = []
    target_cols = ["PostDCA", "PreDCA"]
    if "FCV" in df.columns:
        target_cols.insert(1, "FCV")
    for variant in sorted(df["Variant"].unique()):
        sub = df[df["Variant"] == variant].copy()
        for target in target_cols:
            for metric in [
                "OO_count",
                "OO_fraction",
                "ON_count",
                "ON_fraction",
                "NO_count",
                "NO_fraction",
                "NN_count",
                "NN_fraction",
                "InterOutDegree",
            ]:
                x = np.log10(sub[metric].to_numpy(float) + 1.0) if metric.endswith("count") or metric == "InterOutDegree" else sub[metric].to_numpy(float)
                y = sub[target].to_numpy(float)
                r, p, n = _safe_pearson(x, y)
                rho, sp = _safe_spearman(x, y)
                rows.append({
                    "Target": target,
                    "Variant": variant,
                    "Metric": metric,
                    "Transform": "log10(x+1)" if metric.endswith("count") or metric == "InterOutDegree" else "none",
                    "Pearson_r": r,
                    "Pearson_p": p,
                    "Spearman_rho": rho,
                    "Spearman_p": sp,
                    "N": n,
                    "Level": "region_mean",
                })

            x = np.log10(sub["OO_count"].to_numpy(float) + 1.0)
            degree = sub["LogInterOutDegree"].to_numpy(float)
            y = sub[target].to_numpy(float)
            x_resid = _residualize(x, degree)
            y_resid = _residualize(y, degree)
            r, p, n = _safe_pearson(x_resid, y_resid)
            rho, sp = _safe_spearman(x_resid, y_resid)
            rows.append({
                "Target": target,
                "Variant": variant,
                "Metric": "OO_count_partial_degree",
                "Transform": "log10(x+1), residualized by log inter-out-degree",
                "Pearson_r": r,
                "Pearson_p": p,
                "Spearman_rho": rho,
                "Spearman_p": sp,
                "N": n,
                "Level": "region_mean",
            })

    stats = pd.DataFrame(rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_STATS, index=False)
    return stats


def compute_subject_stats(df):
    rows = []
    for variant in sorted(df["Variant"].unique()):
        variant_df = df[df["Variant"] == variant].copy()
        for subject in sorted(variant_df["Subject"].unique()):
            sub = variant_df[variant_df["Subject"] == subject].copy()
            for target in ["PostDCA", "PreDCA"]:
                for metric in [
                    "OO_fraction",
                    "ON_fraction",
                    "NO_fraction",
                    "NN_fraction",
                    "InterOutDegree",
                ]:
                    if metric == "InterOutDegree":
                        x = np.log10(sub[metric].to_numpy(float) + 1.0)
                        transform = "log10(x+1)"
                    else:
                        x = sub[metric].to_numpy(float)
                        transform = "none"
                    y = sub[target].to_numpy(float)
                    r, p, n = _safe_pearson(x, y)
                    rho, sp = _safe_spearman(x, y)
                    rows.append({
                        "Subject": int(subject),
                        "Target": target,
                        "Variant": variant,
                        "Metric": metric,
                        "Transform": transform,
                        "Pearson_r": r,
                        "Pearson_p": p,
                        "Spearman_rho": rho,
                        "Spearman_p": sp,
                        "N": n,
                        "Level": "subject",
                    })

    subject_stats = pd.DataFrame(rows)
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    subject_stats.to_csv(OUT_SUBJECT_STATS, index=False)
    return subject_stats


def _scatter_panel(
    ax,
    df,
    x_col,
    y_col="PostDCA",
    log_x=False,
    title="",
    xlabel="",
    ylabel="Post-DCA",
    label_regions=False,
    xlim_min=None,
):
    x = df[x_col].to_numpy(float)
    if log_x:
        x = np.log10(x + 1.0)
    y = df[y_col].to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    plot_df = df.loc[mask].copy()
    plot_df["_plot_x"] = x[mask]
    plot_df["_plot_y"] = y[mask]
    x_plot = x[mask]
    y_plot = y[mask]

    if mask.sum() >= 3 and np.nanstd(x_plot) > 0:
        m, b = np.polyfit(x_plot, y_plot, 1)
        xs = np.linspace(x_plot.min() - 0.15, x_plot.max() + 0.15, 100)
        ys = m * xs + b
        n_obs = x_plot.size
        x_mean = np.mean(x_plot)
        sxx = np.sum((x_plot - x_mean) ** 2)
        if n_obs > 2 and sxx > 0:
            resid = y_plot - (m * x_plot + b)
            mse = np.sum(resid ** 2) / (n_obs - 2)
            se_mean = np.sqrt(mse * (1.0 / n_obs + (xs - x_mean) ** 2 / sxx))
            ci = t.ppf(0.975, n_obs - 2) * se_mean
            ax.fill_between(xs, ys - ci, ys + ci, color="#4d4d4d", alpha=0.16, linewidth=0, zorder=1)
        ax.plot(xs, ys, color="#4d4d4d", lw=1.8, zorder=2)

    for division in DIVISION_ORDER:
        div_mask = plot_df["Division"].astype(str).to_numpy() == division
        if not np.any(div_mask):
            continue
        is_tel = division == "Tel"
        ax.scatter(
            x_plot[div_mask],
            y_plot[div_mask],
            s=64 if is_tel else 42,
            color=DIVISION_COLORS[division],
            alpha=0.90 if is_tel else 0.78,
            edgecolor="black" if is_tel else "none",
            linewidth=0.4,
            label=division,
        )

    if label_regions:
        label_df = plot_df[plot_df["Division"].astype(str) == "Tel"]
        for row in label_df.itertuples(index=False):
            ax.annotate(
                row.Region,
                (row._plot_x, row._plot_y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=fs.TICK_FS_2COL - 2,
                color=DIVISION_COLORS["Tel"],
                fontweight="bold",
            )

    ax.axhline(0, color="#dddddd", lw=1.0, zorder=0)
    ax.axvline(0, color="#dddddd", lw=1.0, zorder=0)
    r, p, _ = _safe_pearson(x, y)
    ax.text(
        0.03,
        0.97,
        _format_stat_text(r, p),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fs.STAT_FS_2COL + 2,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=fs.TICK_FS_2COL, pad=2)
    ax.xaxis.label.set_size(fs.AXIS_LABEL_FS_2COL)
    ax.yaxis.label.set_size(fs.AXIS_LABEL_FS_2COL)
    ax.margins(x=0.12, y=0.14)
    if xlim_min is not None:
        _, xmax = ax.get_xlim()
        ax.set_xlim(float(xlim_min), xmax)


# 2. Layout preparation

def prepare_layout():
    fig = plt.figure(figsize=(16, 4.0))
    gs = GridSpec(
        1,
        5,
        figure=fig,
        left=0.045,
        right=0.99,
        top=0.84,
        bottom=0.22,
        wspace=0.46,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    ax_e = fig.add_subplot(gs[0, 4])
    axes = {
        "a": ax_a,
        "b": ax_b,
        "c": ax_c,
        "d": ax_d,
        "e": ax_e,
    }
    return fig, axes


# 3. Draw each panel

def draw_panel_a(ax_a, unit):
    _scatter_panel(
        ax_a,
        unit,
        "OO_fraction",
        y_col="PostDCA",
        log_x=False,
        xlabel="OO fraction",
        ylabel=DCA_POST_LABEL,
        xlim_min=0.1,
    )


def draw_panel_b(ax_b, unit):
    _scatter_panel(
        ax_b,
        unit,
        "OO_fraction",
        y_col="FCV",
        log_x=False,
        xlabel="OO fraction",
        ylabel="zFCV",
        xlim_min=0.1,
    )


def draw_panel_c(ax_c, fc_shift_mean_df):
    _scatter_panel(
        ax_c,
        fc_shift_mean_df,
        "SpontaneousFCV",
        y_col="MeanStimulusFCV",
        xlabel="zFCV",
        ylabel="Stimulus zFCV",
    )


def draw_panel_d(ax_d, fc_shift_mean_df):
    _scatter_panel(
        ax_d,
        fc_shift_mean_df,
        "SpontaneousFCV",
        y_col="StdStimulusFCVZ",
        xlabel="zFCV",
        ylabel="SD stimulus zFCV",
    )


def draw_panel_e(ax_e, fc_shift_mean_df):
    _scatter_panel(
        ax_e,
        fc_shift_mean_df,
        "RawPostDCA",
        y_col="MeanStimulusFCV",
        xlabel=DCA_POST_LABEL,
        ylabel="stimulus zFCV",
    )


def draw_all_panels(axes, plot_data):
    draw_panel_a(axes["a"], plot_data["unit_region_df"])
    draw_panel_b(axes["b"], plot_data["unit_region_df"])
    draw_panel_c(axes["c"], plot_data["fc_shift_mean_df"])
    draw_panel_d(axes["d"], plot_data["fc_shift_mean_df"])
    draw_panel_e(axes["e"], plot_data["fc_shift_mean_df"])


# 4. Panel position adjustment and panel labels

def adjust_panel_positions_and_labels(fig, axes):
    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]
    ax_e = axes["e"]

    for ax in [ax_c, ax_e]:
        bottom, _ = ax.get_ylim()
        ax.set_ylim(bottom, 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax_a.set_xlim(0.1, 0.35)
    ax_a.set_ylim(-0.25, 0.15)

    ax_b.set_xlim(0.1, 0.35)
    ax_b.set_ylim(-1.5, 2)

    ax_c.set_xlim(-1.1, 2)
    ax_c.set_ylim(-0.7, 0.7)

    ax_d.set_xlim(-1.1, 2)
    ax_d.set_ylim(-1.7, 1.7)

    ax_e.set_xlim(-0.25, 0.15)
    ax_e.set_ylim(-0.7, 0.7)

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e], "ABCDE"):
        ax.text(
            -0.28,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=fs.PANEL_LABEL_FS_2COL,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
            label=label,
        )
        for label, color in DIVISION_COLORS.items()
    ]
    ax_a.legend(handles=handles, loc="lower right", fontsize=fs.STAT_FS_2COL)


# 5. Save figure and statistics

def save_figure(fig):
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")


def write_interpretation(stats):
    count_row = stats[
        (stats["Target"] == "PostDCA")
        & (stats["Variant"] == "unit_norm")
        & (stats["Metric"] == "OO_count")
    ].iloc[0]
    frac_row = stats[
        (stats["Target"] == "PostDCA")
        & (stats["Variant"] == "unit_norm")
        & (stats["Metric"] == "OO_fraction")
    ].iloc[0]
    fcv_frac_row = stats[
        (stats["Target"] == "FCV")
        & (stats["Variant"] == "unit_norm")
        & (stats["Metric"] == "OO_fraction")
    ].iloc[0]
    partial_row = stats[
        (stats["Target"] == "PostDCA")
        & (stats["Variant"] == "unit_norm")
        & (stats["Metric"] == "OO_count_partial_degree")
    ].iloc[0]

    r = count_row["Pearson_r"]
    p = count_row["Pearson_p"]
    frac_r = frac_row["Pearson_r"]
    frac_p = frac_row["Pearson_p"]
    fcv_frac_r = fcv_frac_row["Pearson_r"]
    fcv_frac_p = fcv_frac_row["Pearson_p"]
    partial_r = partial_row["Pearson_r"]
    partial_p = partial_row["Pearson_p"]
    delta_post_fcv = frac_r - fcv_frac_r
    if np.isfinite(frac_r) and frac_r > 0.5 and frac_p < 0.05:
        interpretation = (
            "Post-DCA was strongly correlated with the regional fraction of "
            f"output-to-output inter-regional motifs (r = {frac_r:.3f}, "
            f"p = {frac_p:.3g}) and with degree-controlled OO enrichment "
            f"(r = {partial_r:.3f}, p = {partial_p:.3g}), confirming that it "
            "captures normalized cascade structure rather than raw outgoing "
            "edge count."
        )
    else:
        interpretation = (
            "The weak correlation between Post-DCA and OO motif count "
            f"(r = {r:.3f}, p = {p:.3g}) suggests that Post-DCA captures the "
            "relative directional embedding position of target populations "
            "rather than the absolute number of broadcast-to-broadcast connections."
        )

    lines = [
        "Figure 14 OO Motif Validation",
        "================================",
        "",
        "Primary unit_norm / OO_count result (region means):",
        f"Pearson r = {r:.6f}",
        f"Pearson p = {p:.6g}",
        f"N = {int(count_row['N'])}",
        "",
        "Primary unit_norm / OO_fraction result (region means):",
        f"Pearson r = {frac_r:.6f}",
        f"Pearson p = {frac_p:.6g}",
        f"N = {int(frac_row['N'])}",
        "",
        "FCV unit_norm / OO_fraction result (region means):",
        f"Pearson r = {fcv_frac_r:.6f}",
        f"Pearson p = {fcv_frac_p:.6g}",
        f"N = {int(fcv_frac_row['N'])}",
        "",
        "Post-DCA minus FCV OO_fraction correlation difference:",
        f"Delta r = {delta_post_fcv:.6f}",
        "",
        "Degree-controlled OO result (region means):",
        f"Pearson r = {partial_r:.6f}",
        f"Pearson p = {partial_p:.6g}",
        f"N = {int(partial_row['N'])}",
        "",
        "Suggested manuscript sentence:",
        interpretation,
        "",
        "Full statistics are in figure14_postdca_oo_motif_stats.csv.",
    ]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_STATS_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _cached_results_available(include_raw=False):
    if not OUT_REGION_MEAN.exists() or not OUT_SUBJECT_SUMMARY.exists():
        return False
    if not include_raw:
        return True
    try:
        subject_df = pd.read_csv(OUT_SUBJECT_SUMMARY, usecols=["Variant"])
    except ValueError:
        return False
    return "raw" in set(subject_df["Variant"].astype(str))


def _load_cached_results():
    region_df = pd.read_csv(OUT_REGION_MEAN)
    region_df = _ensure_fcv_column(region_df)
    region_df.to_csv(OUT_REGION_MEAN, index=False)
    subject_df = pd.read_csv(OUT_SUBJECT_SUMMARY)
    return subject_df, region_df


def _complete_functional_region_set():
    fc_regions, _, x_fc, _ = coupling.load_fc_matrix()
    valid = np.isfinite(np.asarray(x_fc, dtype=float)).all(axis=1)
    return set(np.asarray(fc_regions)[valid])


def _filter_complete_functional_regions(subject_df, region_df):
    keep_regions = _complete_functional_region_set()
    subject_filtered = subject_df[subject_df["Region"].isin(keep_regions)].copy()
    region_filtered = region_df[region_df["Region"].isin(keep_regions)].copy()
    return subject_filtered, region_filtered


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Post-DCA against output-to-output inter-regional motifs."
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help=(
            "Also recompute a raw-DCA robustness variant from sparse intra-region "
            "subnetworks. By default, only saved cp_out/cp_in DCA values are used."
        ),
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Deprecated alias for the default cached plotting behavior. "
            "The script now uses saved results automatically when available."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force recomputation from raw cellular SC and subject DCA files, overwriting saved cached tables.",
    )
    parser.add_argument(
        "--edge-convention",
        choices=("col0_to_col1", "col1_to_col0"),
        default="col0_to_col1",
        help=(
            "Direction convention for cellular_sc_list. The default treats "
            "sc_list[:, 0] as pre/source and sc_list[:, 1] as post/target."
        ),
    )
    parser.add_argument(
        "--dca-convention",
        choices=("prepost_swapped", "stored_labels"),
        default="prepost_swapped",
        help=(
            "How to interpret subject*_cp_cont_data.npz. prepost_swapped "
            "treats arr_0-arr_1 as output DCA because the original DCA "
            "generation used reversed pre/post labels; stored_labels uses "
            "arr_1-arr_0."
        ),
    )
    parser.add_argument(
        "--output-polarity",
        choices=("positive", "negative"),
        default="positive",
        help=(
            "Neuron DCA polarity used to define output-dominated neurons. "
            "positive uses dca > threshold; negative uses dca < threshold."
        ),
    )
    return parser.parse_args()


def prepare_plot_data(args):
    use_cache = (not args.refresh) and _cached_results_available(include_raw=args.include_raw)
    if use_cache:
        print(f"Loading cached Figure 14 results from {FIG14_DATA_DIR}")
        subject_df, region_df = _load_cached_results()
    else:
        if args.plot_only and not _cached_results_available(include_raw=args.include_raw):
            print("Cached Figure 14 results were not found; computing them once now.")
        subject_df = build_validation_table(
            include_raw=args.include_raw,
            edge_convention=args.edge_convention,
            dca_convention=args.dca_convention,
            output_polarity=args.output_polarity,
        )
        region_df = average_regions(subject_df)
    subject_df, region_df = _filter_complete_functional_regions(subject_df, region_df)
    stats = compute_stats(region_df)
    subject_stats = compute_subject_stats(subject_df)
    return {
        "subject_df": subject_df,
        "region_df": region_df,
        "unit_region_df": region_df[region_df["Variant"] == "unit_norm"].copy(),
        "stats": stats,
        "subject_stats": subject_stats,
        "fc_shift_mean_df": fc_shift_summary.load_tables()[2],
    }


def save_statistics(plot_data):
    write_interpretation(plot_data["stats"])


def main():
    args = parse_args()
    plot_data = prepare_plot_data(args)
    fig, axes = prepare_layout()
    draw_all_panels(axes, plot_data)
    adjust_panel_positions_and_labels(fig, axes)
    save_figure(fig)
    save_statistics(plot_data)
    plt.close(fig)


if __name__ == "__main__":
    main()
