import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

import figure_regionwise_multivariate_coupling as coupling
import figure12_clean as figure12
import figure_style as fs

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_sc_fc_final_overview.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_sc_fc_final_overview.pdf")
STATS_DIR = os.path.join(PROJECT_ROOT, "output", "stats")
STATS_SUMMARY = os.path.join(STATS_DIR, "figure_sc_fc_final_overview_stats.csv")
STATS_CORR = os.path.join(STATS_DIR, "figure_sc_fc_final_overview_pairwise_correlations.csv")
STATS_WEIGHTS = os.path.join(STATS_DIR, "figure_sc_fc_final_overview_weights.csv")
STATS_PREDICTIONS = os.path.join(STATS_DIR, "figure_sc_fc_final_overview_fcv_predictions.csv")

DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}

SC_DISPLAY_LABELS = ["Clust.", "Mod Q", "Glob Eff.", "Post-DCA", "Pre-DCA", "log10(O/I deg)"]
FC_DISPLAY_LABELS = ["FCS", "FCV", "Metastab.", "Net TE", "Neigh. TE"]
ANNOT_FS = 6.5
SMALL_FS = 8
AXIS_FS = 9
PANEL_FS = 12
OTHER_SMALL_FS = 9
OTHER_AXIS_FS = 10
OTHER_PANEL_FS = 13
TEL_LABELS = {"lOB", "rP", "rPO", "rSP"}


def load_data():
    fc_regions, fc_divisions, X_fc, fc_labels = coupling.load_fc_matrix()
    sc_matrix, sc_labels, sc_regions, _ = figure12._load_panel_a_feature_matrix()
    X_sc_all = sc_matrix.T
    sc_labels = [label.replace("\n", " ") for label in sc_labels]

    sc_lookup = {region: idx for idx, region in enumerate(sc_regions)}
    keep = [idx for idx, region in enumerate(fc_regions) if region in sc_lookup]
    regions = np.asarray(fc_regions)[keep]
    divisions = np.asarray(fc_divisions)[keep]
    X_fc = np.asarray(X_fc, dtype=float)[keep]
    X_sc = np.vstack([X_sc_all[sc_lookup[region]] for region in regions])

    valid = np.isfinite(X_fc).all(axis=1) & np.isfinite(X_sc).all(axis=1)
    regions = regions[valid]
    divisions = divisions[valid]
    X_fc = X_fc[valid]
    X_sc = X_sc[valid]

    return {
        "regions": np.asarray(regions),
        "divisions": np.asarray(divisions),
        "X_fc": np.asarray(X_fc, dtype=float),
        "X_sc": np.asarray(X_sc, dtype=float),
        "fc_labels": list(fc_labels),
        "sc_labels": list(sc_labels),
    }


def compute_corr(X_fc, X_sc):
    corr = np.zeros((X_fc.shape[1], X_sc.shape[1]), dtype=float)
    pvals = np.zeros((X_fc.shape[1], X_sc.shape[1]), dtype=float)
    for i in range(X_fc.shape[1]):
        for j in range(X_sc.shape[1]):
            corr[i, j], pvals[i, j] = pearsonr(X_fc[:, i], X_sc[:, j])
    reject_flat, p_fdr_flat, _, _ = multipletests(
        pvals.ravel(), alpha=0.05, method="fdr_bh"
    )
    return (
        corr,
        pvals,
        p_fdr_flat.reshape(pvals.shape),
        reject_flat.reshape(pvals.shape),
    )


def fit_pls(X_fc, X_sc):
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    Xf = scaler_fc.fit_transform(X_fc)
    Xs = scaler_sc.fit_transform(X_sc)

    pls = PLSCanonical(n_components=2, scale=False)
    pls.fit(Xf, Xs)
    fc_scores, sc_scores = pls.transform(Xf, Xs)
    score_r = pearsonr(fc_scores[:, 0], sc_scores[:, 0])[0]

    # Flip to keep the shared axis positively aligned.
    if score_r < 0:
        fc_scores[:, 0] *= -1
        sc_scores[:, 0] *= -1
        pls.x_weights_[:, 0] *= -1
        pls.y_weights_[:, 0] *= -1
        score_r = -score_r

    score_p = pearsonr(fc_scores[:, 0], sc_scores[:, 0])[1]
    return pls, fc_scores[:, 0], sc_scores[:, 0], score_r, score_p


def leave_one_region_out_pls_weights(X_fc, X_sc, full_pls):
    full_ref = np.r_[full_pls.x_weights_[:, 0], full_pls.y_weights_[:, 0]]
    fc_weights = []
    sc_weights = []
    for omit_idx in range(X_fc.shape[0]):
        keep = np.ones(X_fc.shape[0], dtype=bool)
        keep[omit_idx] = False
        loo_pls, _, _, _, _ = fit_pls(X_fc[keep], X_sc[keep])
        loo_ref = np.r_[loo_pls.x_weights_[:, 0], loo_pls.y_weights_[:, 0]]
        if np.dot(loo_ref, full_ref) < 0:
            loo_pls.x_weights_[:, 0] *= -1
            loo_pls.y_weights_[:, 0] *= -1
        fc_weights.append(loo_pls.x_weights_[:, 0].copy())
        sc_weights.append(loo_pls.y_weights_[:, 0].copy())
    return np.asarray(fc_weights), np.asarray(sc_weights)


def fit_linear_fcv(X_sc, X_fc, fc_labels):
    y = X_fc[:, fc_labels.index("FCV")]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X_sc, y, cv=cv)
    cv_r, cv_p = pearsonr(y, y_pred)
    cv_r2 = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    Xz = StandardScaler().fit_transform(X_sc)
    yz = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()
    beta_model = LinearRegression().fit(Xz, yz)
    betas = beta_model.coef_.copy()
    return y, y_pred, cv_r, cv_p, cv_r2, betas


def leave_one_region_out_fcv_betas(X_sc, X_fc, fc_labels):
    beta_rows = []
    for omit_idx in range(X_sc.shape[0]):
        keep = np.ones(X_sc.shape[0], dtype=bool)
        keep[omit_idx] = False
        _, _, _, _, _, betas = fit_linear_fcv(X_sc[keep], X_fc[keep], fc_labels)
        beta_rows.append(betas)
    return np.asarray(beta_rows, dtype=float)


def scatter_by_division(ax, x, y, divisions, regions):
    tel_mask = divisions == "Tel"
    for div in DIVISION_ORDER:
        mask = divisions == div
        ax.scatter(
            x[mask],
            y[mask],
            s=64 if div == "Tel" else 42,
            color=DIVISION_COLORS[div],
            alpha=0.90 if div == "Tel" else 0.78,
            edgecolor="black" if div == "Tel" else "none",
            linewidth=0.4,
            label=div,
        )
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min() - 0.15, x.max() + 0.15, 100)
    ax.plot(xs, m * xs + b, color="#4d4d4d", lw=1.8)
    ax.axhline(0, color="#dddddd", lw=1)
    ax.axvline(0, color="#dddddd", lw=1)
    ax.margins(x=0.12, y=0.14)


def add_identity_line(ax, x, y):
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#999999")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def scatter_predicted_by_division(ax, x, y, divisions, regions):
    tel_mask = divisions == "Tel"
    add_identity_line(ax, x, y)
    for div in DIVISION_ORDER:
        mask = divisions == div
        ax.scatter(
            x[mask],
            y[mask],
            s=64 if div == "Tel" else 42,
            color=DIVISION_COLORS[div],
            alpha=0.90 if div == "Tel" else 0.78,
            edgecolor="black" if div == "Tel" else "none",
            linewidth=0.4,
            label=div,
        )

def draw_horizontal_bars(
    ax,
    labels,
    values,
    base_color,
    highlight_label=None,
    highlight_color="#E45756",
    xlabel="",
    title="",
    point_values=None,
):
    colors = [highlight_color if label == highlight_label else base_color for label in labels]
    y = np.arange(len(labels))
    if point_values is not None:
        point_values = np.asarray(point_values, dtype=float)
        point_means = np.nanmean(point_values, axis=0)
        ax.barh(
            y,
            point_means,
            color=colors,
            alpha=0.28,
            height=0.46,
            edgecolor="none",
            zorder=1,
        )
        for yi in y:
            vals = point_values[:, yi]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            mean = float(np.mean(vals))
            ci95 = (
                1.96 * float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                if vals.size > 1
                else 0.0
            )
            jitter = np.linspace(-0.19, 0.19, vals.size) if vals.size > 1 else np.zeros(vals.size)
            ax.scatter(
                vals,
                np.full(vals.size, yi, dtype=float) + jitter,
                s=9,
                facecolor="#111111",
                edgecolor="white",
                linewidths=0.25,
                alpha=0.45,
                zorder=2,
            )
            ax.errorbar(
                mean,
                yi,
                xerr=ci95,
                fmt="D",
                ms=3.8,
                color=colors[yi],
                ecolor="black",
                elinewidth=0.9,
                capsize=2.3,
                capthick=0.9,
                markeredgecolor="black",
                markeredgewidth=0.4,
                zorder=4,
            )
    else:
        ax.barh(y, values, color=colors, alpha=0.92, height=0.72)
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.margins(x=0.12)


def main():
    data = load_data()
    corr, corr_p, corr_p_fdr, corr_fdr_sig = compute_corr(data["X_fc"], data["X_sc"])
    pls, fc_lv1, sc_lv1, pls_r, pls_p = fit_pls(data["X_fc"], data["X_sc"])
    loo_fc_weights, loo_sc_weights = leave_one_region_out_pls_weights(
        data["X_fc"], data["X_sc"], pls
    )
    y_obs, y_pred, cv_r, cv_p, cv_r2, betas = fit_linear_fcv(data["X_sc"], data["X_fc"], data["fc_labels"])
    loo_betas = leave_one_region_out_fcv_betas(
        data["X_sc"], data["X_fc"], data["fc_labels"]
    )

    fig = plt.figure(figsize=(16.0, 3.9))
    gs = GridSpec(
        1, 6, figure=fig,
        width_ratios=[1.50, 1.18, 0.80, 0.95, 1.18, 0.95],
        left=0.052, right=0.992, top=0.835, bottom=0.315,
        hspace=0.0, wspace=0.56,
    )

    ax_heat = fig.add_subplot(gs[0, 0])
    ax_pls = fig.add_subplot(gs[0, 1])
    ax_fcw = fig.add_subplot(gs[0, 2])
    ax_scw = fig.add_subplot(gs[0, 3])
    ax_pred = fig.add_subplot(gs[0, 4])
    ax_beta = fig.add_subplot(gs[0, 5])

    heat = sns.heatmap(
        corr,
        ax=ax_heat,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": ANNOT_FS},
        cmap="RdYlBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=1.1,
        linecolor="white",
        cbar_kws={
            #"label": "Pearson r",
            "shrink": 0.90,
            "pad": 0.025,
            "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0],
        },
        xticklabels=SC_DISPLAY_LABELS,
        yticklabels=FC_DISPLAY_LABELS,
    )
    for text, value, is_sig in zip(ax_heat.texts, corr.flatten(), corr_fdr_sig.flatten()):
        text.set_fontweight("bold" if is_sig else "normal")
        text.set_color("#222222")
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if corr_fdr_sig[i, j]:
                ax_heat.add_patch(Rectangle((j, i), 1, 1, fill=False, edgecolor="black", linewidth=1.4))
    ax_heat.set_title("")
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", rotation=32, labelsize=SMALL_FS, pad=2)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=SMALL_FS, pad=2)
    for label in ax_heat.get_xticklabels():
        label.set_horizontalalignment("right")
    heat.collections[0].colorbar.ax.tick_params(labelsize=SMALL_FS)
    heat.collections[0].colorbar.ax.yaxis.label.set_size(SMALL_FS)

    scatter_by_division(ax_pls, fc_lv1, sc_lv1, data["divisions"], data["regions"])
    ax_pls.set_title("")
    ax_pls.set_xlabel("FC score (LV1)", labelpad=2)
    ax_pls.set_ylabel("SC score (LV1)", labelpad=2)
    p_text = f"{pls_p:.3g}" if pls_p >= 0.001 else "< 0.001"
    ax_pls.text(0.04, 0.96, f"r = {pls_r:.3f}\np {p_text}", transform=ax_pls.transAxes, ha="left", va="top", fontsize=OTHER_SMALL_FS)
    ax_pls.legend(frameon=False, loc="lower right", ncol=2, fontsize=OTHER_SMALL_FS, handletextpad=0.3, columnspacing=0.75)
    ax_pls.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_pls.yaxis.set_major_locator(MaxNLocator(nbins=4))

    draw_horizontal_bars(
        ax_fcw,
        FC_DISPLAY_LABELS,
        pls.x_weights_[:, 0],
        base_color="#4C78A8",
        highlight_label="FCV",
        highlight_color="#E45756",
        xlabel="FC LV1 weight",
        title="",
        point_values=loo_fc_weights,
    )
    ax_fcw.tick_params(axis="y", labelsize=OTHER_SMALL_FS)
    ax_fcw.tick_params(axis="x", labelsize=OTHER_SMALL_FS)
    ax_fcw.xaxis.labelpad = 2

    draw_horizontal_bars(
        ax_scw,
        SC_DISPLAY_LABELS,
        pls.y_weights_[:, 0],
        base_color="#4C78A8",
        highlight_label="Post-DCA",
        highlight_color="#E45756",
        xlabel="SC LV1 weight",
        title="",
        point_values=loo_sc_weights,
    )
    ax_scw.tick_params(axis="y", labelsize=OTHER_SMALL_FS)
    ax_scw.tick_params(axis="x", labelsize=OTHER_SMALL_FS)
    ax_scw.xaxis.labelpad = 2

    scatter_predicted_by_division(ax_pred, y_obs, y_pred, data["divisions"], data["regions"])
    ax_pred.set_title("")
    ax_pred.set_xlabel("Observed FCV", labelpad=2)
    ax_pred.set_ylabel("Predicted FCV", labelpad=2)
    cv_p_text = f"{cv_p:.3g}" if cv_p >= 0.001 else "< 0.001"
    ax_pred.text(
        0.03, 0.97,
        f"5-fold CV\nr = {cv_r:.3f}\np {cv_p_text}\nR² = {cv_r2:.3f}",
        transform=ax_pred.transAxes,
        ha="left",
        va="top",
        fontsize=OTHER_SMALL_FS,
    )
    ax_pred.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.08, -0.02),
        ncol=2,
        fontsize=OTHER_SMALL_FS,
        handletextpad=0.3,
        columnspacing=0.75,
    )
    ax_pred.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_pred.yaxis.set_major_locator(MaxNLocator(nbins=4))

    draw_horizontal_bars(
        ax_beta,
        SC_DISPLAY_LABELS,
        betas,
        base_color="#54A24B",
        highlight_label="Post-DCA",
        highlight_color="#E45756",
        xlabel="Standardized beta",
        title="",
        point_values=loo_betas,
    )
    ax_beta.tick_params(axis="y", labelsize=OTHER_SMALL_FS)
    ax_beta.tick_params(axis="x", labelsize=OTHER_SMALL_FS)
    ax_beta.xaxis.labelpad = 2

    panel_axes = [ax_heat, ax_pls, ax_fcw, ax_scw, ax_pred, ax_beta]
    panel_labels = list("ABCDEF")
    ax_heat.tick_params(labelsize=SMALL_FS)
    ax_heat.xaxis.label.set_size(AXIS_FS)
    ax_heat.yaxis.label.set_size(AXIS_FS)
    for ax in panel_axes[1:]:
        ax.tick_params(labelsize=OTHER_SMALL_FS)
        ax.xaxis.label.set_size(OTHER_AXIS_FS)
        ax.yaxis.label.set_size(OTHER_AXIS_FS)

    fig.canvas.draw()
    for ax, label in zip(panel_axes, panel_labels):
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.030,
            pos.y1 + 0.010,
            label,
            fontsize=PANEL_FS if label == "A" else OTHER_PANEL_FS,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        
        
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03,transparent=True)
    fig.savefig(OUT_PDF, dpi=600, bbox_inches="tight", pad_inches=0.03)
    os.makedirs(STATS_DIR, exist_ok=True)
    pd.DataFrame([
        {"figure": "figure_sc_fc_final_overview", "panel": "B", "test": "Pearson correlation", "metric": "PLS LV1 FC score vs SC score", "r": pls_r, "p_value": pls_p, "n_regions": len(data["regions"])},
        {"figure": "figure_sc_fc_final_overview", "panel": "E", "test": "5-fold cross-validated linear regression", "metric": "Observed FCV vs predicted FCV", "r": cv_r, "p_value": cv_p, "r2_cv": cv_r2, "n_regions": len(data["regions"])},
    ]).to_csv(STATS_SUMMARY, index=False)
    corr_rows = []
    for i, fc_label in enumerate(data["fc_labels"]):
        for j, sc_label in enumerate(data["sc_labels"]):
            corr_rows.append({
                "figure": "figure_sc_fc_final_overview",
                "panel": "A",
                "fc_feature": fc_label,
                "sc_feature": sc_label,
                "pearson_r": corr[i, j],
                "p_value": corr_p[i, j],
                "p_fdr_bh": corr_p_fdr[i, j],
                "fdr_bh_significant": bool(corr_fdr_sig[i, j]),
                "n_regions": len(data["regions"]),
            })
    pd.DataFrame(corr_rows).to_csv(STATS_CORR, index=False)
    weight_rows = []
    for label, weight in zip(data["fc_labels"], pls.x_weights_[:, 0]):
        weight_rows.append({"figure": "figure_sc_fc_final_overview", "panel": "C", "model": "PLS LV1", "feature_set": "FC", "feature": label, "weight": weight})
    for label, weight in zip(data["sc_labels"], pls.y_weights_[:, 0]):
        weight_rows.append({"figure": "figure_sc_fc_final_overview", "panel": "D", "model": "PLS LV1", "feature_set": "SC", "feature": label, "weight": weight})
    for region, division, weights in zip(data["regions"], data["divisions"], loo_fc_weights):
        for label, weight in zip(data["fc_labels"], weights):
            weight_rows.append({
                "figure": "figure_sc_fc_final_overview",
                "panel": "C",
                "model": "leave-one-region-out PLS LV1",
                "feature_set": "FC",
                "left_out_region": region,
                "left_out_division": division,
                "feature": label,
                "weight": weight,
            })
    for region, division, weights in zip(data["regions"], data["divisions"], loo_sc_weights):
        for label, weight in zip(data["sc_labels"], weights):
            weight_rows.append({
                "figure": "figure_sc_fc_final_overview",
                "panel": "D",
                "model": "leave-one-region-out PLS LV1",
                "feature_set": "SC",
                "left_out_region": region,
                "left_out_division": division,
                "feature": label,
                "weight": weight,
            })
    for label, beta in zip(data["sc_labels"], betas):
        weight_rows.append({"figure": "figure_sc_fc_final_overview", "panel": "F", "model": "linear FCV prediction", "feature_set": "SC", "feature": label, "standardized_beta": beta})
    for region, division, beta_row in zip(data["regions"], data["divisions"], loo_betas):
        for label, beta in zip(data["sc_labels"], beta_row):
            weight_rows.append({
                "figure": "figure_sc_fc_final_overview",
                "panel": "F",
                "model": "leave-one-region-out linear FCV prediction",
                "feature_set": "SC",
                "left_out_region": region,
                "left_out_division": division,
                "feature": label,
                "standardized_beta": beta,
            })
    pd.DataFrame(weight_rows).to_csv(STATS_WEIGHTS, index=False)
    pd.DataFrame({
        "figure": "figure_sc_fc_final_overview",
        "panel": "E",
        "region": data["regions"],
        "division": data["divisions"],
        "observed_fcv": y_obs,
        "predicted_fcv_5fold": y_pred,
        "residual": y_obs - y_pred,
    }).to_csv(STATS_PREDICTIONS, index=False)
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {STATS_SUMMARY}")


if __name__ == "__main__":
    main()
