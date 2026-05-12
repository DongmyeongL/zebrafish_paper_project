import argparse
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import figure_regionwise_multivariate_coupling as coupling
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
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_sc_fc_fcv_prediction_robustness.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_sc_fc_fcv_prediction_robustness.pdf")
OUT_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_sc_fc_fcv_prediction_robustness_summary.csv")
OUT_RESIDUAL_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_sc_fc_fcv_residual_proxy_summary.csv")
OUT_MEDIATION_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_sc_fc_fcv_mediation_summary.csv")
OUT_NEURON_COUNT_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_region_actual_neuron_count_summary.csv")
EMP_CA_REGION_FILE = os.path.join(DATA, "data_saved_emp_ca_data_root_areas_region.pkl")

SC_DISPLAY_LABELS = ["Clust.", "Mod Q", "Glob Eff.", "Post-DCA", "Pre-DCA", "log10(O/I deg)"]
DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}


def r2_from_pred(y, y_pred):
    return 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def linear_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def rf_model(seed=0):
    return RandomForestRegressor(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1,
    )


def cv_prediction_stats(model, X, y, cv):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    r, p = pearsonr(y, y_pred)
    r2 = r2_from_pred(y, y_pred)
    return y_pred, r, p, r2


def permutation_null_cv_r2(model_factory, X, y, cv, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    null_r2 = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        y_perm_pred = cross_val_predict(model_factory(), X, y_perm, cv=cv)
        null_r2[i] = r2_from_pred(y_perm, y_perm_pred)
    return null_r2


def drop_one_cv_delta_r2(X, y, labels, cv, full_r2):
    rows = []
    for i, label in enumerate(labels):
        X_reduced = np.delete(X, i, axis=1)
        y_pred = cross_val_predict(linear_model(), X_reduced, y, cv=cv)
        reduced_r2 = r2_from_pred(y, y_pred)
        rows.append({
            "Feature": label,
            "Reduced_CV_R2": reduced_r2,
            "Delta_CV_R2": full_r2 - reduced_r2,
        })
    return pd.DataFrame(rows)


def residual_proxy_table(regions, residuals):
    rows = []
    residual_abs = np.abs(residuals)
    residual_sq = residuals ** 2

    bar_path = os.path.join(DATA, "fig1_prism_D_FCS_FCV_bar.csv")
    if os.path.exists(bar_path):
        bar_df = pd.read_csv(bar_path).set_index("Region")
        if "FCV_SEM" in bar_df.columns:
            values = bar_df.reindex(regions)["FCV_SEM"].astype(float).values
            rows.extend(_proxy_stats("Animal-level variability proxy", "FCV_SEM", values, residual_abs, residual_sq))

    ca_sem_df = calcium_trace_variance_sem_table(regions)
    if not ca_sem_df.empty:
        values = ca_sem_df.set_index("Region").reindex(regions)["CaTraceVarianceSEM"].astype(float).values
        rows.extend(_proxy_stats(
            "Imaging/signal variance proxy",
            "Ca_trace_variance_SEM",
            values,
            residual_abs,
            residual_sq,
            note="SEM across subjects of each region mean calcium trace temporal variance.",
        ))

    neuron_df = load_region_neuron_count_table()
    if not neuron_df.empty:
        values = neuron_df.set_index("Region").reindex(regions)["NeuronCountMean"].astype(float).values
        values = np.log10(values)
        rows.extend(_proxy_stats(
            "Neuron count proxy",
            "Log10_mean_neuron_count",
            values,
            residual_abs,
            residual_sq,
            note="Log10 mean number of neurons assigned to each anatomical region across subjects.",
        ))
    else:
        rows.append({
            "ProxyGroup": "Neuron count proxy",
            "Proxy": "Not found",
            "N": 0,
            "AbsResidual_r": np.nan,
            "AbsResidual_p": np.nan,
            "SqResidual_r": np.nan,
            "SqResidual_p": np.nan,
            "Note": "No region-matched neuron assignment table was found in the local figure data.",
        })
    return pd.DataFrame(rows)


def load_region_neuron_count_table():
    if not os.path.exists(OUT_NEURON_COUNT_CSV):
        return pd.DataFrame()
    return pd.read_csv(OUT_NEURON_COUNT_CSV)


def calcium_trace_variance_sem_table(regions):
    if not os.path.exists(EMP_CA_REGION_FILE):
        return pd.DataFrame()
    import pickle

    with open(EMP_CA_REGION_FILE, "rb") as f:
        data = pickle.load(f)

    emp_ca_data = data["emp_ca_data"]
    root_areas = data["root_areas"]
    region_names = list(fs.region)
    rows = []
    for region in regions:
        if region not in region_names:
            continue
        region_idx = region_names.index(region)
        subject_vars = []
        for subj_ca, subj_roots in zip(emp_ca_data, root_areas):
            subj_ca = np.asarray(subj_ca, dtype=float)
            subj_roots = np.asarray(subj_roots, dtype=int)
            cluster_idx = np.where(subj_roots == region_idx)[0]
            if cluster_idx.size == 0:
                continue
            region_trace = np.nanmean(subj_ca[cluster_idx], axis=0)
            subject_vars.append(np.nanvar(region_trace))
        subject_vars = np.asarray(subject_vars, dtype=float)
        subject_vars = subject_vars[np.isfinite(subject_vars)]
        if subject_vars.size == 0:
            continue
        sem = np.nan if subject_vars.size < 2 else np.nanstd(subject_vars, ddof=1) / np.sqrt(subject_vars.size)
        rows.append({
            "Region": region,
            "CaTraceVarianceMean": np.nanmean(subject_vars),
            "CaTraceVarianceSEM": sem,
            "NSubjects": int(subject_vars.size),
        })
    return pd.DataFrame(rows)


def _proxy_stats(proxy_group, proxy_name, values, residual_abs, residual_sq, note=""):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(residual_abs) & np.isfinite(residual_sq)
    if np.sum(mask) < 4 or np.nanstd(values[mask]) == 0:
        return [{
            "ProxyGroup": proxy_group,
            "Proxy": proxy_name,
            "N": int(np.sum(mask)),
            "AbsResidual_r": np.nan,
            "AbsResidual_p": np.nan,
            "SqResidual_r": np.nan,
            "SqResidual_p": np.nan,
            "Note": "Insufficient finite variation.",
        }]
    abs_r, abs_p = pearsonr(values[mask], residual_abs[mask])
    sq_r, sq_p = pearsonr(values[mask], residual_sq[mask])
    return [{
        "ProxyGroup": proxy_group,
        "Proxy": proxy_name,
        "N": int(np.sum(mask)),
        "AbsResidual_r": abs_r,
        "AbsResidual_p": abs_p,
        "SqResidual_r": sq_r,
        "SqResidual_p": sq_p,
        "Note": note,
    }]


def scatter_pred(ax, y, y_pred, divisions, title, stats_text):
    lo = min(np.min(y), np.min(y_pred))
    hi = max(np.max(y), np.max(y_pred))
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "--", color="#999999", lw=1.1)
    for div in DIVISION_ORDER:
        mask = divisions == div
        ax.scatter(
            y[mask],
            y_pred[mask],
            s=54 if div != "Tel" else 72,
            color=DIVISION_COLORS[div],
            edgecolor="black" if div == "Tel" else "none",
            linewidth=0.4,
            alpha=0.86,
            label=div,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel("Observed FCV")
    ax.set_ylabel("Predicted FCV")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes, ha="left", va="top")


def p_text(p):
    return f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"


def coefficient(X, y):
    return LinearRegression().fit(np.asarray(X), y).coef_


def bootstrap_mediation(x, mediator, y, n_boot=20000, seed=0):
    x, mediator, y = StandardScaler().fit_transform(np.column_stack([x, mediator, y])).T
    total = coefficient(x.reshape(-1, 1), y)[0]
    path_a = coefficient(x.reshape(-1, 1), mediator)[0]
    path_b, direct = coefficient(np.column_stack([mediator, x]), y)
    indirect = path_a * path_b

    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {
        "total": np.zeros(n_boot),
        "path_a": np.zeros(n_boot),
        "path_b": np.zeros(n_boot),
        "direct": np.zeros(n_boot),
        "indirect": np.zeros(n_boot),
    }
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        mb = mediator[idx]
        yb = y[idx]
        aa = coefficient(xb.reshape(-1, 1), mb)[0]
        bb, cc = coefficient(np.column_stack([mb, xb]), yb)
        tt = coefficient(xb.reshape(-1, 1), yb)[0]
        boot["path_a"][i] = aa
        boot["path_b"][i] = bb
        boot["direct"][i] = cc
        boot["total"][i] = tt
        boot["indirect"][i] = aa * bb

    rows = []
    for effect, estimate, boot_key in [
        ("Total effect c", total, "total"),
        ("Path a", path_a, "path_a"),
        ("Path b", path_b, "path_b"),
        ("Direct effect cprime", direct, "direct"),
        ("Indirect effect ab", indirect, "indirect"),
    ]:
        values = boot[boot_key]
        p_value = 2.0 * min(
            (np.sum(values <= 0) + 1) / (len(values) + 1),
            (np.sum(values >= 0) + 1) / (len(values) + 1),
        )
        ci_low, ci_high = np.percentile(values, [2.5, 97.5])
        rows.append({
            "Effect": effect,
            "Estimate": estimate,
            "Bootstrap_p": p_value,
            "CI95_low": ci_low,
            "CI95_high": ci_high,
        })

    for effect, values_a, values_b in [
        ("Pearson X-M", x, mediator),
        ("Pearson X-Y", x, y),
        ("Pearson M-Y", mediator, y),
    ]:
        r_value, p_value = pearsonr(values_a, values_b)
        rows.append({
            "Effect": effect,
            "Estimate": r_value,
            "Bootstrap_p": np.nan,
            "CI95_low": np.nan,
            "CI95_high": np.nan,
            "Pearson_p": p_value,
        })
    return pd.DataFrame(rows)


def _mediation_value(mediation_df, effect, column):
    return mediation_df.loc[mediation_df["Effect"] == effect, column].iloc[0]


def draw_node(ax, xy, text, width=0.27, height=0.15, facecolor="#f7f7f7"):
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor="#333333",
        linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center")


def draw_arrow(ax, start, end, text, text_xy, color="#333333", lw=1.4, linestyle="-", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
    )
    arrow.set_connectionstyle(f"arc3,rad={rad}")
    ax.add_patch(arrow)
    ax.text(text_xy[0], text_xy[1], text, ha="center", va="center", color=color)


def plot_mediation(ax, mediation_df):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    path_a = _mediation_value(mediation_df, "Path a", "Estimate")
    path_b = _mediation_value(mediation_df, "Path b", "Estimate")
    total = _mediation_value(mediation_df, "Total effect c", "Estimate")
    total_p = _mediation_value(mediation_df, "Total effect c", "Bootstrap_p")
    direct = _mediation_value(mediation_df, "Direct effect cprime", "Estimate")
    direct_p = _mediation_value(mediation_df, "Direct effect cprime", "Bootstrap_p")
    indirect = _mediation_value(mediation_df, "Indirect effect ab", "Estimate")
    indirect_p = _mediation_value(mediation_df, "Indirect effect ab", "Bootstrap_p")
    indirect_low = _mediation_value(mediation_df, "Indirect effect ab", "CI95_low")
    indirect_high = _mediation_value(mediation_df, "Indirect effect ab", "CI95_high")

    draw_node(ax, (0.18, 0.24), "Output bias\nlog10(O/I)", facecolor="#f1f5fb")
    draw_node(ax, (0.50, 0.72), "Post-DCA", facecolor="#fdeceb")
    draw_node(ax, (0.82, 0.24), "FCV", facecolor="#f1f5fb")

    draw_arrow(ax, (0.30, 0.31), (0.42, 0.64), f"a = {path_a:.3f}", (0.31, 0.55))
    draw_arrow(ax, (0.58, 0.64), (0.72, 0.31), f"b = {path_b:.3f}", (0.69, 0.55), color="#E45756", lw=2.0)
    draw_arrow(
        ax,
        (0.32, 0.24),
        (0.68, 0.24),
        f"c' = {direct:.3f}",
        (0.50, 0.33),
    )
    ax.text(
        0.50,
        0.95,
        f"Indirect ab = {indirect:.3f}, {p_text(indirect_p)}\n"
        f"95% CI [{indirect_low:.3f}, {indirect_high:.3f}]",
        ha="center",
        va="top",
    )
    ax.text(
        0.50,
        0.055,
        f"total c = {total:.3f}, {p_text(total_p)}; c' {p_text(direct_p)}",
        ha="center",
        va="bottom",
        fontsize=fs.TICK_FS_2COL,
    )
    ax.set_title("Output bias -> Post-DCA -> FCV")


def make_plot(results, drop_df, residual_df, rf_importance, mediation_df, n_perm):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    ax_med, ax_rf, ax_perm, ax_drop, ax_imp, ax_res = axes.ravel()

    plot_mediation(ax_med, mediation_df)
    scatter_pred(
        ax_rf,
        results["y"],
        results["rf_pred"],
        results["divisions"],
        "Random forest SC -> FCV",
        f"5-fold CV\nr = {results['rf_r']:.3f}\n{p_text(results['rf_p'])}\nR2 = {results['rf_r2']:.3f}",
    )
    ax_rf.legend(frameon=False, loc="lower right", ncol=2, fontsize=fs.TICK_FS_2COL - 1)

    null_r2 = results["null_r2"]
    ax_perm.hist(null_r2, bins=35, color="#bdbdbd", edgecolor="white")
    ax_perm.axvline(results["linear_r2"], color="#E45756", lw=2.0)
    ax_perm.set_title("Permutation null")
    ax_perm.set_xlabel("Linear CV R2")
    ax_perm.set_ylabel("Permutations")
    ax_perm.text(
        0.04,
        0.96,
        f"n = {n_perm}\n95% = {np.percentile(null_r2, 95):.3f}\n{p_text(results['perm_p'])}",
        transform=ax_perm.transAxes,
        ha="left",
        va="top",
    )

    drop_vals = drop_df["Delta_CV_R2"].values
    drop_colors = ["#E45756" if f == "Post-DCA" else "#8c8c8c" for f in SC_DISPLAY_LABELS]
    ax_drop.barh(np.arange(len(SC_DISPLAY_LABELS)), drop_vals, color=drop_colors)
    ax_drop.axvline(0, color="#bbbbbb", lw=1)
    ax_drop.set_yticks(np.arange(len(SC_DISPLAY_LABELS)))
    ax_drop.set_yticklabels(SC_DISPLAY_LABELS)
    ax_drop.invert_yaxis()
    ax_drop.set_title("Drop-one linear CV")
    ax_drop.set_xlabel("Delta R2 after removal")

    imp_vals = rf_importance["Importance"].values
    imp_err = rf_importance["Importance_SD"].values
    imp_colors = ["#E45756" if f == "Post-DCA" else "#54A24B" for f in SC_DISPLAY_LABELS]
    ax_imp.barh(np.arange(len(SC_DISPLAY_LABELS)), imp_vals, xerr=imp_err, color=imp_colors, alpha=0.92)
    ax_imp.axvline(0, color="#bbbbbb", lw=1)
    ax_imp.set_yticks(np.arange(len(SC_DISPLAY_LABELS)))
    ax_imp.set_yticklabels(SC_DISPLAY_LABELS)
    ax_imp.invert_yaxis()
    ax_imp.set_title("RF permutation importance")
    ax_imp.set_xlabel("Decrease in R2")

    plot_residual_proxy(ax_res, residual_df)

    for ax, label in zip(axes.ravel(), list("ABCDEF")):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes,
                fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold")
    fig.tight_layout(w_pad=1.4, h_pad=1.5)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def plot_residual_proxy(ax, residual_df):
    plot_df = residual_df[np.isfinite(residual_df["AbsResidual_r"])].copy()
    if plot_df.empty:
        ax.axis("off")
        ax.text(0.05, 0.95, "No residual proxy data available", transform=ax.transAxes, ha="left", va="top")
        return
    labels = plot_df["Proxy"].tolist()
    values = plot_df["AbsResidual_r"].values
    colors = ["#4C78A8", "#F2A541", "#7A5195"][:len(values)]
    label_map = {
        "FCV_SEM": "FCV SEM",
        "Ca_trace_variance_SEM": "Ca var. SEM",
        "Log10_mean_neuron_count": "log10 neuron count",
    }
    ax.barh(np.arange(len(labels)), values, color=colors, alpha=0.92)
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([label_map.get(label, label) for label in labels])
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_title("Residual proxy check")
    ax.set_xlabel("r with |linear residual|")
    for y_pos, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(values[y_pos] + (0.04 if values[y_pos] >= 0 else -0.04), y_pos,
                p_text(row["AbsResidual_p"]).replace("p = ", "p="),
                va="center", ha="left" if values[y_pos] >= 0 else "right",
                fontsize=fs.TICK_FS_2COL - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = coupling.align_matrices()
    y = X_fc[:, fc_labels.index("FCV")]
    cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    linear_pred, linear_r, linear_p, linear_r2 = cv_prediction_stats(linear_model(), X_sc, y, cv)
    rf_pred, rf_r, rf_p, rf_r2 = cv_prediction_stats(rf_model(args.seed), X_sc, y, cv)
    null_r2 = permutation_null_cv_r2(linear_model, X_sc, y, cv, n_perm=args.n_perm, seed=args.seed)
    perm_p = (1.0 + np.sum(null_r2 >= linear_r2)) / (args.n_perm + 1.0)

    drop_df = drop_one_cv_delta_r2(X_sc, y, sc_labels, cv, linear_r2)
    drop_df["DisplayFeature"] = SC_DISPLAY_LABELS

    rf_full = rf_model(args.seed).fit(X_sc, y)
    rf_perm = permutation_importance(
        rf_full,
        X_sc,
        y,
        scoring="r2",
        n_repeats=200,
        random_state=args.seed,
        n_jobs=1,
    )
    rf_importance = pd.DataFrame({
        "Feature": sc_labels,
        "DisplayFeature": SC_DISPLAY_LABELS,
        "Importance": rf_perm.importances_mean,
        "Importance_SD": rf_perm.importances_std,
    })

    mediation_df = bootstrap_mediation(
        X_sc[:, sc_labels.index("log10(Out/In-degree)")],
        X_sc[:, sc_labels.index("Post-DCA")],
        y,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    residual_df = residual_proxy_table(regions, y - linear_pred)
    neuron_count_df = load_region_neuron_count_table()
    summary_df = pd.DataFrame([
        {"Analysis": "Linear 5-fold CV", "Metric": "r", "Value": linear_r},
        {"Analysis": "Linear 5-fold CV", "Metric": "p", "Value": linear_p},
        {"Analysis": "Linear 5-fold CV", "Metric": "R2", "Value": linear_r2},
        {"Analysis": "Random forest 5-fold CV", "Metric": "r", "Value": rf_r},
        {"Analysis": "Random forest 5-fold CV", "Metric": "p", "Value": rf_p},
        {"Analysis": "Random forest 5-fold CV", "Metric": "R2", "Value": rf_r2},
        {"Analysis": "Linear permutation null", "Metric": "n_perm", "Value": args.n_perm},
        {"Analysis": "Linear permutation null", "Metric": "null_mean_R2", "Value": np.mean(null_r2)},
        {"Analysis": "Linear permutation null", "Metric": "null_sd_R2", "Value": np.std(null_r2, ddof=1)},
        {"Analysis": "Linear permutation null", "Metric": "null_95pct_R2", "Value": np.percentile(null_r2, 95)},
        {"Analysis": "Linear permutation null", "Metric": "p_perm", "Value": perm_p},
    ])
    summary_df = pd.concat([
        summary_df,
        drop_df.assign(Analysis="Drop-one linear CV").rename(columns={"Feature": "Metric", "Delta_CV_R2": "Value"})[
            ["Analysis", "Metric", "Value", "Reduced_CV_R2"]
        ],
        rf_importance.assign(Analysis="RF permutation importance").rename(columns={"Feature": "Metric", "Importance": "Value"})[
            ["Analysis", "Metric", "Value", "Importance_SD"]
        ],
    ], ignore_index=True)

    results = {
        "regions": regions,
        "divisions": divisions,
        "y": y,
        "linear_pred": linear_pred,
        "linear_r": linear_r,
        "linear_p": linear_p,
        "linear_r2": linear_r2,
        "rf_pred": rf_pred,
        "rf_r": rf_r,
        "rf_p": rf_p,
        "rf_r2": rf_r2,
        "null_r2": null_r2,
        "perm_p": perm_p,
    }

    make_plot(results, drop_df, residual_df, rf_importance, mediation_df, args.n_perm)
    summary_df.to_csv(OUT_CSV, index=False)
    residual_df.to_csv(OUT_RESIDUAL_CSV, index=False)
    mediation_df.to_csv(OUT_MEDIATION_CSV, index=False)
    if not neuron_count_df.empty:
        neuron_count_df.to_csv(OUT_NEURON_COUNT_CSV, index=False)

    print(f"Linear CV R2 = {linear_r2:.4f}, p_perm = {perm_p:.4g}")
    print(f"Random forest CV R2 = {rf_r2:.4f}")
    print(f"Post-DCA drop-one delta R2 = {drop_df.loc[drop_df['Feature'] == 'Post-DCA', 'Delta_CV_R2'].iloc[0]:.4f}")
    print(mediation_df.to_string(index=False))
    print(residual_df.to_string(index=False))
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_RESIDUAL_CSV}")
    print(f"Saved {OUT_MEDIATION_CSV}")
    if not neuron_count_df.empty:
        print(f"Saved {OUT_NEURON_COUNT_CSV}")


if __name__ == "__main__":
    main()
