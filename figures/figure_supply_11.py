import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import KFold

import figure_regionwise_multivariate_coupling as coupling
import figure_sc_fc_fcv_prediction_robustness as robust
import figure_sc_fc_subject_replication as repl
import figure_supply_12_proc as divctrl
import figure_style as fs


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
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_supply_11.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_supply_11.pdf")
OUT_DIVISION_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_supply_11_division_control_summary.csv")
OUT_DIVISION_REGION_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_supply_11_division_control_region_values.csv")
SC_DISPLAY_LABELS = robust.SC_DISPLAY_LABELS


def permutation_cache_path(n_perm, seed):
    return os.path.join(DATA, f"figure_supply_11_perm_null_seed{seed}_n{n_perm}.npz")


def load_or_compute_null_r2(X_sc, y, cv, n_perm, seed, refresh=False):
    cache_path = permutation_cache_path(n_perm, seed)
    if os.path.exists(cache_path) and not refresh:
        cache = np.load(cache_path)
        return cache["null_r2"]

    null_r2 = robust.permutation_null_cv_r2(
        robust.linear_model, X_sc, y, cv, n_perm=n_perm, seed=seed
    )
    np.savez(cache_path, null_r2=null_r2, n_perm=n_perm, seed=seed)
    return null_r2


def build_robustness_results(n_perm, seed, refresh_null=False):
    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = coupling.align_matrices()
    y = X_fc[:, fc_labels.index("FCV")]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    linear_pred, linear_r, linear_p, linear_r2 = robust.cv_prediction_stats(
        robust.linear_model(), X_sc, y, cv
    )
    null_r2 = load_or_compute_null_r2(
        X_sc, y, cv, n_perm=n_perm, seed=seed, refresh=refresh_null
    )
    perm_p = (1.0 + np.sum(null_r2 >= linear_r2)) / (n_perm + 1.0)

    drop_df = robust.drop_one_cv_delta_r2(X_sc, y, sc_labels, cv, linear_r2)
    residual_df = robust.residual_proxy_table(regions, y - linear_pred)

    results = {
        "regions": regions,
        "divisions": divisions,
        "y": y,
        "linear_r2": linear_r2,
        "null_r2": null_r2,
        "perm_p": perm_p,
    }
    return results, drop_df, residual_df


def build_replication_results():
    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = coupling.align_matrices()
    post_dca = X_sc[:, sc_labels.index("Post-DCA")]
    subject_fcv = repl.load_subject_fcv_matrix(regions)
    corr_df = repl.subject_correlation_table(regions, divisions, post_dca, subject_fcv)
    loao_df, pred_matrix = repl.leave_one_animal_out(X_sc, subject_fcv)
    return corr_df, loao_df, subject_fcv, pred_matrix, divisions


def build_division_control_results(n_perm, seed):
    regions, divisions, post_dca, fcv = divctrl.load_analysis_data()
    post_resid, fcv_resid, r_partial, p_partial, null_r, perm_p = divctrl.stratified_permutation(
        post_dca, fcv, divisions, n_perm=n_perm, seed=seed
    )
    model_stats = divctrl.fit_division_models(post_dca, fcv, divisions)
    summary_rows = [
        {"Analysis": "Division-controlled partial correlation", "Metric": "partial_r", "Value": r_partial},
        {"Analysis": "Division-controlled partial correlation", "Metric": "parametric_p", "Value": p_partial},
        {"Analysis": "Division-stratified permutation", "Metric": "n_perm", "Value": n_perm},
        {"Analysis": "Division-stratified permutation", "Metric": "p_perm", "Value": perm_p},
        {"Analysis": "Division-only model", "Metric": "R2", "Value": model_stats["division_r2"]},
        {"Analysis": "Division-plus-Post-DCA model", "Metric": "R2", "Value": model_stats["full_r2"]},
        {"Analysis": "Division-plus-Post-DCA model", "Metric": "Delta_R2", "Value": model_stats["delta_r2"]},
        {"Analysis": "Division-plus-Post-DCA model", "Metric": "PostDCA_beta", "Value": model_stats["post_dca_beta"]},
    ]
    pd.DataFrame(summary_rows).to_csv(OUT_DIVISION_CSV, index=False)
    pd.DataFrame({
        "Region": regions,
        "Division": divisions,
        "PostDCA": post_dca,
        "FCV": fcv,
        "PostDCA_resid_division": post_resid,
        "FCV_resid_division": fcv_resid,
        "Division_model_pred": model_stats["division_pred"],
        "Division_PostDCA_model_pred": model_stats["full_pred"],
    }).to_csv(OUT_DIVISION_REGION_CSV, index=False)
    return {
        "regions": regions,
        "divisions": divisions,
        "post_resid": post_resid,
        "fcv_resid": fcv_resid,
        "r_partial": r_partial,
        "p_partial": p_partial,
        "null_r": null_r,
        "perm_p": perm_p,
        "model_stats": model_stats,
    }


def plot_permutation_null(ax, results, n_perm):
    null_r2 = results["null_r2"]
    ax.hist(null_r2, bins=35, color="#bdbdbd", edgecolor="white")
    ax.axvline(results["linear_r2"], color="#E45756", lw=2.0)
    ax.set_title("Permutation null")
    ax.set_xlabel("Linear CV R2")
    ax.set_ylabel("Permutations")
    ax.text(
        0.04,
        0.96,
        f"n = {n_perm}\n95% = {np.percentile(null_r2, 95):.3f}\n"
        f"{robust.p_text(results['perm_p'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def plot_drop_one(ax, drop_df):
    drop_vals = drop_df["Delta_CV_R2"].values
    drop_colors = ["#E45756" if f == "Post-DCA" else "#8c8c8c" for f in SC_DISPLAY_LABELS]
    ax.barh(np.arange(len(SC_DISPLAY_LABELS)), drop_vals, color=drop_colors)
    ax.axvline(0, color="#bbbbbb", lw=1)
    ax.set_yticks(np.arange(len(SC_DISPLAY_LABELS)))
    ax.set_yticklabels(SC_DISPLAY_LABELS)
    ax.invert_yaxis()
    ax.set_title("Drop-one linear CV")
    ax.set_xlabel("Delta R2 after removal")


def add_source_panel_label(ax, label):
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
    )


def shrink_axes(ax, width_scale=0.80, height_scale=0.80):
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    cy = pos.y0 + pos.height / 2
    new_w = pos.width * width_scale
    new_h = pos.height * height_scale
    ax.set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh-null", action="store_true")
    args = parser.parse_args()

    results, drop_df, residual_df = build_robustness_results(
        args.n_perm, args.seed, refresh_null=args.refresh_null
    )
    corr_df, loao_df, subject_fcv, pred_matrix, divisions = build_replication_results()
    division_results = build_division_control_results(args.n_perm, args.seed)

    fig = plt.figure(figsize=(17.0, 8.4))
    gs = GridSpec(
        2,
        16,
        figure=fig,
        left=0.055,
        right=0.98,
        top=0.92,
        bottom=0.12,
        hspace=0.48,
        wspace=0.95,
    )

    ax_perm = fig.add_subplot(gs[0, 0:3])
    ax_drop = fig.add_subplot(gs[0, 3:6])
    ax_resid = fig.add_subplot(gs[0, 6:10])
    ax_rep_corr = fig.add_subplot(gs[0, 10:16])
    ax_rep_pred = fig.add_subplot(gs[1, 0:5])
    ax_partial = fig.add_subplot(gs[1, 5:9])
    ax_strat = fig.add_subplot(gs[1, 9:13])
    ax_model = fig.add_subplot(gs[1, 13:16])

    plot_permutation_null(ax_perm, results, args.n_perm)
    plot_drop_one(ax_drop, drop_df)
    robust.plot_residual_proxy(ax_resid, residual_df)
    repl.plot_forest(ax_rep_corr, corr_df)
    repl.plot_loao_scatter(ax_rep_pred, subject_fcv, pred_matrix, loao_df, divisions)
    divctrl.plot_partial_residual(
        ax_partial,
        division_results["post_resid"],
        division_results["fcv_resid"],
        division_results["divisions"],
        division_results["r_partial"],
        division_results["p_partial"],
    )
    divctrl.plot_stratified_null(
        ax_strat,
        division_results["null_r"],
        division_results["r_partial"],
        division_results["perm_p"],
    )
    divctrl.plot_model_comparison(ax_model, division_results["model_stats"])

    for ax in [
        ax_perm,
        ax_drop,
        ax_resid,
        ax_rep_corr,
        ax_rep_pred,
        ax_partial,
        ax_strat,
        ax_model,
    ]:
        shrink_axes(ax, width_scale=0.80, height_scale=0.80)

    for ax, label in [
        (ax_perm, "A"),
        (ax_drop, "B"),
        (ax_resid, "C"),
        (ax_rep_corr, "D"),
        (ax_rep_pred, "E"),
        (ax_partial, "F"),
        (ax_strat, "G"),
        (ax_model, "H"),
    ]:
        add_source_panel_label(ax, label)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_DIVISION_CSV}")
    print(f"Saved {OUT_DIVISION_REGION_CSV}")


if __name__ == "__main__":
    main()
