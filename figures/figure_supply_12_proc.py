import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import figure_regionwise_multivariate_coupling as coupling
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
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_supply_12.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_supply_12.pdf")
OUT_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_supply_12_division_control_summary.csv")
OUT_REGION_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_supply_12_division_control_region_values.csv")

DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}
POST_DCA_LABELS = ("Post-DCA", r"$\mathrm{DCA}_{\mathrm{post}}$")


def _post_dca_index(labels):
    for candidate in POST_DCA_LABELS:
        if candidate in labels:
            return labels.index(candidate)
    raise ValueError("Post-DCA feature label not found")


def p_text(p):
    return f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"


def r2_score(y, y_pred):
    return 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)


def division_design(divisions):
    return pd.get_dummies(
        pd.Categorical(divisions, categories=DIVISION_ORDER),
        drop_first=True,
        dtype=float,
    ).to_numpy()


def residualize_against_division(values, divisions):
    X_div = division_design(divisions)
    return values - LinearRegression().fit(X_div, values).predict(X_div)


def fit_division_models(post_dca, fcv, divisions):
    X_div = division_design(divisions)
    post_z = StandardScaler().fit_transform(post_dca.reshape(-1, 1))
    X_full = np.column_stack([X_div, post_z[:, 0]])

    div_model = LinearRegression().fit(X_div, fcv)
    full_model = LinearRegression().fit(X_full, fcv)
    y_div = div_model.predict(X_div)
    y_full = full_model.predict(X_full)

    return {
        "division_r2": r2_score(fcv, y_div),
        "full_r2": r2_score(fcv, y_full),
        "delta_r2": r2_score(fcv, y_full) - r2_score(fcv, y_div),
        "post_dca_beta": full_model.coef_[-1],
        "division_pred": y_div,
        "full_pred": y_full,
    }


def stratified_permutation(post_dca, fcv, divisions, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    observed_post_resid = residualize_against_division(post_dca, divisions)
    fcv_resid = residualize_against_division(fcv, divisions)
    observed_r, observed_p = pearsonr(observed_post_resid, fcv_resid)

    null_r = np.zeros(n_perm, dtype=float)
    for perm_idx in range(n_perm):
        permuted = np.array(post_dca, copy=True)
        for division in DIVISION_ORDER:
            mask = divisions == division
            permuted[mask] = rng.permutation(permuted[mask])
        perm_resid = residualize_against_division(permuted, divisions)
        null_r[perm_idx] = pearsonr(perm_resid, fcv_resid)[0]

    perm_p = (1.0 + np.sum(np.abs(null_r) >= abs(observed_r))) / (n_perm + 1.0)
    return observed_post_resid, fcv_resid, observed_r, observed_p, null_r, perm_p


def load_analysis_data():
    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = coupling.align_matrices()
    post_dca = X_sc[:, _post_dca_index(sc_labels)]
    fcv = X_fc[:, fc_labels.index("FCV")]
    return regions, divisions, post_dca, fcv


def plot_partial_residual(ax, post_resid, fcv_resid, divisions, r_partial, p_partial):
    for division in DIVISION_ORDER:
        mask = divisions == division
        ax.scatter(
            post_resid[mask],
            fcv_resid[mask],
            s=70 if division == "Tel" else 54,
            color=DIVISION_COLORS[division],
            edgecolor="black" if division == "Tel" else "none",
            linewidth=0.4,
            alpha=0.86,
            label=division,
        )
    slope, intercept = np.polyfit(post_resid, fcv_resid, 1)
    xs = np.linspace(np.min(post_resid), np.max(post_resid), 100)
    ax.plot(xs, slope * xs + intercept, color="#222222", lw=1.8)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.set_xlabel(r"$\mathrm{DCA}_{\mathrm{post}}$ residual after division")
    ax.set_ylabel("FCV residual after division")
    ax.set_title("Division-controlled association")
    ax.text(
        0.04,
        0.96,
        f"partial r = {r_partial:.3f}\n{p_text(p_partial)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=fs.TICK_FS_2COL - 1)


def plot_stratified_null(ax, null_r, observed_r, perm_p):
    ax.hist(null_r, bins=40, color="#bdbdbd", edgecolor="white")
    ax.axvline(observed_r, color="#E45756", lw=2.0)
    ax.axvline(-observed_r, color="#E45756", lw=1.2, linestyle="--")
    ax.set_title("Division-stratified permutation")
    ax.set_xlabel("Partial r under within-division shuffle")
    ax.set_ylabel("Permutations")
    ax.text(
        0.04,
        0.96,
        f"observed r = {observed_r:.3f}\n{p_text(perm_p)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )


def plot_model_comparison(ax, model_stats):
    values = [model_stats["division_r2"], model_stats["full_r2"]]
    labels = ["Division\nonly", "Division +\n" + r"$\mathrm{DCA}_{\mathrm{post}}$"]
    ax.bar([0, 1], values, color=["#8c8c8c", "#E45756"], width=0.58)
    ax.plot([0, 1], values, color="#333333", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("In-sample R2")
    ax.set_title("Model comparison")
    y_top = max(values) * 1.28 if max(values) > 0 else 0.1
    ax.set_ylim(0, y_top)
    ax.text(
        0.50,
        max(values) * 1.08,
        f"Delta R2 = {model_stats['delta_r2']:.3f}\n"
        f"beta = {model_stats['post_dca_beta']:.3f}",
        ha="center",
        va="bottom",
    )


def add_panel_label(ax, label):
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=fs.PANEL_LABEL_FS_2COL,
        fontweight="bold",
    )


def main():
    n_perm = 10000
    seed = 0
    regions, divisions, post_dca, fcv = load_analysis_data()
    post_resid, fcv_resid, r_partial, p_partial, null_r, perm_p = stratified_permutation(
        post_dca, fcv, divisions, n_perm=n_perm, seed=seed
    )
    model_stats = fit_division_models(post_dca, fcv, divisions)

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
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame({
        "Region": regions,
        "Division": divisions,
        "PostDCA": post_dca,
        "FCV": fcv,
        "PostDCA_resid_division": post_resid,
        "FCV_resid_division": fcv_resid,
        "Division_model_pred": model_stats["division_pred"],
        "Division_PostDCA_model_pred": model_stats["full_pred"],
    }).to_csv(OUT_REGION_CSV, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    plot_partial_residual(axes[0], post_resid, fcv_resid, divisions, r_partial, p_partial)
    plot_stratified_null(axes[1], null_r, r_partial, perm_p)
    plot_model_comparison(axes[2], model_stats)
    for ax, label in zip(axes, "ABC"):
        add_panel_label(ax, label)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")
    print(f"Saved {OUT_REGION_CSV}")


if __name__ == "__main__":
    main()
