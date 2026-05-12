import os
import pickle
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LinearRegression
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
FC_SUBJECT_FILE = os.path.join(DATA, "figure1_box_scatter_fc_mean_var_data.pkl")
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_sc_fc_subject_replication.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_sc_fc_subject_replication.pdf")
OUT_CORR_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_sc_fc_subject_postdca_fcv_correlations.csv")
OUT_LOAO_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_sc_fc_subject_leave_one_animal_out.csv")

SUBJECT_IDS = list(range(12, 19))
SC_DISPLAY_LABELS = ["Clust.", "Mod Q", "Glob Eff.", "Post-DCA", "Pre-DCA", "log10(O/I deg)"]
DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}


def p_text(p):
    return f"p = {p:.3g}" if p >= 0.001 else "p < 0.001"


def linear_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def fisher_ci(r_value, n):
    if n <= 3 or abs(r_value) >= 1:
        return np.nan, np.nan
    z = np.arctanh(r_value)
    se = 1.0 / np.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def random_effects_fisher(rs, ns):
    z = np.arctanh(np.clip(rs, -0.999999, 0.999999))
    vi = 1.0 / (ns - 3)
    wi = 1.0 / vi
    fixed = np.sum(wi * z) / np.sum(wi)
    q = np.sum(wi * (z - fixed) ** 2)
    df = len(z) - 1
    c_val = np.sum(wi) - np.sum(wi ** 2) / np.sum(wi)
    tau2 = max(0.0, (q - df) / c_val) if c_val > 0 else 0.0
    w_re = 1.0 / (vi + tau2)
    z_re = np.sum(w_re * z) / np.sum(w_re)
    se_re = np.sqrt(1.0 / np.sum(w_re))
    z_stat = z_re / se_re
    p_value = 2.0 * norm.sf(abs(z_stat))
    ci_low, ci_high = z_re - 1.96 * se_re, z_re + 1.96 * se_re
    return {
        "r": np.tanh(z_re),
        "ci_low": np.tanh(ci_low),
        "ci_high": np.tanh(ci_high),
        "p": p_value,
        "tau2": tau2,
        "q": q,
    }


def load_subject_fcv_matrix(regions):
    with open(FC_SUBJECT_FILE, "rb") as f:
        fc_data = pickle.load(f)
    region_to_idx = {name: idx for idx, name in enumerate(fs.region)}
    rows = []
    for region in regions:
        region_idx = region_to_idx[region]
        values = np.asarray(fc_data["final_fc_std_mean_data"][region_idx], dtype=float)
        if values.size != len(SUBJECT_IDS):
            raise ValueError(f"{region} has {values.size} subject FCV values, expected {len(SUBJECT_IDS)}")
        rows.append(values)
    return np.vstack(rows)


def subject_correlation_table(regions, divisions, post_dca, subject_fcv):
    rows = []
    for subj_pos, subject_id in enumerate(SUBJECT_IDS):
        y = subject_fcv[:, subj_pos]
        mask = np.isfinite(post_dca) & np.isfinite(y)
        r_value, p_value = pearsonr(post_dca[mask], y[mask])
        ci_low, ci_high = fisher_ci(r_value, int(np.sum(mask)))
        rows.append({
            "Subject": subject_id,
            "N_regions": int(np.sum(mask)),
            "PostDCA_FCV_r": r_value,
            "PostDCA_FCV_p": p_value,
            "CI95_low": ci_low,
            "CI95_high": ci_high,
        })
    df = pd.DataFrame(rows)
    re = random_effects_fisher(df["PostDCA_FCV_r"].values, df["N_regions"].values)
    df = pd.concat([df, pd.DataFrame([{
        "Subject": "Random effects",
        "N_regions": int(df["N_regions"].iloc[0]),
        "PostDCA_FCV_r": re["r"],
        "PostDCA_FCV_p": re["p"],
        "CI95_low": re["ci_low"],
        "CI95_high": re["ci_high"],
        "Tau2": re["tau2"],
        "Q": re["q"],
    }])], ignore_index=True)
    return df


def leave_one_animal_out(X_sc, subject_fcv):
    rows = []
    pred_matrix = np.full_like(subject_fcv, np.nan, dtype=float)
    for subj_pos, subject_id in enumerate(SUBJECT_IDS):
        train_idx = [i for i in range(len(SUBJECT_IDS)) if i != subj_pos]
        y_train = np.nanmean(subject_fcv[:, train_idx], axis=1)
        y_test = subject_fcv[:, subj_pos]
        model = linear_model().fit(X_sc, y_train)
        y_pred = model.predict(X_sc)
        pred_matrix[:, subj_pos] = y_pred
        r_value, p_value = pearsonr(y_test, y_pred)
        r2_value = 1.0 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        rows.append({
            "HeldOutSubject": subject_id,
            "N_regions": len(y_test),
            "ObservedPredicted_r": r_value,
            "ObservedPredicted_p": p_value,
            "HeldOut_R2": r2_value,
        })
    stacked_obs = subject_fcv.ravel(order="F")
    stacked_pred = pred_matrix.ravel(order="F")
    stacked_r, stacked_p = pearsonr(stacked_obs, stacked_pred)
    stacked_r2 = 1.0 - np.sum((stacked_obs - stacked_pred) ** 2) / np.sum((stacked_obs - np.mean(stacked_obs)) ** 2)
    rows.append({
        "HeldOutSubject": "Stacked",
        "N_regions": subject_fcv.size,
        "ObservedPredicted_r": stacked_r,
        "ObservedPredicted_p": stacked_p,
        "HeldOut_R2": stacked_r2,
    })
    return pd.DataFrame(rows), pred_matrix


def plot_forest(ax, corr_df):
    plot_df = corr_df[corr_df["Subject"] != "Random effects"].copy()
    summary = corr_df[corr_df["Subject"] == "Random effects"].iloc[0]
    y_pos = np.arange(len(plot_df), 0, -1)
    ax.axvline(0, color="#bbbbbb", lw=1)
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax.plot([row["CI95_low"], row["CI95_high"]], [y, y], color="#444444", lw=1.5)
        ax.scatter(row["PostDCA_FCV_r"], y, s=55, color="#E45756", zorder=3)
    ax.plot([summary["CI95_low"], summary["CI95_high"]], [0, 0], color="#111111", lw=2.0)
    ax.scatter(summary["PostDCA_FCV_r"], 0, s=75, color="#111111", marker="D", zorder=3)
    ax.set_yticks(list(y_pos) + [0])
    ax.set_yticklabels([f"Subject {s}" for s in plot_df["Subject"]] + ["Random effects"])
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel("Post-DCA vs FCV Pearson r")
    ax.set_title("Subject-level consistency")
    ax.text(
        0.04,
        0.05,
        f"summary r = {summary['PostDCA_FCV_r']:.3f}\n{p_text(summary['PostDCA_FCV_p'])}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
    )


def plot_loao_scatter(ax, subject_fcv, pred_matrix, loao_df, divisions):
    y_obs = subject_fcv.ravel(order="F")
    y_pred = pred_matrix.ravel(order="F")
    div_stacked = np.tile(divisions, len(SUBJECT_IDS))
    lo = min(np.nanmin(y_obs), np.nanmin(y_pred))
    hi = max(np.nanmax(y_obs), np.nanmax(y_pred))
    pad = (hi - lo) * 0.08
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#999999", lw=1.1)
    for div in DIVISION_ORDER:
        mask = div_stacked == div
        ax.scatter(
            y_obs[mask],
            y_pred[mask],
            s=32,
            color=DIVISION_COLORS[div],
            edgecolor="none",
            alpha=0.55,
            label=div,
        )
    stacked = loao_df[loao_df["HeldOutSubject"] == "Stacked"].iloc[0]
    subj_df = loao_df[loao_df["HeldOutSubject"] != "Stacked"]
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Held-out animal FCV")
    ax.set_ylabel("Predicted FCV")
    ax.set_title("Leave-one-animal-out prediction")
    ax.text(
        0.04,
        0.96,
        f"stacked r = {stacked['ObservedPredicted_r']:.3f}\n"
        f"{p_text(stacked['ObservedPredicted_p'])}\n"
        f"mean subject r = {subj_df['ObservedPredicted_r'].mean():.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=fs.TICK_FS_2COL - 1)


def plot_replication_note(ax):
    ax.axis("off")
    ax.set_title("Replication structure")
    text = (
        "Structural features\n"
        "Population-level connectome\n"
        "DCA / Post-DCA shared across animals\n\n"
        "Functional features\n"
        "FCV estimated independently in 7 animals\n\n"
        "Control analyses\n"
        "1. Subject-wise Post-DCA--FCV correlations\n"
        "2. Random-effects summary across animals\n"
        "3. Leave-one-animal-out FCV prediction"
    )
    ax.text(0.02, 0.92, text, transform=ax.transAxes, ha="left", va="top", linespacing=1.45)


def make_plot(corr_df, loao_df, subject_fcv, pred_matrix, divisions):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.05, 1.0, 0.95]})
    plot_forest(axes[0], corr_df)
    plot_loao_scatter(axes[1], subject_fcv, pred_matrix, loao_df, divisions)
    plot_replication_note(axes[2])
    for ax, label in zip(axes, list("ABC")):
        ax.text(-0.13, 1.07, label, transform=ax.transAxes,
                fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_PNG, dpi=600)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = coupling.align_matrices()
    post_dca = X_sc[:, sc_labels.index("Post-DCA")]
    subject_fcv = load_subject_fcv_matrix(regions)
    corr_df = subject_correlation_table(regions, divisions, post_dca, subject_fcv)
    loao_df, pred_matrix = leave_one_animal_out(X_sc, subject_fcv)

    corr_df.to_csv(OUT_CORR_CSV, index=False)
    loao_df.to_csv(OUT_LOAO_CSV, index=False)
    make_plot(corr_df, loao_df, subject_fcv, pred_matrix, divisions)

    print(corr_df.to_string(index=False))
    print(loao_df.to_string(index=False))
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CORR_CSV}")
    print(f"Saved {OUT_LOAO_CSV}")


if __name__ == "__main__":
    main()
