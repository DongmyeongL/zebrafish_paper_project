import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, pearsonr

import figure_style as fs

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

fs.set_paper_style()
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data")
FC_SHIFT_CSV = os.path.join(DATA, "fc_dynamics_state_fc_shift_by_subject_region.csv")
STIMULUS_FCV_CSV = os.path.join(DATA, "fc_dynamics_stimulus_fcv_by_subject_region.csv")
FC_SHIFT_MEAN_CSV = os.path.join(DATA, "fc_dynamics_separate_fc_shift_region_mean.csv")
FCV_MEAN_CSV = os.path.join(DATA, "fc_dynamics_separate_fcv_region_mean.csv")
SPONTANEOUS_FCV_CSV = os.path.join(DATA, "fig1_prism_D_FCS_FCV_bar.csv")
DCA_NPZ = os.path.join(DATA, "total_selected_region_dac_data.npz")

OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_fc_dynamics_final_fc_shift_summary.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_fc_dynamics_final_fc_shift_summary.pdf")
OUT_SUMMARY = os.path.join(PROJECT_ROOT, "output", "figure_fc_dynamics_final_fc_shift_summary.csv")
STATS_DIR = os.path.join(PROJECT_ROOT, "output", "stats")
OUT_STATS_SUMMARY = os.path.join(STATS_DIR, "figure_fc_dynamics_final_fc_shift_summary_stats.csv")
OUT_REGION_SUMMARY = os.path.join(STATS_DIR, "figure_fc_dynamics_final_fc_shift_summary_region_values.csv")

DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}
STIM_KEEP = [10, 11, 12]
PANEL_FS = 11
STAT_FS = 8
LABEL_FS = 7
TICK_FS = 8
AXIS_FS = 9
POINT_SIZE = 30
TEL_ANNOTATIONS = set()
TEL_LABEL_OFFSETS = {
    "lOB": (5, 4),
    "rP": (5, 5),
}


def p_text(p_value):
    return f"p = {p_value:.3g}" if p_value >= 0.001 else "p < 0.001"


def zscore(values):
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - np.nanmean(values)) / sd


def center_on_mean(values):
    values = np.asarray(values, dtype=float)
    return values - np.nanmean(values)


def summarize_region_readout(df, value_col, mean_col, sem_col):
    return (
        df.groupby(["Region", "RegionID", "Division", "PostDCA"], as_index=False)
        .agg(
            **{
                mean_col: (value_col, "mean"),
                sem_col: (value_col, lambda x: x.std(ddof=1) / np.sqrt(x.count())),
            }
        )
    )


def add_bilateral_pairs(df):
    df = df.copy()
    regions = set(df["Region"].dropna())
    valid_pairs = {
        region[1:]
        for region in regions
        if len(region) > 1 and region[0] in {"l", "r"} and f"{'r' if region[0] == 'l' else 'l'}{region[1:]}" in regions
    }
    df["RegionPair"] = df["Region"].map(lambda r: r[1:] if isinstance(r, str) and len(r) > 1 and r[0] in {"l", "r"} else np.nan)
    return df[df["RegionPair"].isin(valid_pairs)].copy()


def average_bilateral_pairs(df, value_cols):
    pair_df = add_bilateral_pairs(df)
    group_cols = [col for col in ["Subject", "StimulusIndex", "RegionPair"] if col in pair_df.columns]
    agg = {col: (col, "mean") for col in value_cols if col in pair_df.columns}
    if "RegionID" in pair_df.columns:
        agg["RegionID"] = ("RegionID", "min")
    if "Division" in pair_df.columns:
        agg["Division"] = ("Division", "first")
    if "PostDCA" in pair_df.columns:
        agg["PostDCA"] = ("PostDCA", "mean")
    if "NeuronCount" in pair_df.columns:
        agg["NeuronCount"] = ("NeuronCount", "sum")
    if "NTimepoints" in pair_df.columns:
        agg["NTimepoints"] = ("NTimepoints", "first")
    if "NWindows" in pair_df.columns:
        agg["NWindows"] = ("NWindows", "first")
    pair_df = pair_df.groupby(group_cols, as_index=False).agg(**agg)
    return pair_df.rename(columns={"RegionPair": "Region"})


def load_raw_postdca(mean_df):
    dca_data = np.load(DCA_NPZ)
    post_dca_by_region = dca_data["arr_2"]
    rows = []
    for row in mean_df[["Region", "RegionID"]].itertuples(index=False):
        values = np.asarray(post_dca_by_region[int(row.RegionID)], dtype=float)
        rows.append({
            "Region": row.Region,
            "RawPostDCA": np.nanmean(values),
        })
    return pd.DataFrame(rows)


def compute_panel_a_heatmap_values(readout_df, value_col, z_col, mean_col):
    heat_df = readout_df.copy()
    heat_df[z_col] = (
        heat_df
        .groupby(["Subject", "StimulusIndex"])[value_col]
        .transform(zscore)
    )
    return (
        heat_df
        .groupby(["StimulusIndex", "Region"], as_index=False)
        .agg(**{mean_col: (z_col, "mean")})
    )


def compute_panel_a_raw_heatmap_values(readout_df, value_col, mean_col):
    return (
        readout_df
        .groupby(["StimulusIndex", "Region"], as_index=False)
        .agg(**{mean_col: (value_col, "mean")})
    )


def compute_panel_a_subject_z_heatmap_values(readout_df, value_col, z_col, mean_col):
    heat_df = readout_df.copy()
    heat_df[z_col] = (
        heat_df
        .groupby("Subject")[value_col]
        .transform(zscore)
    )
    return (
        heat_df
        .groupby(["StimulusIndex", "Region"], as_index=False)
        .agg(**{mean_col: (z_col, "mean")})
    )


def compute_region_panel_a_summary(readout_df, value_col, z_col, mean_col, prefix):
    heat_df = compute_panel_a_heatmap_values(readout_df, value_col, z_col, mean_col)
    mean_summary_col = f"{prefix}MeanSubjectZ"
    std_summary_col = f"{prefix}StdSubjectZ"
    return (
        heat_df
        .groupby("Region", as_index=False)
        .agg(
            **{
                mean_summary_col: (mean_col, "mean"),
                std_summary_col: (mean_col, "std"),
            }
        )
    )


def compute_region_panel_a_raw_summary(readout_df, value_col, mean_col, prefix):
    heat_df = compute_panel_a_raw_heatmap_values(readout_df, value_col, mean_col)
    mean_summary_col = f"{prefix}Mean"
    std_summary_col = f"{prefix}Std"
    return (
        heat_df
        .groupby("Region", as_index=False)
        .agg(
            **{
                mean_summary_col: (mean_col, "mean"),
                std_summary_col: (mean_col, "std"),
            }
        )
    )


def compute_region_panel_a_subject_z_summary(readout_df, value_col, z_col, mean_col, prefix):
    heat_df = compute_panel_a_subject_z_heatmap_values(readout_df, value_col, z_col, mean_col)
    mean_summary_col = f"{prefix}MeanSubjectZ"
    std_summary_col = f"{prefix}StdSubjectZ"
    return (
        heat_df
        .groupby("Region", as_index=False)
        .agg(
            **{
                mean_summary_col: (mean_col, "mean"),
                std_summary_col: (mean_col, "std"),
            }
        )
    )



def load_tables():
    fc_shift = pd.read_csv(FC_SHIFT_CSV)
    fc_shift = fc_shift[fc_shift["StimulusIndex"].isin(STIM_KEEP)].copy()
    stim_fcv = pd.read_csv(STIMULUS_FCV_CSV)
    stim_fcv = stim_fcv[stim_fcv["StimulusIndex"].isin(STIM_KEEP)].copy()
    stim_fcv["SubjectZFCV"] = stim_fcv.groupby("Subject")["FCV"].transform(zscore)
    spontaneous_fcv = pd.read_csv(SPONTANEOUS_FCV_CSV).rename(columns={"FCV": "SpontaneousFCV", "FCV_SEM": "SpontaneousFCVSEM"})

    shift_mean = summarize_region_readout(fc_shift, "FCShift", "MeanFCShift", "SEMFCShift")
    fcv_mean = summarize_region_readout(stim_fcv, "SubjectZFCV", "MeanStimulusFCV", "SEMStimulusFCV")
    raw_postdca = load_raw_postdca(shift_mean)

    mean_df = shift_mean.merge(
        fcv_mean[["Region", "MeanStimulusFCV", "SEMStimulusFCV"]],
        on="Region",
        how="inner",
    )
    mean_df = mean_df.merge(
        spontaneous_fcv[["Region", "SpontaneousFCV", "SpontaneousFCVSEM"]],
        on="Region",
        how="inner",
    )
    mean_df = mean_df.merge(raw_postdca, on="Region", how="inner")
    panel_a_fcv_mean = compute_region_panel_a_subject_z_summary(
        stim_fcv,
        "FCV",
        "SubjectZFCV",
        "MeanSubjectZFCV",
        "PanelAFCV",
    )
    mean_df = mean_df.merge(panel_a_fcv_mean, on="Region", how="left")
    panel_a_raw_fcv_mean = compute_region_panel_a_raw_summary(
        stim_fcv,
        "FCV",
        "MeanFCV",
        "PanelARawFCV",
    )
    mean_df = mean_df.merge(panel_a_raw_fcv_mean, on="Region", how="left")
    stimulus_fcv_std = (
        stim_fcv
        .groupby("Region", as_index=False)
        .agg(StdStimulusFCV=("SubjectZFCV", "std"))
    )
    mean_df = mean_df.merge(stimulus_fcv_std, on="Region", how="left")
    mean_df["MeanStimulusFCVZ"] = zscore(mean_df["MeanStimulusFCV"])
    mean_df["StdStimulusFCVZ"] = zscore(mean_df["StdStimulusFCV"])
    shift_std = (
        fc_shift
        .groupby("Region", as_index=False)
        .agg(StdFCShift=("FCShift", "std"))
    )
    mean_df = mean_df.merge(shift_std, on="Region", how="left")
    mean_df["Division"] = pd.Categorical(mean_df["Division"], categories=DIVISION_ORDER, ordered=True)
    mean_df = mean_df.sort_values(["Division", "RegionID"]).reset_index(drop=True)
    return fc_shift, stim_fcv, mean_df


def region_order(mean_df):
    return mean_df.sort_values(["Division", "RegionID"])["Region"].tolist()


def plot_heatmap(ax, cax, bar_ax, stim_fcv, mean_df):
    regions = region_order(mean_df)
    stim_values = STIM_KEEP
    heat_df = compute_panel_a_subject_z_heatmap_values(
        stim_fcv[stim_fcv["Region"].isin(regions)],
        "FCV",
        "SubjectZFCV",
        "MeanSubjectZFCV",
    )
    lookup = {(row.StimulusIndex, row.Region): row.MeanSubjectZFCV for row in heat_df.itertuples(index=False)}
    matrix = np.full((len(stim_values), len(regions)), np.nan)
    for i, stim in enumerate(stim_values):
        for j, region in enumerate(regions):
            matrix[i, j] = lookup.get((stim, region), np.nan)
    vmax = np.nanmax(np.abs(matrix))
    im = ax.imshow(matrix, aspect="auto", cmap="PiYG_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_yticks(np.arange(len(stim_values)))
    ax.set_yticklabels([str(v) for v in stim_values])
    ax.set_ylabel("Stimulus index")
    ax.set_xlabel("Region", labelpad=10)
    ax.set_title("")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(regions, rotation=90, fontsize=5)
    ax.tick_params(axis="x", length=2, pad=1)

    boundaries = []
    divs = mean_df.set_index("Region").loc[regions, "Division"].tolist()
    for idx in range(1, len(divs)):
        if divs[idx] != divs[idx - 1]:
            boundaries.append(idx - 0.5)
    for boundary in boundaries:
        ax.axvline(boundary, color="white", lw=0.7)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Mean subject z-scored FCV", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    div_to_idx = {div: i for i, div in enumerate(DIVISION_ORDER)}
    div_arr = np.array([[div_to_idx[d] for d in divs]])
    div_cmap = ListedColormap([DIVISION_COLORS[d] for d in DIVISION_ORDER])
    bar_ax.imshow(div_arr, aspect="auto", cmap=div_cmap, vmin=-0.5, vmax=len(DIVISION_ORDER) - 0.5)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    bar_ax.set_title("")
    for spine in bar_ax.spines.values():
        spine.set_visible(False)
    for boundary in boundaries:
        bar_ax.axvline(boundary, color="white", lw=0.7)
    x0 = 0
    for div in DIVISION_ORDER:
        n = sum(d == div for d in divs)
        if n:
            bar_ax.text(x0 + (n - 1) / 2, -0.75, div, ha="center", va="bottom", fontsize=8, color=DIVISION_COLORS[div], fontweight="bold")
        x0 += n


def scatter_panel(ax, df, x_col, y_col, xlabel, ylabel, title, label_regions=False, label_tel=False):
    colors = [DIVISION_COLORS[d] for d in df["Division"]]
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        ax=ax,
        scatter=False,
        ci=95,
        line_kws={"color": "black", "lw": 1.2},
    )
    ax.scatter(df[x_col], df[y_col], c=colors, s=POINT_SIZE, edgecolor="white", linewidth=0.45, alpha=0.90)
    mask = df[x_col].notna() & df[y_col].notna()
    r_value, p_value = pearsonr(df.loc[mask, x_col], df.loc[mask, y_col])
    ax.text(
        0.05,
        0.95,
        f"r = {r_value:.3f}\n{p_text(p_value)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=STAT_FS,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
    )
    if label_regions:
        label_df = pd.concat([
            df.sort_values("MeanFCShift", ascending=False).head(3),
            df[df["Division"] == "Tel"].sort_values("MeanFCShift", ascending=False).head(3),
        ]).drop_duplicates("Region")
        for row in label_df.itertuples(index=False):
            ax.annotate(
                row.Region,
                (getattr(row, x_col), getattr(row, y_col)),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                color=DIVISION_COLORS[row.Division],
                fontweight="bold" if row.Division == "Tel" else "normal",
            )
    if label_tel:
        label_df = df[(df["Division"] == "Tel") & (df["Region"].isin(TEL_ANNOTATIONS))]
        for row in label_df.itertuples(index=False):
            offset = TEL_LABEL_OFFSETS.get(row.Region, (4, 3))
            ax.annotate(
                row.Region,
                (getattr(row, x_col), getattr(row, y_col)),
                xytext=offset,
                textcoords="offset points",
                fontsize=LABEL_FS,
                color=DIVISION_COLORS["Tel"],
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.54, pad=0.25),
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=TICK_FS, pad=2)
    ax.xaxis.label.set_size(AXIS_FS)
    ax.yaxis.label.set_size(AXIS_FS)
    return r_value, p_value


def shrink_axes(ax, width_scale=0.70, height_scale=0.56):
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    cy = pos.y0 + pos.height / 2
    new_w = pos.width * width_scale
    new_h = pos.height * height_scale
    ax.set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])


def plot_division_box(ax, df):
    vals, labels = [], []
    for div in DIVISION_ORDER:
        v = df.loc[df["Division"] == div, "StdFCShift"].dropna().values
        vals.extend(v)
        labels.extend([div] * len(v))
    sns.boxplot(
        x=labels,
        y=vals,
        order=DIVISION_ORDER,
        hue=labels,
        hue_order=DIVISION_ORDER,
        legend=False,
        palette=[DIVISION_COLORS[d] for d in DIVISION_ORDER],
        showfliers=False,
        width=0.5,
        linewidth=1.0,
        ax=ax,
    )
    sns.stripplot(x=labels, y=vals, order=DIVISION_ORDER, color="black", size=3, alpha=0.25, jitter=True, ax=ax)
    tel = df.loc[df["Division"] == "Tel", "StdFCShift"].dropna()
    other = df.loc[df["Division"] != "Tel", "StdFCShift"].dropna()
    u_res = mannwhitneyu(tel, other, alternative="greater")
    ax.text(
        0.05,
        0.95,
        f"Tel > other\n{p_text(u_res.pvalue)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
    )
    ax.set_xlabel("")
    ax.set_ylabel("STD FC shift")
    ax.set_title("Division summary")
    return u_res.pvalue


def make_summary(mean_df, r_mean, p_mean, r_std, p_std, r_postdca, p_postdca):
    top = mean_df.sort_values("MeanStimulusFCV", ascending=False).iloc[0]
    top_tel = mean_df[mean_df["Division"] == "Tel"].sort_values("MeanStimulusFCV", ascending=False).iloc[0]
    rows = [
        {"Metric": "MeanStimulusFCV_vs_SpontaneousFCV_r", "Value": r_mean},
        {"Metric": "MeanStimulusFCV_vs_SpontaneousFCV_p", "Value": p_mean},
        {"Metric": "StimulusWiseFCVModulationZ_vs_SpontaneousFCV_r", "Value": r_std},
        {"Metric": "StimulusWiseFCVModulationZ_vs_SpontaneousFCV_p", "Value": p_std},
        {"Metric": "MeanStimulusFCV_vs_PostDCA_r", "Value": r_postdca},
        {"Metric": "MeanStimulusFCV_vs_PostDCA_p", "Value": p_postdca},
        {"Metric": "TopRegion", "Value": top["Region"]},
        {"Metric": "TopRegionDivision", "Value": top["Division"]},
        {"Metric": "TopRegionMeanStimulusFCV", "Value": top["MeanStimulusFCV"]},
        {"Metric": "TopTelRegion", "Value": top_tel["Region"]},
        {"Metric": "TopTelMeanStimulusFCV", "Value": top_tel["MeanStimulusFCV"]},
    ]
    os.makedirs(STATS_DIR, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False)
    summary.to_csv(OUT_STATS_SUMMARY, index=False)
    mean_df.to_csv(OUT_REGION_SUMMARY, index=False)


def make_figure():
    _, _, mean_df = load_tables()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(8.0, 2.55),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.070, right=0.990, bottom=0.235, top=0.820, wspace=0.42)
    r_mean, p_mean = scatter_panel(
        ax_a,
        mean_df,
        "SpontaneousFCV",
        "MeanStimulusFCV",
        "Spontaneous FCV",
        "Mean stimulus FCV",
        "",
        label_tel=True,
    )
    r_std, p_std = scatter_panel(
        ax_b,
        mean_df,
        "SpontaneousFCV",
        "StdStimulusFCVZ",
        "Spontaneous FCV",
        "SD stimulus FCV (z)",
        "",
        label_tel=True,
    )
    r_postdca, p_postdca = scatter_panel(
        ax_c,
        mean_df,
        "RawPostDCA",
        "MeanStimulusFCV",
        "Post-DCA",
        "Mean stimulus FCV",
        "",
        label_tel=True,
    )

    for ax in [ax_a, ax_c]:
        bottom, _ = ax.get_ylim()
        ax.set_ylim(bottom, 1.0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    for label, ax in zip(["A", "B", "C"], [ax_a, ax_b, ax_c]):
        ax.text(-0.32, 1.055, label, transform=ax.transAxes, fontsize=PANEL_FS, fontweight="bold", ha="left", va="bottom")

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=DIVISION_COLORS[d], markeredgecolor="white", markersize=4.8, label=d)
        for d in DIVISION_ORDER
    ]
    ax_a.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.99),
        ncol=2,
        frameon=False,
        handletextpad=0.25,
        columnspacing=0.55,
        fontsize=8.5,
    )
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    make_summary(mean_df, r_mean, p_mean, r_std, p_std, r_postdca, p_postdca)


def main():
    make_figure()
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_SUMMARY}")
    print(f"Saved {OUT_STATS_SUMMARY}")
    print(f"Saved {OUT_REGION_SUMMARY}")
    print(pd.read_csv(OUT_SUMMARY).to_string(index=False))


if __name__ == "__main__":
    main()
