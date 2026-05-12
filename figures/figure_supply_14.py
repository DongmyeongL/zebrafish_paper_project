import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

import figure_style as fs

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

fs.set_paper_style()
plt.rcParams.update({
    "font.size": fs.AXIS_LABEL_FS_2COL,
    "axes.labelsize": fs.AXIS_LABEL_FS_2COL,
    "axes.titlesize": fs.AXIS_LABEL_FS_2COL,
    "xtick.labelsize": max(5, fs.TICK_FS_2COL - 2),
    "ytick.labelsize": fs.TICK_FS_2COL,
})

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data")
IN_CSV = os.path.join(DATA, "fc_dynamics_stimulus_fcv_by_subject_region.csv")
OUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure_supply_14.png")
OUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure_supply_14.pdf")
OUT_CSV = os.path.join(PROJECT_ROOT, "output", "stats", "figure_supply_14_stimulus_fcv_heatmap_values.csv")

DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}
STIM_KEEP = [10, 11, 12]
STIM_LABELS = {
    10: "10: OMR forward",
    11: "11: OMR rightward",
    12: "12: OMR leftward",
}


def zscore(values):
    values = np.asarray(values, dtype=float)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - np.nanmean(values)) / sd


def load_heatmap_table():
    df = pd.read_csv(IN_CSV)
    df = df[df["StimulusIndex"].isin(STIM_KEEP)].copy()
    df["SubjectZFCV"] = df.groupby("Subject")["FCV"].transform(zscore)
    region_meta = (
        df[["Region", "RegionID", "Division"]]
        .drop_duplicates("Region")
        .copy()
    )
    region_meta["Division"] = pd.Categorical(
        region_meta["Division"],
        categories=DIVISION_ORDER,
        ordered=True,
    )
    region_meta = region_meta.sort_values(["Division", "RegionID"]).reset_index(drop=True)
    regions = region_meta["Region"].tolist()
    stim_values = STIM_KEEP

    mean_df = (
        df.groupby(["StimulusIndex", "Region"], as_index=False)
        .agg(
            MeanStimulusFCV=("SubjectZFCV", "mean"),
            SEMStimulusFCV=("SubjectZFCV", lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum())),
            N=("SubjectZFCV", "count"),
        )
    )
    mean_df = mean_df.merge(region_meta, on="Region", how="left")
    mean_df = mean_df.sort_values(["StimulusIndex", "Division", "RegionID"])
    mean_df.to_csv(OUT_CSV, index=False)

    lookup = {
        (int(row.StimulusIndex), row.Region): row.MeanStimulusFCV
        for row in mean_df.itertuples(index=False)
    }
    matrix = np.full((len(stim_values), len(regions)), np.nan)
    for i, stim in enumerate(stim_values):
        for j, region in enumerate(regions):
            matrix[i, j] = lookup.get((stim, region), np.nan)
    return matrix, stim_values, region_meta


def add_division_bar(ax, region_meta):
    divisions = region_meta["Division"].astype(str).tolist()
    div_to_idx = {div: i for i, div in enumerate(DIVISION_ORDER)}
    div_arr = np.array([[div_to_idx[d] for d in divisions]])
    div_cmap = ListedColormap([DIVISION_COLORS[d] for d in DIVISION_ORDER])
    ax.imshow(div_arr, aspect="auto", cmap=div_cmap, vmin=-0.5, vmax=len(DIVISION_ORDER) - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    boundaries = division_boundaries(divisions)
    for boundary in boundaries:
        ax.axvline(boundary, color="white", lw=0.8)

    x0 = 0
    for div in DIVISION_ORDER:
        n = sum(d == div for d in divisions)
        if n:
            ax.text(
                x0 + (n - 1) / 2,
                -0.65,
                div,
                ha="center",
                va="bottom",
                fontsize=fs.TICK_FS_2COL,
                color=DIVISION_COLORS[div],
                fontweight="bold",
            )
        x0 += n


def division_boundaries(divisions):
    return [
        idx - 0.5
        for idx in range(1, len(divisions))
        if divisions[idx] != divisions[idx - 1]
    ]


def make_figure():
    matrix, stim_values, region_meta = load_heatmap_table()
    regions = region_meta["Region"].tolist()
    divisions = region_meta["Division"].astype(str).tolist()

    fig = plt.figure(figsize=(8.4, 3.2))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.12, 1.0, 0.05],
        width_ratios=[1.0, 0.035],
        left=0.08,
        right=0.94,
        top=0.90,
        bottom=0.26,
        hspace=0.04,
        wspace=0.04,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    add_division_bar(ax_bar, region_meta)
    vmax = np.nanpercentile(np.abs(matrix), 98)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="PiYG_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    for boundary in division_boundaries(divisions):
        ax.axvline(boundary, color="white", lw=0.8)

    ax.set_xlabel("Region")
    ax.set_ylabel("Stimulus")
    ax.set_xticks(np.arange(len(regions)))
    ax.set_xticklabels(
        regions,
        rotation=90,
        ha="center",
        fontsize=fs.TICK_FS_2COL - 2,
        fontweight="bold",
    )
    for tick_label, division in zip(ax.get_xticklabels(), divisions):
        tick_label.set_color(DIVISION_COLORS[division])
    ax.set_yticks(np.arange(len(stim_values)))
    ax.set_yticklabels([STIM_LABELS[stim] for stim in stim_values])

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Mean subject z-scored FCV")
    cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    make_figure()
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
