import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from figure_style import set_paper_style


set_paper_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
NETWORK_DIR = DATA / "network_diagrams"
STATS_DIR = DATA / "figure6_NULL_P_SP"
OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_5.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_5.pdf"

PAIR_SPECS = [
    {
        "label": "rP-rimRF",
        "base": "network_rP_rimRF_network_diagarm.png",
        "null_in": "network_null_in_rP_rimRF_network_diagarm.png",
        "null_out": "network_null_out_rP_rimRF_network_diagarm.png",
    },
    {
        "label": "rP-rMOS4",
        "base": "network_rP_rMOS4_network_diagarm.png",
        "null_in": "network_null_in_rP_rMOS4_network_diagarm.png",
        "null_out": "network_null_out_rP_rMOS4_network_diagarm.png",
    },
]

COLUMNS = [
    ("base", "Original"),
    ("null_in", "Null-In"),
    ("null_out", "Null-Out"),
]


def p_text(p_value):
    return f"p = {p_value:.3g}" if p_value >= 0.001 else "p < 0.001"


def read_network_image(filename):
    path = NETWORK_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def plot_null_box(ax, npz_name, ylabel, xticklabels):
    data = np.load(STATS_DIR / npz_name)
    base = -1 * np.asarray(data["x_data"], dtype=float)
    null = -1 * np.asarray(data["y_data"], dtype=float)
    base = base[np.isfinite(base)]
    null = null[np.isfinite(null)]
    p_value = mannwhitneyu(base, null, alternative="two-sided").pvalue

    bp = ax.boxplot(
        [base, null],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], ["#6baed6", "#fdae6b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for idx, values in enumerate([base, null], start=1):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=8,
            color="black",
            alpha=0.38,
            linewidth=0,
            zorder=3,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.text(
        0.98,
        0.94,
        p_text(p_value),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        fontstyle="italic",
    )
    ax.tick_params(axis="both", which="both", direction="out", length=4, width=1.0)


def add_panel_label(ax, label):
    ax.text(
        -0.05,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def shrink_axis_width(ax, width_scale=0.70):
    pos = ax.get_position()
    cx = pos.x0 + pos.width / 2
    new_w = pos.width * width_scale
    ax.set_position([cx - new_w / 2, pos.y0, new_w, pos.height])


def make_figure():
    fig = plt.figure(figsize=(8.2, 6.7))
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[0.85, 1.0, 1.0],
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=0.055,
        wspace=0.25,
        hspace=0.18,
    )

    ax_stat_out = fig.add_subplot(gs[0, 0:3])
    ax_stat_in = fig.add_subplot(gs[0, 3:6])
    plot_null_box(
        ax_stat_out,
        "figure6_network_properites_in.npz",
        "Post-DCA",
        ["Base", "Null-Out"],
    )
    plot_null_box(
        ax_stat_in,
        "figure6_network_properites_out.npz",
        "Pre-DCA",
        ["Base", "Null-In"],
    )
    shrink_axis_width(ax_stat_out, width_scale=0.70)
    shrink_axis_width(ax_stat_in, width_scale=0.70)

    network_axes = np.empty((len(PAIR_SPECS), len(COLUMNS)), dtype=object)
    for row_idx in range(len(PAIR_SPECS)):
        for col_idx in range(len(COLUMNS)):
            network_axes[row_idx, col_idx] = fig.add_subplot(
                gs[row_idx + 1, col_idx * 2:(col_idx + 1) * 2]
            )

    add_panel_label(ax_stat_out, "A")
    add_panel_label(ax_stat_in, "B")
    for row_idx, pair in enumerate(PAIR_SPECS):
        for col_idx, (key, col_title) in enumerate(COLUMNS):
            ax = network_axes[row_idx, col_idx]
            image = read_network_image(pair[key])
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=11, fontweight="bold", pad=4)
            if col_idx == 0:
                ax.text(
                    -0.08,
                    0.5,
                    pair["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=11,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
            if col_idx == 0:
                add_panel_label(ax, "C" if row_idx == 0 else "F")

    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    make_figure()
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
