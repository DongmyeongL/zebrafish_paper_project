"""
Combined C. elegans and Drosophila figure.

Rows:
  C. elegans: network | high/low examples | scatter | FCV box | DCA box
  Drosophila: network | high/low examples | scatter | FCV box | DCA box
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
sys.path.insert(0, str(FIGURES_DIR))

import figure14_celegans_full_combined_FINAL as celegans
import figure15_drosophila_full_combined_FINAL as drosophila


OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure14_15_celegans_drosophila_AFHI_combined.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure14_15_celegans_drosophila_AFHI_combined.pdf"
INHERITED_PANEL_LABELS = {"A", "F", "H", "I"}
PANEL_LABELS_BY_ROW = [["A", "B", "", "C", "D", "E"], ["F", "G", "", "H", "I", "J"]]
BOTTOM_ROW_SHIFT_Y = -0.015
SPECIES_TITLE_OFFSET_Y = 0.064
MEASURE_TICK_LABELSIZE = 7.5
PANEL_LABEL_Y_PAD = 0.016
PANEL_LABEL_X_OFFSETS = {
    "B": -0.030,
    "G": -0.030,
    "C": -0.032,
    "D": -0.032,
    "E": -0.032,
    "H": -0.032,
    "I": -0.032,
    "J": -0.032,
    
}


def remove_inherited_panel_labels(axes: list[plt.Axes]) -> None:
    for ax in axes:
        for text in list(ax.texts):
            if text.get_text() in INHERITED_PANEL_LABELS:
                text.remove()


def add_aligned_panel_labels(fig: plt.Figure, label_axes_by_row: list[list[plt.Axes]]) -> None:
    fig.canvas.draw()
    for axes, panel_labels in zip(label_axes_by_row, PANEL_LABELS_BY_ROW):
        row_y = max(
            ax.get_position().y1
            for ax, label in zip(axes, panel_labels)
            if label
        ) + PANEL_LABEL_Y_PAD
        for ax, label in zip(axes, panel_labels):
            if not label:
                continue
            pos = ax.get_position()
            dx = PANEL_LABEL_X_OFFSETS.get(label, -0.012)
            fig.text(
                pos.x0 + dx,
                row_y,
                label,
                ha="right",
                va="bottom",
                fontsize=13,
                fontweight="bold",
            )


def add_species_title(fig: plt.Figure, axes: list[plt.Axes], text: str) -> None:
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    fig.text(
        (left + right) / 2,
        top + SPECIES_TITLE_OFFSET_Y,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        style="italic" if text.startswith("C.") else "normal",
    )


def shift_axes_y(axes: list[plt.Axes], dy: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def sync_example_xaxis(ax_trace: plt.Axes, ax_corr: plt.Axes, xlim: tuple[float, float] | None = None) -> None:
    if xlim is None:
        xlim = ax_corr.get_xlim()
    ax_trace.set_xlim(*xlim)
    ax_corr.set_xlim(*xlim)


def match_network_dca_label(ax: plt.Axes) -> None:
    for text in ax.texts:
        if text.get_text() == "Post-DCA":
            text.set_text(celegans.DCA_POST_LABEL)


def keep_only_r_p_annotation(ax: plt.Axes) -> None:
    for text in ax.texts:
        lines = text.get_text().splitlines()
        if lines and lines[0].startswith("r="):
            keep = [line for line in lines if line.startswith("r=") or line.startswith("p=")]
            text.set_text("\n".join(keep))


def style_measure_ticks(
    top_axes: list[plt.Axes],
    bottom_axes: list[plt.Axes],
) -> None:
    # Current row axes: network, high example, low example, scatter, FCV box, DCA box.
    panel_axes = [top_axes[3], top_axes[4], top_axes[5], bottom_axes[3], bottom_axes[4], bottom_axes[5]]
    for ax in panel_axes:
        ax.tick_params(axis="both", labelsize=MEASURE_TICK_LABELSIZE)

    # Reduce tick density for panel C. Panel D has categorical x ticks, so keep all x labels.
    top_axes[3].xaxis.set_major_locator(MaxNLocator(nbins=4))
    top_axes[3].yaxis.set_major_locator(MaxNLocator(nbins=4))


def celegans_example_title(example: dict, prefix: str) -> str:
    pair = example["pair"]
    return f"{prefix}: {pair['neuron_a']}-{pair['neuron_b']}"


def drosophila_example_title(example: dict, prefix: str) -> str:
    return f"{prefix}: {example['label_a']}-{example['label_b']}"


def draw_celegans_example(
    fig: plt.Figure,
    subspec,
    example: dict,
    kind: str,
    title: str,
    show_corr_xlabel: bool = True,
) -> plt.Axes:
    gs = subspec.subgridspec(2, 1, height_ratios=[1.35, 0.8], hspace=0.08)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[1, 0])
    celegans.rep.add_trace_panel(ax_trace, example, kind)
    for text in list(ax_trace.texts):
        text.remove()
    ax_trace.set_title(title, fontsize=8.5, pad=2)
    celegans.rep.add_corr_panel(ax_corr, example, kind)
    sync_example_xaxis(ax_trace, ax_corr, (float(example["time_s"][0]), float(example["time_s"][-1])))
    if not show_corr_xlabel:
        ax_corr.set_xlabel("")
    return ax_trace


def draw_drosophila_trace_like_celegans(ax: plt.Axes, example: dict, kind: str) -> None:
    t = drosophila.np.arange(len(example["trace_a"])) / drosophila.TRACE_SAMPLING_RATE_HZ
    start = int(example.get("display_start", 0))
    stop = int(example.get("display_stop", min(len(t), 900)))
    tt = t[start:stop] - t[start]
    duration = float(tt[-1]) if len(tt) else 0.0
    colors = drosophila.TRACE_PAIR_COLORS[kind]
    offsets = [1.45, -1.45]
    labels = [example["label_a"], example["label_b"]]
    traces = [example["trace_a"][start:stop], example["trace_b"][start:stop]]

    for name, color, offset, trace in zip(labels, colors, offsets, traces):
        ax.plot(tt, trace + offset, color=color, lw=0.75)
    ax.axis("off")
    ax.axvline(duration, color="#999999", lw=0.7, ls=":")
    ax.set_yticks([])
    ax.set_ylabel("Calcium\nz-score", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xlim(0, duration + 45)


def draw_drosophila_example(
    fig: plt.Figure,
    subspec,
    example: dict,
    kind: str,
    title: str,
    show_corr_xlabel: bool = True,
) -> plt.Axes:
    gs = subspec.subgridspec(2, 1, height_ratios=[1.35, 0.8], hspace=0.08)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_corr = fig.add_subplot(gs[1, 0])
    draw_drosophila_trace_like_celegans(ax_trace, example, kind)
    ax_trace.set_title(title, fontsize=8.5, pad=2)
    drosophila.draw_corr_panel(ax_corr, example)
    sync_example_xaxis(ax_trace, ax_corr)
    if not show_corr_xlabel:
        ax_corr.set_xlabel("")
    remove_inherited_panel_labels([ax_trace, ax_corr])
    return ax_trace


def draw_celegans_row(fig: plt.Figure, row: dict) -> tuple[list[plt.Axes], list[plt.Axes]]:
    celegans.configure_shared_style()
    _, _, _, high_example, low_example, network_nodes, network_sc = celegans.load_representation_examples()
    neurons, scatter_neurons, recording_points = celegans.summary.load_panel_data()

    ax_network = fig.add_subplot(row["network"])
    celegans.rep.add_network_panel(ax_network, network_nodes, network_sc, high_example, low_example)
    match_network_dca_label(ax_network)
    celegans.add_panel_label(ax_network, "A", x=-0.10)

    ex_grid = row["examples"].subgridspec(2, 1, hspace=0.48)
    ax_high = draw_celegans_example(
        fig,
        ex_grid[0, 0],
        high_example,
        "high",
        celegans_example_title(high_example, "High FCV"),
        show_corr_xlabel=False,
    )
    ax_low = draw_celegans_example(
        fig, ex_grid[1, 0], low_example, "low", celegans_example_title(low_example, "Low FCV")
    )

    ax_scatter = fig.add_subplot(row["scatter"])
    celegans.add_scatter_panel(ax_scatter, scatter_neurons, "F")
    keep_only_r_p_annotation(ax_scatter)

    ax_fcv = fig.add_subplot(row["fcv_box"])
    celegans.summary.add_box_panel(ax_fcv, recording_points, "FCV_z", "zFCV", "H", rasterized=True)
    celegans.replace_panel_label(ax_fcv, "H")
    ax_fcv.set_ylim(-3.0, 6.7)

    ax_dca = fig.add_subplot(row["dca_box"])
    celegans.summary.add_box_panel(ax_dca, neurons, "PostDCA", celegans.DCA_POST_LABEL, "I")
    celegans.replace_panel_label(ax_dca, "I")
    ax_dca.set_ylim(-0.24, 0.19)

    axes = [ax_network, ax_high, ax_low, ax_scatter, ax_fcv, ax_dca]
    remove_inherited_panel_labels(axes)
    return axes, axes


def draw_drosophila_row(fig: plt.Figure, row: dict) -> tuple[list[plt.Axes], list[plt.Axes]]:
    drosophila.set_style()

    ax_network = fig.add_subplot(row["network"])
    drosophila.draw_ito48_network(ax_network)

    high_example, low_example = drosophila.choose_trace_examples()
    ex_grid = row["examples"].subgridspec(2, 1, hspace=0.48)
    ax_high = draw_drosophila_example(
        fig,
        ex_grid[0, 0],
        high_example,
        "high",
        drosophila_example_title(high_example, "High FCV"),
        show_corr_xlabel=False,
    )
    ax_low = draw_drosophila_example(
        fig, ex_grid[1, 0], low_example, "low", drosophila_example_title(low_example, "Low FCV")
    )

    ax_scatter = fig.add_subplot(row["scatter"])
    drosophila.draw_best_scatter(ax_scatter, "F")
    keep_only_r_p_annotation(ax_scatter)

    j_points = drosophila.load_recording_fcv_points_for_j()
    k_points = pd.read_csv(drosophila.SCATTER_SCORES).dropna(
        subset=["weighted_mean_PostDCA_positive", "big_group"]
    )
    ax_fcv = fig.add_subplot(row["fcv_box"])
    drosophila.draw_group_box_panel(
        ax_fcv,
        j_points,
        "FCV_z",
        "zFCV",
        "H",
        point_size=3.0,
        point_alpha=0.10,
        jitter_width=0.22,
        sig_start_y=4.2,
    )
    ax_fcv.set_ylim(top=7.1)

    ax_dca = fig.add_subplot(row["dca_box"])
    drosophila.draw_group_box_panel(
        ax_dca,
        k_points,
        "weighted_mean_PostDCA_positive",
        drosophila.DCA_POST_LABEL,
        "I",
        point_size=3.0,
        point_alpha=0.20,
        jitter_width=0.22,
    )
    axes = [ax_network, ax_high, ax_low, ax_scatter, ax_fcv, ax_dca]
    remove_inherited_panel_labels(axes)
    return axes, axes


def prepare_layout() -> tuple[plt.Figure, list[dict]]:
    fig = plt.figure(figsize=(16, 6.2))
    grid = fig.add_gridspec(
        2,
        5,
        left=0.075,
        right=0.985,
        top=0.935,
        bottom=0.105,
        width_ratios=[1.45, 0.92, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        hspace=0.70,
        wspace=0.40,
    )
    rows = [
        {
            "network": grid[row, 0],
            "examples": grid[row, 1],
            "scatter": grid[row, 2],
            "fcv_box": grid[row, 3],
            "dca_box": grid[row, 4],
        }
        for row in range(2)
    ]
    return fig, rows


def save_figure(fig: plt.Figure) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.04, transparent=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.04)


def main() -> None:
    fig, rows = prepare_layout()
    top_axes, top_label_axes = draw_celegans_row(fig, rows[0])
    bottom_axes, bottom_label_axes = draw_drosophila_row(fig, rows[1])
    fig.canvas.draw()
    shift_axes_y(bottom_axes, BOTTOM_ROW_SHIFT_Y)
    style_measure_ticks(top_axes, bottom_axes)
    fig.canvas.draw()
    add_aligned_panel_labels(fig, [top_label_axes, bottom_label_axes])
    add_species_title(fig, top_axes, "C. elegans")
    add_species_title(fig, bottom_axes, "Drosophila")
    save_figure(fig)
    plt.close(fig)
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")


if __name__ == "__main__":
    main()
