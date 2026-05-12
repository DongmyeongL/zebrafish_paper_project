import os
from types import SimpleNamespace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"


SUBJECT_ID = 13
P_REGION_ID = 22
BRAIN_DIVISION_LIST = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 4, 1, 1, 1, 4, 2, 2, 2, 1,
    1, 2, 1, 4, 2, 3, 1, 3, 3, 4, 0, 0,
] * 2)
BRAIN_DIVISION_ORDER = [2, 1, 3, 0]
BRAIN_DIVISION_LABELS = ["Tel", "Di", "Mes", "Hind"]
CAUSALITY_NPZ = (
    DATA
    / "region_community_io"
    / "subject_13"
    / "subject_13_causality.npz"
)
EXAMPLE_NPZ = DATA / "figure_supply_13" / "figure_supply_13_example.npz"
OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_13.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_13.pdf"


def draw_net_te_panel(ax, causality_npz, module):
    region_num = causality_npz["region_num"]
    net_te = causality_npz["net_te_matrix"]

    community_division = BRAIN_DIVISION_LIST[region_num]
    division_indices = []
    for div_id in BRAIN_DIVISION_ORDER:
        idx = np.flatnonzero(community_division == div_id)
        idx = idx[np.lexsort((region_num[idx] % 36, region_num[idx] >= 36))]
        division_indices.append(idx)
    order = np.concatenate(division_indices)
    sorted_net = net_te[np.ix_(order, order)]
    abs_max = float(np.nanmax(np.abs(sorted_net)))

    im = ax.imshow(sorted_net, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="auto")
    boundaries = np.cumsum([len(idx) for idx in division_indices])[:-1] - 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="black", linewidth=1.3)
        ax.axvline(boundary, color="black", linewidth=1.3)

    ax.set_xlabel("Target community")
    ax.set_ylabel("Source community")
    sizes = [len(idx) for idx in division_indices]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    centers = starts + np.array(sizes) / 2 - 0.5
    ax.set_xticks(centers)
    ax.set_xticklabels(BRAIN_DIVISION_LABELS, fontsize=8)
    ax.set_yticks(centers)
    ax.set_yticklabels(BRAIN_DIVISION_LABELS, fontsize=8)
    return im


def draw_activity_panel(ax, src_z, tgt_z, pair, module):
    x = np.arange(len(src_z))
    ax.plot(x, src_z, linewidth=1.2, label=f"P C{pair['src_global_idx'] + 1}")
    ax.plot(
        x,
        tgt_z,
        linewidth=1.2,
        alpha=0.75,
        label=f"{module.REGION_NAMES[pair['tgt_region_id']]} C{pair['tgt_global_idx'] + 1}",
    )
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Frame")
    ax.set_ylabel("z-score")
    ax.legend(fontsize=7, loc="upper right")


def draw_null_panel(ax, te_null, te_forward):
    ax.hist(te_null, bins=30, color="lightgray", edgecolor="black")
    ax.axvline(te_forward, color="crimson", linewidth=2.0)
    ax.set_title("Permutation null")
    ax.set_xlabel("Forward TE (bits)")
    ax.set_ylabel("Count")


def draw_direction_panel(ax, te_forward, te_reverse):
    ax.bar(["source->target", "target->source"], [te_forward, te_reverse],
           color=["crimson", "steelblue"])
    ax.set_title("Directional TE")
    ax.set_ylabel("TE (bits)")
    ax.tick_params(axis="x", labelrotation=18)


def add_panel_labels(axes, labels):
    for ax, label in zip(axes, labels):
        ax.text(
            -0.12,
            1.08,
            label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )


def main():
    causality_npz = np.load(CAUSALITY_NPZ, allow_pickle=True)
    example = np.load(EXAMPLE_NPZ, allow_pickle=True)
    module = SimpleNamespace(REGION_NAMES=example["region_names"].tolist())
    pair = example["pair"].item()
    src_z = example["src_z"]
    tgt_z = example["tgt_z"]
    te_forward = float(example["te_forward"])
    te_reverse = float(example["te_reverse"])
    te_null = example["te_null"]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 4.6),
        gridspec_kw={"width_ratios": [1.2, 1.7, 1.0, 0.9]},
    )

    im = draw_net_te_panel(axes[0], causality_npz, module)
    draw_activity_panel(axes[1], src_z, tgt_z, pair, module)
    draw_null_panel(axes[2], te_null, te_forward)
    draw_direction_panel(axes[3], te_forward, te_reverse)
    add_panel_labels(axes, ["A", "B", "C", "D"])

    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Net TE")
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()
