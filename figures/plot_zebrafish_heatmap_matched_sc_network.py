import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

import figure_style as fs
import figure_supply_sc_heatmap as hm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUTPUT_PNG = PROJECT_ROOT / "output" / "png" / "zebrafish_heatmap_matched_region_sc_network_figure14A_style.png"
OUTPUT_PDF = PROJECT_ROOT / "output" / "pdf" / "zebrafish_heatmap_matched_region_sc_network_figure14A_style.pdf"
SC_CACHE_NPZ = DATA / "zebrafish_heatmap_matched_region_sc_network_data.npz"
RAW_SC_ROOT = Path("/remotenas2/remotedata/share/zebrafish/original_raw_data")
N_REGION = 72
SUBJECTS = range(12, 19)

DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]
DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}


def load_heatmap_region_set():
    _, _, heatmap_regions, heatmap_divisions = hm._load_sc_feature_matrix()
    region_to_idx = {name: idx for idx, name in enumerate(fs.region[:N_REGION])}
    valid_regions = [region_to_idx[name] for name in heatmap_regions]
    return valid_regions, heatmap_regions, heatmap_divisions


def load_post_dca_axis(valid_regions):
    dac = np.load(DATA / "total_selected_region_dac_data.npz", allow_pickle=True)
    post_dca = np.nanmean(dac["arr_2"], axis=1)
    finite_fill = np.nanmedian(post_dca[np.isfinite(post_dca)])
    post_dca = np.where(np.isfinite(post_dca), post_dca, finite_fill)
    post_min = float(np.nanmin(post_dca[valid_regions]))
    post_max = float(np.nanmax(post_dca[valid_regions]))
    post_denom = post_max - post_min if post_max > post_min else 1.0
    return post_dca, post_max, post_denom


def aggregate_region_sc(valid_regions):
    count_sum = np.zeros((N_REGION, N_REGION), dtype=np.float64)
    valid_regions = np.asarray(valid_regions, dtype=np.int64)

    for subject_id in SUBJECTS:
        path = RAW_SC_ROOT / f"subject_{subject_id}_data_cellular_synapse_sc_100_data.pkl"
        print(f"Loading subject {subject_id}")
        with open(path, "rb") as f:
            data = pickle.load(f)

        sc = np.asarray(data["cellular_sc_list"], dtype=np.int64)
        neuron_cluster = np.asarray(data["neuron_region_id"], dtype=np.int64)
        root_area = np.asarray(data["root_area"], dtype=np.int64)
        neuron_region = root_area[neuron_cluster]
        src = neuron_region[sc[:, 0]]
        dst = neuron_region[sc[:, 1]]
        valid = (
            (src >= 0) & (src < N_REGION)
            & (dst >= 0) & (dst < N_REGION)
            & (src != dst)
            & np.isin(src, valid_regions)
            & np.isin(dst, valid_regions)
        )
        flat = src[valid] * N_REGION + dst[valid]
        count_sum += np.bincount(flat, minlength=N_REGION * N_REGION).reshape(N_REGION, N_REGION)

    np.fill_diagonal(count_sum, 0)
    return count_sum / len(list(SUBJECTS))


def load_or_create_region_sc(valid_regions, heatmap_regions, heatmap_divisions):
    valid_regions = np.asarray(valid_regions, dtype=np.int64)
    if SC_CACHE_NPZ.exists():
        cached = np.load(SC_CACHE_NPZ, allow_pickle=True)
        cached_valid_regions = cached["valid_regions"].astype(np.int64)
        if np.array_equal(cached_valid_regions, valid_regions):
            print(f"Loading cached SC data: {SC_CACHE_NPZ}")
            return cached["weights"]
        print("Cached SC data region set differs from heatmap; regenerating cache")

    weights = aggregate_region_sc(valid_regions)
    np.savez_compressed(
        SC_CACHE_NPZ,
        weights=weights,
        valid_regions=valid_regions,
        heatmap_regions=np.asarray(heatmap_regions, dtype=str),
        heatmap_divisions=np.asarray(heatmap_divisions, dtype=str),
        subjects=np.asarray(list(SUBJECTS), dtype=np.int64),
    )
    print(f"Saved cached SC data: {SC_CACHE_NPZ}")
    return weights


def build_nodes(valid_regions, heatmap_regions, heatmap_divisions, post_dca):
    return [
        {
            "idx": idx,
            "region": region_name,
            "division": str(division),
            "post_dca": float(post_dca[idx]),
        }
        for idx, region_name, division in zip(valid_regions, heatmap_regions, heatmap_divisions)
    ]


def network_positions(nodes, post_max, post_denom):
    positions = {}
    for row_idx, division in enumerate(DIVISION_ORDER):
        sub = sorted(
            [node for node in nodes if node["division"] == division],
            key=lambda node: (-node["post_dca"], node["region"]),
        )
        if not sub:
            continue
        jitter = np.linspace(-0.32, 0.32, len(sub)) if len(sub) > 1 else np.array([0.0])
        for offset, node in zip(jitter, sub):
            x = (post_max - node["post_dca"]) / post_denom
            y = len(DIVISION_ORDER) - 1 - row_idx + offset
            positions[node["idx"]] = (x, y)
    return positions


def draw_network(valid_regions, nodes, positions, weights):
    edge_count = int((weights > 0).sum())
    max_weight = float(np.nanmax(weights)) if np.isfinite(weights).any() else 1.0

    fs.set_paper_style()
    fig, ax = plt.subplots(figsize=(4.9, 6.1))

    for source in valid_regions:
        for target in valid_regions:
            weight = float(weights[source, target])
            if weight <= 0:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=12.2,
                    shrinkA=2.8,
                    shrinkB=2.8,
                    lw=0.10 + 0.32 * np.log1p(weight) / np.log1p(max_weight),
                    color="#444444",
                    alpha=0.105,
                    zorder=1,
                )
            )

    for division in DIVISION_ORDER:
        sub = [node for node in nodes if node["division"] == division]
        if not sub:
            continue
        xy = np.asarray([positions[node["idx"]] for node in sub], dtype=float)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=70,
            c=DIVISION_COLORS[division],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )

    y_ticks = [len(DIVISION_ORDER) - 1 - i for i in range(len(DIVISION_ORDER))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(DIVISION_ORDER, fontsize=10, fontweight="bold")
    for tick_label, division in zip(ax.get_yticklabels(), DIVISION_ORDER):
        tick_label.set_color(DIVISION_COLORS[division])

    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.72, len(DIVISION_ORDER) - 0.28)

    arrow_y = -0.5
    x0, x1, x2 = 0.08, 0.86, 0.94
    left_half, right_half, head_half = 0.042, 0.010, 0.034
    arrow_vertices = [
        (x0, arrow_y - left_half),
        (x1, arrow_y - right_half),
        (x1, arrow_y - head_half),
        (x2, arrow_y),
        (x1, arrow_y + head_half),
        (x1, arrow_y + right_half),
        (x0, arrow_y + left_half),
    ]
    ax.add_patch(
        Polygon(
            arrow_vertices,
            closed=True,
            facecolor="#5b5b5b",
            edgecolor="none",
            alpha=0.82,
            clip_on=False,
            zorder=6,
        )
    )
    ax.text(
        0.02,
        arrow_y - 0.15,
        "high",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
        #fontweight="bold",
    )
    
    ax.text(
        0.92,
        arrow_y - 0.15,
        "low",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
       # fontweight="bold",
    )
    
    ax.text(
        0.42,
        arrow_y - 0.24,
        "Post-DCA",
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        fontweight="bold",
    )
    
    #ax.text(0.92, arrow_y - 0.18, "low", ha="right", va="top", fontsize=6.5, color="#9a9a9a")
    
    
    #ax.set_title(f"Directed SC network\nn={len(valid_regions)}, edges={edge_count}", fontsize=9)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, edge_count


def main():
    valid_regions, heatmap_regions, heatmap_divisions = load_heatmap_region_set()
    print(f"Heatmap-matched regions: {len(valid_regions)}")
    post_dca, post_max, post_denom = load_post_dca_axis(valid_regions)
    weights = load_or_create_region_sc(valid_regions, heatmap_regions, heatmap_divisions)
    nodes = build_nodes(valid_regions, heatmap_regions, heatmap_divisions, post_dca)
    positions = network_positions(nodes, post_max, post_denom)
    fig, edge_count = draw_network(valid_regions, nodes, positions, weights)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", transparent=False)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", transparent=False)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Nodes drawn: {len(valid_regions)}")
    print(f"Edges drawn: {edge_count}")


if __name__ == "__main__":
    sys.exit(main())
