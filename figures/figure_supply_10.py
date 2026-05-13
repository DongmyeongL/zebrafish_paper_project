import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figure_style as fs
from figure_style import add_panel_label_fig, darw_region_bar


fs.set_paper_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_10.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_10.pdf"
SUBJECT_IDS = range(12, 19)
BASE_NET = DATA / "region_community_io"

DIVISION_COLUMNS = ["Tel", "Di", "Mes", "Hind"]
DIVISION_ORDER = {"Tel": 0, "Di": 1, "Mes": 2, "Hind": 3}

TEL_REGIONS = {"lOB", "lSP", "lP", "lPO", "rOB", "rSP", "rP", "rPO"}
DI_REGIONS = {"lHb", "lTh", "lPT", "lPrT", "rHb", "rTh", "rPT", "rPrT"}
MES_REGIONS = {"lTeO", "lTL", "lTS", "rTeO", "rTL", "rTS"}
HIND_REGIONS = {
    "lCb", "lT", "laRF", "lMOS1", "lMOS2", "lMOS3", "lMOS4", "lMOS5",
    "limRF", "lpRF", "lMON", "lNX", "lRa",
    "rCb", "rT", "raRF", "rMOS1", "rMOS2", "rMOS3", "rMOS4", "rMOS5",
    "rimRF", "rpRF", "rMON", "rNX", "rRa",
}


def first_existing_path(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return Path(paths[0])


def region_to_division(region_name):
    if region_name in TEL_REGIONS:
        return "Tel"
    if region_name in DI_REGIONS:
        return "Di"
    if region_name in MES_REGIONS:
        return "Mes"
    if region_name in HIND_REGIONS:
        return "Hind"
    return None


def zscore(values):
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - np.nanmean(values)) / std


def load_subject_causality(subject_id):
    return np.load(BASE_NET / f"subject_{subject_id}" / f"subject_{subject_id}_causality.npz")


def load_subject_fc_neighbors(subject_id):
    return np.load(
        BASE_NET
        / f"subject_{subject_id}"
        / f"subject_{subject_id}_net_te_drive_fc_neighbors.npz"
    )


def aggregate_subject_region_lists(values_by_subject, region_order):
    region_values = [[] for _ in region_order]
    order_lookup = {region_idx: pos for pos, region_idx in enumerate(region_order)}

    for region_num, values in values_by_subject:
        region_num = np.asarray(region_num)
        values = np.asarray(values, dtype=float)
        for reg_idx in region_order:
            mask = region_num == reg_idx
            if not np.any(mask):
                continue
            region_subject_values = values[mask]
            region_subject_values = region_subject_values[np.isfinite(region_subject_values)]
            if region_subject_values.size:
                region_values[order_lookup[reg_idx]].append(float(np.mean(region_subject_values)))

    return region_values


def ordered_lists_to_region_lists(region_order, ordered_lists):
    region_lists = [[] for _ in range(len(fs.region))]
    selected = []
    for region_idx, values in zip(region_order, ordered_lists):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            region_lists[int(region_idx)] = arr.tolist()
            selected.append(int(region_idx))
    return region_lists, selected


def replace_region_with_division_values(ordered_lists, region_order, target_region):
    target_idx = fs.region.index(target_region)
    if target_idx not in region_order:
        return ordered_lists

    target_pos = region_order.index(target_idx)
    target_division = fs.brain_division_list[target_idx]
    division_values = []

    for region_idx, values in zip(region_order, ordered_lists):
        if region_idx == target_idx:
            continue
        if fs.brain_division_list[region_idx] != target_division:
            continue
        arr = np.asarray(values, dtype=float)
        division_values.extend(arr[np.isfinite(arr)].tolist())

    if division_values:
        ordered_lists[target_pos] = division_values
    return ordered_lists


# ============================================================
# Load Figure 9 panel A heatmap features
# ============================================================

bar_df = pd.read_csv(first_existing_path(DATA / "fig1_prism_D_FCS_FCV_bar.csv"))
bar_df["_div"] = bar_df["Region"].apply(region_to_division)
bar_df["_div_order"] = bar_df["_div"].map(DIVISION_ORDER)
bar_df = (
    bar_df.dropna(subset=["_div"])
    .sort_values(["_div_order", "Region"])
    .reset_index(drop=True)
)

regions_for_heatmap = bar_df["Region"].to_numpy()
region_order = [fs.region.index(region_name) for region_name in regions_for_heatmap]

fc_data_path = first_existing_path(
    DATA / "figure1_box_scatter_fc_mean_var_data.pkl",
)
with open(fc_data_path, "rb") as f:
    fc_box_data = pickle.load(f)

fcs_lists = [
    fc_box_data["final_fc_mean_data"][region_idx]
    for region_idx in region_order
]
fcv_lists = [
    fc_box_data["final_fc_std_mean_data"][region_idx]
    for region_idx in region_order
]

metastability_df = pd.read_csv(
    first_existing_path(
        DATA / "fc_dynamics_metastability_by_subject_region.csv",
    )
)
metastability_lists = []
for region_name in regions_for_heatmap:
    values = metastability_df.loc[
        metastability_df["Region"] == region_name,
        "RegionwiseMetastability",
    ].to_numpy(dtype=float)
    metastability_lists.append(values[np.isfinite(values)].tolist())

net_te_subject_data = []
neighbor_subject_data = []
for subject_id in SUBJECT_IDS:
    causality = load_subject_causality(subject_id)
    net_te_subject_data.append(
        (causality["region_num"], np.nanmean(causality["net_te_matrix"], axis=1))
    )

    neighbor = load_subject_fc_neighbors(subject_id)
    neighbor_subject_data.append(
        (causality["region_num"], neighbor["fc_neighbor_mean_drive"])
    )

net_te_lists = aggregate_subject_region_lists(net_te_subject_data, region_order)
neighbor_net_te_lists = aggregate_subject_region_lists(neighbor_subject_data, region_order)
net_te_lists = replace_region_with_division_values(net_te_lists, region_order, "rTS")
neighbor_net_te_lists = replace_region_with_division_values(
    neighbor_net_te_lists, region_order, "rTS"
)

feature_panels = [
    ("A", "FCS(z-score)", fcs_lists),
    ("B", "FCV(z-score)", fcv_lists),
    ("C", "Metastability", metastability_lists),
    ("D", r"$\mathrm{TE}_{\mathrm{net}}$", net_te_lists),
    ("E", "Neighbor " + r"$\mathrm{TE}_{\mathrm{net}}$", neighbor_net_te_lists),
]


# ============================================================
# Create Figure Supply 10
# ============================================================

fig = plt.figure(figsize=(16, 16))
axes = [
    plt.subplot2grid((5, 6), (row, 0), colspan=6)
    for row in range(5)
]

for ax, (panel_label, ylabel, values) in zip(axes, feature_panels):
    region_lists, selected_regions = ordered_lists_to_region_lists(region_order, values)
    darw_region_bar(ax, region_lists, selected_regions)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1.1, len(selected_regions))
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        bottom=True,
        left=True,
        length=4,
        width=1.2,
    )
    ax.yaxis.set_label_coords(-0.055, 0.5)
    add_panel_label_fig(fig, ax, panel_label, dx=-0.08, dy=0.01)

for ax in axes:
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.02, pos.y0, pos.width * 0.95, pos.height * 0.90])

plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
