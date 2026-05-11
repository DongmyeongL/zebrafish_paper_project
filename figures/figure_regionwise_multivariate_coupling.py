import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSCanonical
from sklearn.preprocessing import StandardScaler

import figure_style as fs
import figure_supply_sc_heatmap as hm

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
OUTPUT = os.path.join(PROJECT_ROOT, "output")
STATS_DIR = os.path.join(OUTPUT, "stats")
STATS_SUMMARY = os.path.join(STATS_DIR, "figure_regionwise_multivariate_coupling_stats.csv")
STATS_SCORES = os.path.join(STATS_DIR, "figure_regionwise_multivariate_coupling_scores.csv")
STATS_WEIGHTS = os.path.join(STATS_DIR, "figure_regionwise_multivariate_coupling_weights.csv")
STATS_NULL = os.path.join(STATS_DIR, "figure_regionwise_multivariate_coupling_permutation_null.csv")
SUBJECT_IDS = range(12, 19)
BASE_NET = os.path.join(DATA, "region_community_io")
METASTABILITY_FILE = os.path.join(DATA, "fc_dynamics_metastability_by_subject_region.csv")

DIVISION_COLORS = {
    "Tel": fs.division_colors[2],
    "Di": fs.division_colors[1],
    "Mes": fs.division_colors[3],
    "Hind": fs.division_colors[0],
}
DIVISION_ORDER = ["Tel", "Di", "Mes", "Hind"]

_tel = {"lOB", "lSP", "lP", "lPO", "rOB", "rSP", "rP", "rPO"}
_di = {"lHb", "lTh", "lPT", "lPrT", "rHb", "rTh", "rPT", "rPrT"}
_mes = {"lTeO", "lTL", "lTS", "rTeO", "rTL", "rTS"}
_hind = {
    "lCb", "lT", "laRF", "lMOS1", "lMOS2", "lMOS3", "lMOS4", "lMOS5",
    "limRF", "lpRF", "lMON", "lNX", "lRa",
    "rCb", "rT", "raRF", "rMOS1", "rMOS2", "rMOS3", "rMOS4", "rMOS5",
    "rimRF", "rpRF", "rMON", "rNX", "rRa",
}

lregion = ["lMON","lCb","lMOS1","lMOS2","lMOS3","lMOS4","lMOS5","lIPN","lIO","lHc","lRa","lT",
           "laRF","limRF","lpRF","lGG","lHb","lHi","lHR","lOG","lOB","lOE","lP","lPi","lPT",
           "lPO","lPrT","lR","lSP","lTeO","lTh","lTL","lTS","lTG","lVR","lNX"]
rregion = ["rMON","rCb","rMOS1","rMOS2","rMOS3","rMOS4","rMOS5","rIPN","rIO","rHc",
           "rRa","rT","raRF","rimRF","rpRF","rGG","rHb","rHi","rHR","rOG","rOB",
           "rOE","rP","rPi","rPT","rPO","rPrT","rR","rSP","rTeO","rTh","rTL",
           "rTS","rTG","rVR","rNX"]
REGION = lregion + rregion
REGION_INDEX = {name: idx for idx, name in enumerate(REGION)}


def _region_to_div(name):
    if name in _tel:
        return "Tel"
    if name in _di:
        return "Di"
    if name in _mes:
        return "Mes"
    if name in _hind:
        return "Hind"
    return None


def _zscore(values):
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values * np.nan
    return (values - np.nanmean(values)) / std


def _load_subject_causality(subject_id):
    return np.load(f"{BASE_NET}/subject_{subject_id}/subject_{subject_id}_causality.npz")


def _load_subject_fc_neighbors(subject_id):
    return np.load(f"{BASE_NET}/subject_{subject_id}/subject_{subject_id}_net_te_drive_fc_neighbors.npz")


def _aggregate_region(values_by_subject, region_orders, n_regions_total):
    region_values = [[] for _ in region_orders]
    order_lookup = {region_idx: pos for pos, region_idx in enumerate(region_orders)}
    for region_num, values in values_by_subject:
        for reg_idx, value in zip(region_num, values):
            pos = order_lookup.get(reg_idx)
            if pos is not None:
                region_values[pos].append(value)
    region_means = np.array([np.mean(samples) if samples else np.nan for samples in region_values], dtype=float)
    return region_means


def load_fc_matrix():
    bar_df = pd.read_csv(os.path.join(DATA, "fig1_prism_D_FCS_FCV_bar.csv"))
    bar_df["Division"] = bar_df["Region"].apply(_region_to_div)
    bar_df = bar_df.sort_values(["Division", "Region"], key=lambda s: s.map({d: i for i, d in enumerate(DIVISION_ORDER)}) if s.name == "Division" else s).reset_index(drop=True)

    regions = bar_df["Region"].values
    region_order = [REGION_INDEX[name] for name in regions]
    fcs_vals = _zscore(bar_df["FCS"].values)
    fcv_vals = _zscore(bar_df["FCV"].values)

    metastability_df = pd.read_csv(METASTABILITY_FILE)
    metastability_region_mean = (
        metastability_df
        .groupby("Region", observed=True)["RegionwiseMetastability"]
        .mean()
    )
    metastability_z = pd.Series(
        _zscore(metastability_region_mean.values),
        index=metastability_region_mean.index,
    )
    metastability_vals = np.array([
        metastability_z.get(region_name, 0.0)
        for region_name in regions
    ], dtype=float)

    net_te_subject_data = []
    for subject_id in SUBJECT_IDS:
        subject_data = _load_subject_causality(subject_id)
        net_te_subject_data.append((subject_data["region_num"], np.nanmean(subject_data["net_te_matrix"], axis=1)))
    net_te_vals = _zscore(_aggregate_region(net_te_subject_data, region_order, len(REGION)))

    neigh_subject_data = []
    for subject_id in SUBJECT_IDS:
        subject_data = _load_subject_causality(subject_id)
        neighbor_data = _load_subject_fc_neighbors(subject_id)
        neigh_subject_data.append((subject_data["region_num"], neighbor_data["fc_neighbor_mean_drive"]))
    neigh_vals = _zscore(_aggregate_region(neigh_subject_data, region_order, len(REGION)))

    X_fc = np.column_stack([fcs_vals, fcv_vals, metastability_vals, net_te_vals, neigh_vals])
    fc_labels = ["FCS", "FCV", "Metastability", "Net TE", "Neighbor TE"]
    divisions = np.array([_region_to_div(r) for r in regions])
    return regions, divisions, X_fc, fc_labels


def load_sc_matrix():
    sc_matrix, sc_labels, sc_regions, sc_divisions = hm._load_sc_feature_matrix()
    X_sc = sc_matrix.T
    sc_labels = [lbl.replace("\n", " ") for lbl in sc_labels]
    return sc_regions, sc_divisions, X_sc, sc_labels


def align_matrices():
    fc_regions, fc_divisions, X_fc, fc_labels = load_fc_matrix()
    sc_regions, sc_divisions, X_sc, sc_labels = load_sc_matrix()

    sc_lookup = {r: i for i, r in enumerate(sc_regions)}
    keep = [i for i, r in enumerate(fc_regions) if r in sc_lookup]
    fc_regions = fc_regions[keep]
    fc_divisions = fc_divisions[keep]
    X_fc = X_fc[keep]
    X_sc = np.vstack([X_sc[sc_lookup[r]] for r in fc_regions])
    valid = np.isfinite(X_fc).all(axis=1) & np.isfinite(X_sc).all(axis=1)
    fc_regions = fc_regions[valid]
    fc_divisions = fc_divisions[valid]
    X_fc = X_fc[valid]
    X_sc = X_sc[valid]
    return fc_regions, fc_divisions, X_fc, X_sc, fc_labels, sc_labels


def fit_pls_with_permutation(X_fc, X_sc, n_perm=2000, seed=0):
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    Xf = scaler_fc.fit_transform(X_fc)
    Xs = scaler_sc.fit_transform(X_sc)

    pls = PLSCanonical(n_components=2, scale=False)
    pls.fit(Xf, Xs)
    fc_scores, sc_scores = pls.transform(Xf, Xs)
    obs_r = pearsonr(fc_scores[:, 0], sc_scores[:, 0])[0]

    rng = np.random.default_rng(seed)
    null_r = np.zeros(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(Xs.shape[0])
        pls_perm = PLSCanonical(n_components=2, scale=False)
        pls_perm.fit(Xf, Xs[perm])
        f_perm, s_perm = pls_perm.transform(Xf, Xs[perm])
        null_r[i] = pearsonr(f_perm[:, 0], s_perm[:, 0])[0]

    p_perm = (1 + np.sum(null_r >= obs_r)) / (n_perm + 1)
    return {
        "pls": pls,
        "fc_scores": fc_scores,
        "sc_scores": sc_scores,
        "obs_r": obs_r,
        "null_r": null_r,
        "p_perm": p_perm,
    }


def make_plot():
    regions, divisions, X_fc, X_sc, fc_labels, sc_labels = align_matrices()
    result = fit_pls_with_permutation(X_fc,X_sc)

    fc_scores = -result["fc_scores"][:, 0]
    sc_scores =-result["sc_scores"][:, 0]
    tel_mask = divisions == "Tel"
    score_df = pd.DataFrame({
        "Region": regions,
        "Division": divisions,
        "FC score": fc_scores,
        "SC score": sc_scores,
        "Coupling score": (fc_scores + sc_scores) / 2.0,
    })

    fig = plt.figure(figsize=(13.8, 8.6))
    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.15, 0.90, 0.90],
        height_ratios=[1.0, 1.0],
        left=0.07, right=0.97, top=0.94, bottom=0.11,
        hspace=0.42, wspace=0.38,
    )

    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_fc = fig.add_subplot(gs[0, 1])
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_box = fig.add_subplot(gs[1, 1])
    ax_null = fig.add_subplot(gs[1, 2])

    ax_scatter.text(-0.10, 1.03, "A", transform=ax_scatter.transAxes,
                    fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom")
    ax_fc.text(-0.18, 1.05, "B", transform=ax_fc.transAxes,
               fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom")
    ax_sc.text(-0.18, 1.05, "C", transform=ax_sc.transAxes,
               fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom")
    ax_box.text(-0.18, 1.05, "D", transform=ax_box.transAxes,
                fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom")
    ax_null.text(-0.18, 1.05, "E", transform=ax_null.transAxes,
                 fontsize=fs.PANEL_LABEL_FS_2COL, fontweight="bold", va="bottom")

    # A: score scatter
    for div in DIVISION_ORDER:
        mask = divisions == div
        ax_scatter.scatter(
            fc_scores[mask], sc_scores[mask],
            s=80 if div == "Tel" else 54,
            color=DIVISION_COLORS[div],
            alpha=0.90 if div == "Tel" else 0.72,
            edgecolor="black" if div == "Tel" else "none",
            linewidth=0.4,
            label=div,
        )
    for region, x, y, div in zip(regions[tel_mask], fc_scores[tel_mask], sc_scores[tel_mask], divisions[tel_mask]):
        ax_scatter.text(x + 0.02, y + 0.02, region, color=DIVISION_COLORS[div],
                        fontsize=fs.TICK_FS_2COL - 2, fontweight="bold")
    m, b = np.polyfit(fc_scores, sc_scores, 1)
    xs = np.linspace(fc_scores.min() - 0.2, fc_scores.max() + 0.2, 100)
    ax_scatter.plot(xs, m * xs + b, color="#444444", lw=2)
    ax_scatter.axhline(0, color="#dddddd", lw=1)
    ax_scatter.axvline(0, color="#dddddd", lw=1)
    ax_scatter.set_xlabel("FC latent score (LV1)")
    ax_scatter.set_ylabel("SC latent score (LV1)")
    ax_scatter.legend(frameon=False, loc="upper left", ncol=2)
    ax_scatter.text(
        0.02, 0.98,
        f"r = {result['obs_r']:.3f}\nperm p = {result['p_perm']:.4f}\nTel n = {tel_mask.sum()}",
        transform=ax_scatter.transAxes,
        ha="left", va="top",
        fontsize=fs.TICK_FS_2COL,
    )

    # B: FC loadings
    fc_load = result["pls"].x_weights_[:, 0]
    colors_fc = [DIVISION_COLORS["Tel"] if lbl == "FCV" else "#8c8c8c" for lbl in fc_labels]
    ax_fc.barh(fc_labels, fc_load, color=colors_fc, alpha=0.9)
    ax_fc.axvline(0, color="#333333", lw=1)
    ax_fc.set_xlabel("FC loading")
    ax_fc.invert_yaxis()

    # C: SC loadings
    sc_load = result["pls"].y_weights_[:, 0]
    colors_sc = [DIVISION_COLORS["Tel"] if lbl == "Post-DCA" else "#8c8c8c" for lbl in sc_labels]
    ax_sc.barh(sc_labels, sc_load, color=colors_sc, alpha=0.9)
    ax_sc.axvline(0, color="#333333", lw=1)
    ax_sc.set_xlabel("SC loading")
    ax_sc.invert_yaxis()

    # D: coupling by division
    sns.boxplot(
        data=score_df, x="Division", y="Coupling score", order=DIVISION_ORDER,
        hue="Division", hue_order=DIVISION_ORDER, legend=False,
        palette=[DIVISION_COLORS[d] for d in DIVISION_ORDER],
        showfliers=False, width=0.45, linewidth=1.0, ax=ax_box,
    )
    sns.stripplot(
        data=score_df, x="Division", y="Coupling score", order=DIVISION_ORDER,
        color="black", size=3, alpha=0.25, jitter=True, ax=ax_box,
    )
    ax_box.set_xlabel("")
    ax_box.set_ylabel("Coupling score")
    ax_box.tick_params(axis="both", which="both", direction="out", bottom=True, left=True, length=4, width=1.2)

    # E: permutation null
    ax_null.hist(result["null_r"], bins=35, color="#bdbdbd", edgecolor="white")
    ax_null.axvline(result["obs_r"], color=DIVISION_COLORS["Tel"], lw=2.5)
    ax_null.set_xlabel("Null r")
    ax_null.set_ylabel("Count")
    ax_null.text(
        0.98, 0.98,
        f"obs r = {result['obs_r']:.3f}\np = {result['p_perm']:.4f}",
        transform=ax_null.transAxes,
        ha="right", va="top",
        fontsize=fs.TICK_FS_2COL,
    )

    fig.savefig(os.path.join(DATA, "figure_regionwise_multivariate_coupling.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(DATA, "figure_regionwise_multivariate_coupling.pdf"), bbox_inches="tight")
    os.makedirs(STATS_DIR, exist_ok=True)
    pd.DataFrame([
        {
            "figure": "figure_regionwise_multivariate_coupling",
            "panel": "A/E",
            "test": "PLSCanonical LV1 permutation",
            "observed_r": result["obs_r"],
            "permutation_p": result["p_perm"],
            "n_permutations": len(result["null_r"]),
            "n_regions": len(regions),
        }
    ]).to_csv(STATS_SUMMARY, index=False)
    score_df.to_csv(STATS_SCORES, index=False)
    pd.DataFrame(
        [{"feature_set": "FC", "feature": label, "weight": value} for label, value in zip(fc_labels, fc_load)]
        + [{"feature_set": "SC", "feature": label, "weight": value} for label, value in zip(sc_labels, sc_load)]
    ).to_csv(STATS_WEIGHTS, index=False)
    pd.DataFrame({"null_r": result["null_r"]}).to_csv(STATS_NULL, index=False)
    print(f"Saved {STATS_SUMMARY}")


if __name__ == "__main__":
    make_plot()
