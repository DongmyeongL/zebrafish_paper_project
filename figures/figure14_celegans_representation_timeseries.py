"""
Representation example for C. elegans calcium time series.

Selects one high-FCV neuron pair from M1-M3 and one low-FCV neuron pair
from M3-M4 using the final spontaneous nrec3 FCV matrix, then plots
spontaneous calcium traces and sliding-window correlation dynamics from a
recording where both pairs are present.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "figure14_celegans"
ARCHIVE = BASE / "data" / "atanas_kim_2023_www_archive.bz2"
LABEL_CACHE = BASE / "data" / "figure14_celegans_activity_label_cache.json"
ORDER = BASE / "matrices" / "figure14_celegans_w60_nozero_matrix_neuron_order_nrec3_spontaneous.csv"
CLUSTERS = BASE / "matrices" / "figure14_celegans_w60_nozero_SC_modularity_sohn2011_clusters_k5_nrec3_spontaneous.csv"
SC_MATRIX = BASE / "matrices" / "figure14_celegans_w60_nozero_SC_matrix_nrec3_spontaneous.csv"
FCV_MATRIX = BASE / "matrices" / "figure14_celegans_w60_nozero_FCV_matrix_nrec3_spontaneous.csv"

OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure14_celegans_representation_timeseries.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure14_celegans_representation_timeseries.pdf"
OUT_CSV = BASE / "results" / "figure14_celegans_representation_timeseries_pairs.csv"

WINDOW = 60
STEP = 15
UID_PREFIX = "2023"

PAIR_COLORS = {
    "high": ("#4e79a7", "#59a14f"),
    "low": ("#59a14f", "#e15759"),
}
CLUSTER_ORDER = ["M1", "M2", "M4", "M3", "M5"]
CLUSTER_COLORS = {
    "M1": "#4e79a7",
    "M2": "#f28e2b",
    "M3": "#59a14f",
    "M4": "#e15759",
    "M5": "#b07aa1",
}


def member_uid(name: str) -> str:
    stem = Path(name).stem
    return stem.replace("atanas_kim_2023_", "", 1)


def load_json_from_member(member_file: io.BufferedReader) -> dict:
    return json.load(member_file)


def clean_label(label: str) -> str | None:
    if not label:
        return None
    label = str(label).strip()
    if not label or "?" in label or label.endswith("-alt"):
        return None
    return label


def zscore_traces(traces: np.ndarray) -> np.ndarray:
    traces = traces.astype(float, copy=True)
    for i in range(traces.shape[0]):
        x = traces[i]
        finite = np.isfinite(x)
        if finite.sum() < 5:
            traces[i] = np.nan
            continue
        if not finite.all():
            idx = np.arange(len(x))
            x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])
        sd = np.nanstd(x)
        if sd <= 1e-12:
            traces[i] = np.nan
        else:
            traces[i] = (x - np.nanmean(x)) / sd
    return traces


def pair_recording_uids(label_cache: dict, neuron_a: str, neuron_b: str) -> list[str]:
    uids = []
    for uid, labels in label_cache.items():
        if not uid.startswith(UID_PREFIX):
            continue
        seen = set()
        for meta in labels.values():
            label = clean_label(meta.get("label", ""))
            if label in (neuron_a, neuron_b):
                seen.add(label)
        if {neuron_a, neuron_b}.issubset(seen):
            uids.append(uid)
    return sorted(uids)


def ranked_pairs(fcv: pd.DataFrame, clusters: pd.DataFrame, cluster_a: int, cluster_b: int, high: bool) -> list[dict]:
    cluster_map = dict(zip(clusters["neuron"], clusters["modularity_cluster_k5"]))
    values = fcv.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    computable = set(fcv.index[np.isfinite(values).any(axis=0) | np.isfinite(values).any(axis=1)])

    rows = []
    for i, neuron_a in enumerate(fcv.index):
        for neuron_b in fcv.columns[i + 1 :]:
            if neuron_a not in computable or neuron_b not in computable:
                continue
            if {cluster_map.get(neuron_a), cluster_map.get(neuron_b)} != {cluster_a, cluster_b}:
                continue
            fcv_value = fcv.loc[neuron_a, neuron_b]
            if np.isfinite(fcv_value):
                rows.append(
                    {
                        "matrix_fcv": float(fcv_value),
                        "neuron_a": neuron_a,
                        "neuron_b": neuron_b,
                        "cluster_a": int(cluster_map[neuron_a]),
                        "cluster_b": int(cluster_map[neuron_b]),
                    }
                )
    return sorted(rows, key=lambda row: row["matrix_fcv"], reverse=high)


def computable_neurons_from_fcv() -> set[str]:
    fcv = pd.read_csv(FCV_MATRIX, index_col=0)
    values = fcv.to_numpy(float)
    np.fill_diagonal(values, np.nan)
    ok = np.isfinite(values).any(axis=0) | np.isfinite(values).any(axis=1)
    return set(fcv.index[ok])


def load_network_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_neurons = computable_neurons_from_fcv()
    order = pd.read_csv(ORDER)
    clusters = pd.read_csv(CLUSTERS)
    nodes = order.merge(clusters, on="neuron", how="inner")
    nodes = nodes[nodes["neuron"].isin(valid_neurons)].copy()
    nodes["cluster_label"] = nodes["modularity_cluster_k5"].map(lambda value: f"M{int(value)}")
    nodes["cluster_order"] = nodes["cluster_label"].map({label: i for i, label in enumerate(CLUSTER_ORDER)})
    nodes = nodes.sort_values(["cluster_order", "DCA", "neuron"], ascending=[True, False, True]).reset_index(drop=True)

    sc = pd.read_csv(SC_MATRIX, index_col=0).loc[nodes["neuron"], nodes["neuron"]]
    return nodes, sc


def network_positions(nodes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    postdca = nodes["PostDCA"].to_numpy(float)
    postdca_min = float(np.nanmin(postdca))
    postdca_max = float(np.nanmax(postdca))
    denom = postdca_max - postdca_min if postdca_max > postdca_min else 1.0

    positions = {}
    for label_index, cluster_label in enumerate(CLUSTER_ORDER):
        sub = nodes[nodes["cluster_label"].eq(cluster_label)].sort_values(["PostDCA", "neuron"], ascending=[False, True])
        if sub.empty:
            continue
        jitter = np.linspace(-0.32, 0.32, len(sub)) if len(sub) > 1 else np.array([0.0])
        for offset, (_, row) in zip(jitter, sub.iterrows()):
            x = (postdca_max - float(row["PostDCA"])) / denom
            y = len(CLUSTER_ORDER) - 1 - label_index + offset
            positions[row["neuron"]] = (x, y)
    return positions


def add_network_panel(ax, nodes: pd.DataFrame, sc: pd.DataFrame, high_example: dict, low_example: dict) -> None:
    positions = network_positions(nodes)
    max_weight = float(np.nanmax(sc.to_numpy(float))) if np.isfinite(sc.to_numpy(float)).any() else 1.0

    for source in sc.index:
        for target in sc.columns:
            weight = float(sc.loc[source, target])
            if not np.isfinite(weight) or weight <= 0 or source == target:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            arrow = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=4.8,
                shrinkA=2.8,
                shrinkB=2.8,
                lw=0.16 + 0.34 * np.log1p(weight) / np.log1p(max_weight),
                color="#3f3f3f",
                alpha=0.18,
                zorder=1,
            )
            ax.add_patch(arrow)

    for cluster_label in CLUSTER_ORDER:
        sub = nodes[nodes["cluster_label"].eq(cluster_label)]
        if sub.empty:
            continue
        xy = np.asarray([positions[neuron] for neuron in sub["neuron"]])
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=24,
            c=CLUSTER_COLORS[cluster_label],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
            label=cluster_label,
        )

    highlighted = {
        high_example["pair"]["neuron_a"],
        high_example["pair"]["neuron_b"],
        low_example["pair"]["neuron_a"],
        low_example["pair"]["neuron_b"],
    }
    #for neuron in sorted(highlighted):
    #    if neuron not in positions:
    #        continue
    #    x, y = positions[neuron]
    #    ax.scatter([x], [y], s=56, facecolor="none", edgecolor="#111111", linewidth=1.15, zorder=4)

    y_ticks = [len(CLUSTER_ORDER) - 1 - i for i in range(len(CLUSTER_ORDER))]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(CLUSTER_ORDER, fontsize=9.5)
    for tick, cluster_label in zip(ax.get_yticklabels(), CLUSTER_ORDER):
        tick.set_color(CLUSTER_COLORS[cluster_label])
        tick.set_fontweight("bold")
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.72, len(CLUSTER_ORDER) - 0.28)

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
    )
    ax.text(
        0.92,
        arrow_y - 0.15,
        "low",
        ha="left",
        va="top",
        fontsize=8,
        color="#222222",
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
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)


def choose_examples() -> tuple[str, dict, dict]:
    label_cache = json.loads(LABEL_CACHE.read_text())
    fcv = pd.read_csv(FCV_MATRIX, index_col=0)
    clusters = pd.read_csv(CLUSTERS)

    high_candidates = ranked_pairs(fcv, clusters, cluster_a=1, cluster_b=3, high=True)
    low_candidates = ranked_pairs(fcv, clusters, cluster_a=3, cluster_b=4, high=False)

    for high_pair in high_candidates[:80]:
        high_uids = set(pair_recording_uids(label_cache, high_pair["neuron_a"], high_pair["neuron_b"]))
        if not high_uids:
            continue
        for low_pair in low_candidates[:120]:
            low_uids = set(pair_recording_uids(label_cache, low_pair["neuron_a"], low_pair["neuron_b"]))
            shared = sorted(high_uids & low_uids)
            if shared:
                return shared[0], high_pair, low_pair

    raise RuntimeError("Could not find high and low example pairs in a shared 2023 recording.")


def load_recording(uid: str) -> dict:
    with tarfile.open(ARCHIVE, "r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if member_uid(member.name) != uid:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                break
            return load_json_from_member(extracted)
    raise FileNotFoundError(f"Could not find recording {uid} in {ARCHIVE}")


def extract_pair(data: dict, label_cache: dict, uid: str, pair: dict) -> dict:
    traces = zscore_traces(np.asarray(data["gcamp"]["trace_array"], dtype=float))
    heat_frame = int(data.get("timing", {}).get("event", {}).get("heat", [traces.shape[1]])[0])
    mean_timestep = float(data.get("timing", {}).get("mean_timestep", 1.0))
    labels = label_cache[uid]

    label_to_idx = {}
    for key, meta in labels.items():
        label = clean_label(meta.get("label", ""))
        if label in (pair["neuron_a"], pair["neuron_b"]):
            idx0 = int(key) - 1
            if 0 <= idx0 < traces.shape[0]:
                label_to_idx[label] = idx0

    missing = {pair["neuron_a"], pair["neuron_b"]} - set(label_to_idx)
    if missing:
        raise ValueError(f"Missing labels in {uid}: {sorted(missing)}")

    sub = traces[[label_to_idx[pair["neuron_a"]], label_to_idx[pair["neuron_b"]]], :heat_frame]
    time_s = np.arange(sub.shape[1]) * mean_timestep

    corr_t = []
    corr_values = []
    for start in range(0, sub.shape[1] - WINDOW + 1, STEP):
        window = sub[:, start : start + WINDOW]
        if np.isfinite(window).all() and np.nanstd(window[0]) > 1e-12 and np.nanstd(window[1]) > 1e-12:
            corr_values.append(float(np.corrcoef(window[0], window[1])[0, 1]))
        else:
            corr_values.append(np.nan)
        corr_t.append((start + WINDOW / 2) * mean_timestep)

    pair = dict(pair)
    pair["recording_fcv"] = float(np.nanstd(corr_values))
    return {
        "pair": pair,
        "time_s": time_s,
        "traces": sub,
        "corr_t": np.asarray(corr_t),
        "corr_values": np.asarray(corr_values),
    }


def add_trace_panel(ax, example: dict, kind: str) -> None:
    pair = example["pair"]
    colors = PAIR_COLORS[kind]
    names = [pair["neuron_a"], pair["neuron_b"]]
    offsets = [1.45, -1.45]

    for i, (name, color, offset) in enumerate(zip(names, colors, offsets)):
        ax.plot(example["time_s"], example["traces"][i] + offset, color=color, lw=0.75)
        ax.text(example["time_s"][-1] + 8, offset, name, color=color, va="center", fontsize=8)
    ax.axis('off')
    ax.axvline(example["time_s"][-1], color="#999999", lw=0.7, ls=":")
    ax.set_yticks([])
    ax.set_ylabel("Calcium\nz-score", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_xlim(example["time_s"][0], example["time_s"][-1] + 55)


def add_corr_panel(ax, example: dict, kind: str) -> None:
    pair = example["pair"]
    ax.plot(example["corr_t"], example["corr_values"], color="#222222", lw=1.0)
    ax.axhline(0, color="#777777", lw=0.6, alpha=0.45)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time(s)", fontsize=8)
    ax.set_ylabel("FC", fontsize=8)
    ax.tick_params(axis="both", labelsize=7, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    #ax.text(
    #    0.03,
    #    0.92,
    #    f"recording FCV={pair['recording_fcv']:.3f}",
    #    transform=ax.transAxes,
    #    va="top",
    #    fontsize=7,
    #)


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Missing WormWideWeb archive: {ARCHIVE}")

    uid, high_pair, low_pair = choose_examples()
    label_cache = json.loads(LABEL_CACHE.read_text())
    data = load_recording(uid)
    high_example = extract_pair(data, label_cache, uid, high_pair)
    low_example = extract_pair(data, label_cache, uid, low_pair)
    network_nodes, network_sc = load_network_data()

    fig = plt.figure(figsize=(10.0, 3.55))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 1.0, 1.0], height_ratios=[1.35, 0.8], hspace=0.18, wspace=0.36)
    ax_network = fig.add_subplot(gs[:, 0])
    axes = np.array([[fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])], [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]])

    add_network_panel(ax_network, network_nodes, network_sc, high_example, low_example)
    add_trace_panel(axes[0, 0], high_example, "high")
    add_trace_panel(axes[0, 1], low_example, "low")
    add_corr_panel(axes[1, 0], high_example, "high")
    add_corr_panel(axes[1, 1], low_example, "low")

    fig.subplots_adjust(left=0.055, right=0.975, top=0.88, bottom=0.16)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)

    rows = []
    for kind, example in [("high", high_example), ("low", low_example)]:
        row = {"example": kind, "uid": uid}
        row.update(example["pair"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")
    print(f"Network nodes={len(network_nodes)}, edges={(network_sc.to_numpy() > 0).sum()}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
