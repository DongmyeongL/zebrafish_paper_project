import importlib.util
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import figure_style as fig_help


fig_help.set_paper_style()
plt.rcParams.update(
    {
        "font.size": fig_help.AXIS_LABEL_FS_2COL,
        "axes.labelsize": fig_help.AXIS_LABEL_FS_2COL,
        "axes.titlesize": fig_help.AXIS_LABEL_FS_2COL,
        "xtick.labelsize": fig_help.TICK_FS_2COL,
        "ytick.labelsize": fig_help.TICK_FS_2COL,
    }
)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
LINEAR_DIR = BASE_DIR
if str(LINEAR_DIR) not in sys.path:
    sys.path.insert(0, str(LINEAR_DIR))

EMPIRICAL_TRACE = DATA_DIR / "figure10_large_scale_p_best_corrdiff_region.npz"
EMPIRICAL_MOS5_TRACE = DATA_DIR / "figure_supply_15_rsp_rmos5_trace.npz"
LAYER_DATA = DATA_DIR / "figure13" / "layer_asymmetric_epsilon_linear_data.npz"
LAYER_TRACE_CACHE = DATA_DIR / "figure_supply_15_layer_trace_cache.npz"

OUT_PNG = ROOT_DIR / "output" / "png" / "figure_supply_15.png"
OUT_PDF = ROOT_DIR / "output" / "pdf" / "figure_supply_15.pdf"

EPSILON_CASES = [0.0, 0.3, 1.0]
EPSILON_PLOT_ORDER = [1.0, 0.3, 0.0]
LAYER_TRACE_REALIZATION = 1
LAYER_N_STEPS_SCALE = 10
LAYER_TRACE_START_STEP = 10000
LAYER_TRACE_WINDOW_STEPS = 2500
EMPIRICAL_WINDOW_STEPS = 900
EMPIRICAL_CASES = [
    ("base", "Base"),
    ("null_in", "Null-In"),
    ("null_out", "Null-Out"),
]

RAW_SUBJECT_FILE = DATA_DIR / "raw_not_bundled" / "subject_13_data_cellular_synapse_sc_100_data.pkl"
BASE_SIM_FILE = DATA_DIR / "raw_not_bundled" / "tsubject_13_iter_0_final_fit_cell_by_cell_sim_ca_data_spike_data.npz"
NULL_IN_SIM_FILE = DATA_DIR / "raw_not_bundled" / "subject_13_iter_1_fit_cell_by_cell_null_p_in_result_mean_ca_data_spike_data.npz"
NULL_OUT_SIM_FILE = DATA_DIR / "raw_not_bundled" / "subject_13_iter_2_fit_cell_by_cell_null_p_out_result_mean_ca_data_spike_data_t1.npz"
DT_MS = 800

LREGION = [
    "MON", "Cb", "MOS1", "MOS2", "MOS3", "MOS4", "MOS5", "IPN", "IO", "Hc", "Ra", "T",
    "aRF", "imRF", "pRF", "GG", "Hb", "Hi", "HR", "OG", "OB", "OE", "P", "Pi", "PT",
    "PO", "PrT", "R", "SP", "TeO", "Th", "TL", "TS", "TG", "VR", "NX",
]
RREGION = [
    "rMON", "rCb", "rMOS1", "rMOS2", "rMOS3", "rMOS4", "rMOS5", "rIPN", "rIO", "rHc",
    "rRa", "rT", "raRF", "rimRF", "rpRF", "rGG", "rHb", "rHi", "rHR", "rOG", "rOB",
    "rOE", "rP", "rPi", "rPT", "rPO", "rPrT", "rR", "rSP", "rTeO", "rTh", "rTL",
    "rTS", "rTG", "rVR", "rNX",
]
REGIONS = LREGION + RREGION
EMPIRICAL_SOURCE_REGION = "rSP"
EMPIRICAL_TARGET_REGION = "rMOS5"


def _load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


layer_plot = _load_module(
    LINEAR_DIR / "plot_layer_asymmetric_epsilon_linear_model.py",
    "figure_supply_15_layer_plot",
)
layer_analysis = _load_module(
    LINEAR_DIR / "analyze_layer_asymmetric_epsilon_linear_model.py",
    "figure_supply_15_layer_analysis",
)


def zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return values - mean
    return (values - mean) / std


def smooth(values, window=31):
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size < 3:
        return values
    if window > values.size:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def subtract_fitted_slow_sine(values, time, frequency=0.1):
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float)
    if values.size < 3 or time.size != values.size:
        return values - np.nanmean(values)

    centered_time = time - time[0]
    omega_time = 2.0 * np.pi * frequency * centered_time
    design = np.column_stack(
        [
            np.sin(omega_time),
            np.cos(omega_time),
            np.ones_like(centered_time),
        ]
    )
    valid = np.isfinite(values) & np.all(np.isfinite(design), axis=1)
    if np.count_nonzero(valid) < design.shape[1]:
        return values - np.nanmean(values)

    coeffs, *_ = np.linalg.lstsq(design[valid], values[valid], rcond=None)
    fitted_slow = design @ coeffs
    return values - fitted_slow


def style_trace_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_empirical_trace():
    if not EMPIRICAL_MOS5_TRACE.exists():
        raise FileNotFoundError(
            f"Missing bundled compact trace file: {EMPIRICAL_MOS5_TRACE}. "
            "The public package uses this precomputed trace instead of the large raw simulation files."
        )
    return np.load(EMPIRICAL_MOS5_TRACE, allow_pickle=True)


def load_calcium(path):
    data = np.load(path, allow_pickle=True)
    if "sim_ca_data" in data.files:
        return np.asarray(data["sim_ca_data"], dtype=float)
    return np.asarray(data["arr_0"], dtype=float)


def pick_largest_cluster(root_area, final_id_cluster, region_idx):
    cluster_ids = np.where(root_area == region_idx)[0]
    if cluster_ids.size == 0:
        return None
    sizes = np.asarray([len(final_id_cluster[idx]) for idx in cluster_ids], dtype=int)
    return int(cluster_ids[np.argmax(sizes)])


def corr_z(a_values, b_values):
    a_values = zscore(a_values)
    b_values = zscore(b_values)
    return float(np.corrcoef(a_values, b_values)[0, 1])


def save_empirical_region_trace(output_path):
    with open(RAW_SUBJECT_FILE, "rb") as f:
        raw = pickle.load(f)

    root_area = np.asarray(raw["root_area"], dtype=int)
    final_id_cluster = raw["final_id_cluster"]
    source_idx = REGIONS.index(EMPIRICAL_SOURCE_REGION)
    target_idx = REGIONS.index(EMPIRICAL_TARGET_REGION)

    source_cluster_id = pick_largest_cluster(root_area, final_id_cluster, source_idx)
    target_cluster_id = pick_largest_cluster(root_area, final_id_cluster, target_idx)
    if source_cluster_id is None:
        raise ValueError(f"Could not find cluster for {EMPIRICAL_SOURCE_REGION}")
    if target_cluster_id is None:
        raise ValueError(f"Could not find cluster for {EMPIRICAL_TARGET_REGION}")

    base_ca = load_calcium(BASE_SIM_FILE)
    null_in_ca = load_calcium(NULL_IN_SIM_FILE)
    null_out_ca = load_calcium(NULL_OUT_SIM_FILE)
    t_axis = np.arange(base_ca.shape[1], dtype=float) * DT_MS

    source = {
        "base": base_ca[source_cluster_id],
        "null_in": null_in_ca[source_cluster_id],
        "null_out": null_out_ca[source_cluster_id],
    }
    target = {
        "base": base_ca[target_cluster_id],
        "null_in": null_in_ca[target_cluster_id],
        "null_out": null_out_ca[target_cluster_id],
    }
    corr_values = {key: corr_z(source[key], target[key]) for key, _ in EMPIRICAL_CASES}

    np.savez(
        output_path,
        t_axis=t_axis,
        p_region_name=np.asarray(EMPIRICAL_SOURCE_REGION),
        p_region_idx=np.asarray(source_idx, dtype=int),
        p_cluster_id=np.asarray(source_cluster_id, dtype=int),
        selected_region_name=np.asarray(EMPIRICAL_TARGET_REGION),
        selected_region_idx=np.asarray(target_idx, dtype=int),
        selected_cluster_id=np.asarray(target_cluster_id, dtype=int),
        selected_cluster_size=np.asarray(len(final_id_cluster[target_cluster_id]), dtype=int),
        corr_base=np.asarray(corr_values["base"], dtype=float),
        corr_null_in=np.asarray(corr_values["null_in"], dtype=float),
        corr_null_out=np.asarray(corr_values["null_out"], dtype=float),
        corr_range=np.asarray(max(corr_values.values()) - min(corr_values.values()), dtype=float),
        base_p=np.asarray(source["base"], dtype=float),
        null_in_p=np.asarray(source["null_in"], dtype=float),
        null_out_p=np.asarray(source["null_out"], dtype=float),
        base_selected=np.asarray(target["base"], dtype=float),
        null_in_selected=np.asarray(target["null_in"], dtype=float),
        null_out_selected=np.asarray(target["null_out"], dtype=float),
    )


def select_empirical_window(data, window_steps=EMPIRICAL_WINDOW_STEPS):
    n_time = len(data["t_axis"])
    start_min = min(300, max(0, n_time - window_steps - 1))
    stop_max = max(start_min + window_steps, n_time - 10)
    best_score = -np.inf
    best_start = start_min

    for start in range(start_min, max(start_min + 1, stop_max - window_steps), 25):
        stop = start + window_steps
        score = 0.0
        for key, _ in EMPIRICAL_CASES:
            for suffix in ("p", "selected"):
                trace = smooth(np.asarray(data[f"{key}_{suffix}"][start:stop], dtype=float))
                score += float(np.nanstd(trace))
        if score > best_score:
            best_score = score
            best_start = start

    best_stop = min(best_start + window_steps, n_time)
    return best_start, best_stop


def plot_empirical_trace(ax, data, key, title, window, show_legend=False):
    t_axis = np.asarray(data["t_axis"], dtype=float)
    y_p = zscore(smooth(data[f"{key}_p"]))
    y_selected = zscore(smooth(data[f"{key}_selected"]))

    start, end = window
    end = min(end, len(t_axis), len(y_p), len(y_selected))
    t_axis = t_axis[start:end]
    t_axis = t_axis - t_axis[0]
    y_p = y_p[start:end]
    y_selected = y_selected[start:end]

    ax.plot(t_axis, y_p, color="#c0392b", lw=0.45, label="SP")
    ax.plot(t_axis, y_selected, color="#2f5597", lw=0.45, label=str(data["selected_region_name"]))
    ax.set_title(title, pad=3)
    ax.set_xlim(t_axis[0], t_axis[-1])

    ymin = min(np.nanmin(y_p), np.nanmin(y_selected))
    ymax = max(np.nanmax(y_p), np.nanmax(y_selected))
    pad = 0.08 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=fig_help.TICK_FS_2COL - 2,
            handlelength=1.5,
            borderaxespad=0.1,
        )
    style_trace_axis(ax)


def simulate_layer_traces():
    _, args = layer_plot.load_results(LAYER_DATA)
    n_steps = int(args.n_steps * LAYER_N_STEPS_SCALE)
    seed_sequence = np.random.SeedSequence([int(args.seed), int(LAYER_TRACE_REALIZATION)])
    seeds = seed_sequence.generate_state(len(EPSILON_CASES), dtype=np.uint32)

    cache = {
        "epsilon": np.asarray(EPSILON_CASES, dtype=float),
        "time": None,
        "l1": [],
        "l4": [],
    }
    for epsilon, seed in zip(EPSILON_CASES, seeds):
        time, _, signals, _, slices = layer_analysis.simulate_layer_network(
            epsilon=epsilon,
            layer_sizes=args.layer_sizes,
            n_steps=n_steps,
            dt=args.dt,
            gamma=args.gamma,
            w_intra=args.w_intra,
            w_inter=args.w_inter,
            intra_epsilon=args.intra_epsilon,
            inter_epsilon_scales=args.inter_epsilon_scales,
            layer_decay_offsets=args.layer_decay_offsets,
            noise_sigma=args.noise_sigma,
            epsilon_slow_layer_drive=args.epsilon_slow_layer_drive,
            slow_drive_frequency=args.slow_drive_frequency,
            slow_layer_scales=args.slow_layer_scales,
            input_amplitude=args.input_amplitude,
            input_frequency=args.input_frequency,
            seed=int(seed),
        )
        if cache["time"] is None:
            cache["time"] = time
        cache["l1"].append(np.mean(signals[:, slices[0]], axis=1))
        cache["l4"].append(np.mean(signals[:, slices[-1]], axis=1))

    np.savez(
        LAYER_TRACE_CACHE,
        epsilon=cache["epsilon"],
        time=np.asarray(cache["time"], dtype=float),
        l1=np.asarray(cache["l1"], dtype=float),
        l4=np.asarray(cache["l4"], dtype=float),
        n_steps=np.asarray(n_steps, dtype=int),
        realization=np.asarray(LAYER_TRACE_REALIZATION, dtype=int),
    )


def load_layer_traces():
    if not LAYER_TRACE_CACHE.exists():
        simulate_layer_traces()
    data = np.load(LAYER_TRACE_CACHE)
    epsilon = np.asarray(data["epsilon"], dtype=float)
    expected_steps = int(layer_plot.load_results(LAYER_DATA)[1].n_steps * LAYER_N_STEPS_SCALE)
    cached_steps = int(data["n_steps"]) if "n_steps" in data else int(data["time"].shape[0])
    cached_realization = int(data["realization"]) if "realization" in data else -1
    if (
        epsilon.shape[0] != len(EPSILON_CASES)
        or not np.allclose(epsilon, EPSILON_CASES)
        or cached_steps != expected_steps
        or cached_realization != LAYER_TRACE_REALIZATION
    ):
        simulate_layer_traces()
        data = np.load(LAYER_TRACE_CACHE)
    return data


def plot_layer_trace(ax, trace_data, idx, title, show_legend=False):
    time = np.asarray(trace_data["time"], dtype=float)
    l1 = zscore(subtract_fitted_slow_sine(trace_data["l1"][idx], time))
    l4 = zscore(subtract_fitted_slow_sine(trace_data["l4"][idx], time))

    start = min(LAYER_TRACE_START_STEP, max(0, len(time) - LAYER_TRACE_WINDOW_STEPS))
    end = min(start + LAYER_TRACE_WINDOW_STEPS, len(time), len(l1), len(l4))
    time = time[start:end] - time[start]
    l1 = l1[start:end]
    l4 = l4[start:end]

    ax.plot(time, l1, color="#c0392b", lw=0.45, label="L1")
    ax.plot(time, l4, color="#2f5597", lw=0.45, label="L4")
    ax.set_title(title, pad=3)
    ax.set_xlim(time[0], time[-1])
    ymin = min(np.nanmin(l1), np.nanmin(l4))
    ymax = max(np.nanmax(l1), np.nanmax(l4))
    pad = 0.08 * (ymax - ymin + 1e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    if show_legend:
        ax.legend(
            loc="upper right",
            frameon=False,
            fontsize=fig_help.TICK_FS_2COL - 2,
            handlelength=1.5,
            borderaxespad=0.1,
        )
    style_trace_axis(ax)
    return float(np.nanmin([np.nanmin(l1), np.nanmin(l4)])), float(
        np.nanmax([np.nanmax(l1), np.nanmax(l4)])
    )


def add_panel_label(fig, ax, label):
    fig_help.add_panel_label_fig(
        fig,
        ax,
        label,
        dx=-0.03,
        dy=0.01,
        fontsize=fig_help.PANEL_LABEL_FS_2COL,
    )


def main():
    empirical = load_empirical_trace()
    empirical_window = select_empirical_window(empirical)
    layer_traces = load_layer_traces()

    fig = plt.figure(figsize=(8.0, 4.4))
    gs = GridSpec(
        2,
        3,
        figure=fig,
        left=0.055,
        right=0.985,
        top=0.90,
        bottom=0.08,
        hspace=0.48,
        wspace=0.12,
    )

    empirical_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    layer_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]

    for idx, (key, title) in enumerate(EMPIRICAL_CASES):
        plot_empirical_trace(
            empirical_axes[idx],
            empirical,
            key,
            title,
            empirical_window,
            show_legend=(idx == 0),
        )

    epsilon_to_trace_idx = {
        float(epsilon): idx for idx, epsilon in enumerate(np.asarray(layer_traces["epsilon"], dtype=float))
    }
    layer_ylim = []
    for plot_idx, epsilon in enumerate(EPSILON_PLOT_ORDER):
        layer_ylim.append(
            plot_layer_trace(
                layer_axes[plot_idx],
                layer_traces,
                epsilon_to_trace_idx[float(epsilon)],
                f"epsilon = {epsilon:g}",
                show_legend=(plot_idx == 0),
            )
        )
    layer_ymin = min(ymin for ymin, _ in layer_ylim)
    layer_ymax = max(ymax for _, ymax in layer_ylim)
    layer_pad = 0.08 * (layer_ymax - layer_ymin + 1e-6)
    for ax in layer_axes:
        ax.set_ylim(layer_ymin - layer_pad, layer_ymax + layer_pad)

    fig.text(
        0.52,
        0.955,
        "Empirical P/SP-associated traces under null rewiring",
        ha="center",
        va="center",
        fontsize=fig_help.AXIS_LABEL_FS_2COL,
        fontweight="bold",
    )
    fig.text(
        0.52,
        0.475,
        "Layer-linear model traces under increasing directional asymmetry",
        ha="center",
        va="center",
        fontsize=fig_help.AXIS_LABEL_FS_2COL,
        fontweight="bold",
    )
    fig.text(0.055, 0.515, "Time", ha="left", va="center", fontsize=fig_help.TICK_FS_2COL)
    fig.text(0.055, 0.035, "Time", ha="left", va="center", fontsize=fig_help.TICK_FS_2COL)

    add_panel_label(fig, empirical_axes[0], "A")
    add_panel_label(fig, layer_axes[0], "B")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {LAYER_TRACE_CACHE}")


if __name__ == "__main__":
    main()
