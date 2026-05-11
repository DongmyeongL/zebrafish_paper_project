import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import itertools

_tel  = {'lOB','lSP','lP','lPO','rOB','rSP','rP','rPO'}
_di   = {'lHb','lTh','lPT','lPrT','rHb','rTh','rPT','rPrT'}
_mes  = {'lTeO','lTL','lTS','rTeO','rTL','rTS'}
_hind = {'lCb','lT','laRF','lMOS1','lMOS2','lMOS3','lMOS4','lMOS5',
         'limRF','lpRF','lMON','lNX','lRa',
         'rCb','rT','raRF','rMOS1','rMOS2','rMOS3','rMOS4','rMOS5',
         'rimRF','rpRF','rMON','rNX','rRa'}
div_order_map = {'Tel': 0, 'Di': 1, 'Mes': 2, 'Hind': 3}

def _region_to_div(name):
    if name in _tel:  return 'Tel'
    if name in _di:   return 'Di'
    if name in _mes:  return 'Mes'
    if name in _hind: return 'Hind'
    return None


lregion = ['lMON','lCb','lMOS1','lMOS2','lMOS3','lMOS4','lMOS5','lIPN','lIO','lHc','lRa','lT',
           'laRF','limRF','lpRF','lGG','lHb','lHi','lHR','lOG','lOB','lOE','lP','lPi','lPT',
           'lPO','lPrT','lR','lSP','lTeO','lTh','lTL','lTS','lTG','lVR','lNX']

rregion = ['rMON','rCb','rMOS1','rMOS2','rMOS3','rMOS4','rMOS5','rIPN','rIO','rHc',
           'rRa','rT','raRF','rimRF','rpRF','rGG','rHb','rHi','rHR','rOG','rOB',
           'rOE','rP','rPi','rPT','rPO','rPrT','rR','rSP','rTeO','rTh','rTL',
           'rTS','rTG','rVR','rNX']

region = lregion + rregion
REGION_INDEX = {name: idx for idx, name in enumerate(region)}
SUBJECT_IDS = range(12, 19)
METASTABILITY_FILE = "fc_dynamics_metastability_by_subject_region.csv"
DIVISION_COLUMNS = ['Tel', 'Di', 'Mes', 'Hind']

import figure_style as fs
fs.set_paper_style()
plt.rcParams.update({
    'font.size':       fs.AXIS_LABEL_FS_2COL,
    'axes.labelsize':  fs.AXIS_LABEL_FS_2COL,
    'axes.titlesize':  fs.AXIS_LABEL_FS_2COL,
    'xtick.labelsize': fs.TICK_FS_2COL,
    'ytick.labelsize': fs.TICK_FS_2COL,
})
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PNG = os.path.join(PROJECT_ROOT, "output", "png", "figure9_final.png")
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "output", "pdf", "figure9_final.pdf")
STATS_DIR = os.path.join(PROJECT_ROOT, "output", "stats")
STATS_CSV = os.path.join(STATS_DIR, "figure9_stats.csv")
BASE_NET = os.path.join(DATA, "region_community_io")
STATS_ROWS = []

def _zscore(values):
    values = np.asarray(values, dtype=float)
    return (values - np.nanmean(values)) / np.nanstd(values)

def _load_subject_causality(subject_id):
    return np.load(f"{BASE_NET}/subject_{subject_id}/subject_{subject_id}_causality.npz")

def _load_subject_fc_neighbors(subject_id):
    return np.load(
        f"{BASE_NET}/subject_{subject_id}/subject_{subject_id}_net_te_drive_fc_neighbors.npz"
    )

def _aggregate_region_and_division(values_by_subject, region_orders, n_regions_total):
    region_values = [[] for _ in region_orders]
    division_values = [[] for _ in range(4)]
    order_lookup = {region_idx: pos for pos, region_idx in enumerate(region_orders)}

    for region_num, values in values_by_subject:
        per_region = [[] for _ in range(n_regions_total)]
        for reg_idx, value in zip(region_num, values):
            per_region[reg_idx].append(value)
            ordered_pos = order_lookup.get(reg_idx)
            if ordered_pos is not None:
                region_values[ordered_pos].append(value)

        for reg_idx, samples in enumerate(per_region):
            division_idx = fs.brain_division_list[reg_idx]
            if samples and division_idx < 4:
                division_values[division_idx].append(float(np.mean(samples)))

    region_means = np.array([
        np.mean(samples) if samples else 0.0
        for samples in region_values
    ], dtype=float)
    return region_means, division_values

#fcv-fcs
bar_df = pd.read_csv(f'{DATA}/fig1_prism_D_FCS_FCV_bar.csv')

bar_df['_div']       = bar_df['Region'].apply(_region_to_div)
bar_df['_div_order'] = bar_df['_div'].map(div_order_map)
bar_df = bar_df.sort_values(['_div_order', 'Region']).reset_index(drop=True)

regions_d = bar_df['Region'].values
fcs_vals  = bar_df['FCS'].values
fcv_vals  = bar_df['FCV'].values
divs_d    = bar_df['_div'].values

fcs_vals = _zscore(fcs_vals)
fcv_vals = _zscore(fcv_vals)

new_region_order = [REGION_INDEX[name] for name in regions_d]
n_total_regions = len(region)

# metastability from figure_fc_dynamics_metastability_stimulus_fcv.py
metastability_df = pd.read_csv(os.path.join(DATA, METASTABILITY_FILE))
metastability_region_mean = (
    metastability_df
    .groupby('Region', observed=True)['RegionwiseMetastability']
    .mean()
)
metastability_z = pd.Series(
    _zscore(metastability_region_mean.values),
    index=metastability_region_mean.index,
)
new_metastability_data = np.array([
    metastability_z.get(region_name, 0.0)
    for region_name in regions_d
], dtype=float)

#Net Transfer Entropy
net_te_subject_data = []
for subject_id in SUBJECT_IDS:
    subject_data = _load_subject_causality(subject_id)
    net_te_subject_data.append((
        subject_data['region_num'],
        np.nanmean(subject_data['net_te_matrix'], axis=1),
    ))

net_transfer_data, out_net_te_data = _aggregate_region_and_division(
    net_te_subject_data, new_region_order, n_total_regions
)
net_transfer_data = _zscore(net_transfer_data)


#mean neighbor Net Transfer Entropy
neighbor_subject_data = []
for subject_id in SUBJECT_IDS:
    subject_data = _load_subject_causality(subject_id)
    neighbor_data = _load_subject_fc_neighbors(subject_id)
    neighbor_subject_data.append((
        subject_data['region_num'],
        neighbor_data['fc_neighbor_mean_drive'],
    ))

fc_neighbor_mean_drive_data, out_neigh_net_te_data = _aggregate_region_and_division(
    neighbor_subject_data, new_region_order, n_total_regions
)
fc_neighbor_mean_drive_data = _zscore(fc_neighbor_mean_drive_data)


out_data = np.array([
    np.asarray(fcs_vals).flatten(),
    np.asarray(fcv_vals).flatten(),
    np.asarray(new_metastability_data),
    net_transfer_data,
    fc_neighbor_mean_drive_data,
])

# Step 1: collect all unique region indices across subjects
all_reg_set = set()
for subject_id in SUBJECT_IDS:
    _d = _load_subject_causality(subject_id)
    all_reg_set.update(_d['region_num'].tolist())
net_reg_indices = sorted(all_reg_set)          # e.g. [0, 1, 2, ...]
n_nr            = len(net_reg_indices)
rid2pos         = {r: k for k, r in enumerate(net_reg_indices)}

# Step 2: accumulate region×region net-TE
nte_sum   = np.zeros((n_nr, n_nr))
nte_cnt   = np.zeros((n_nr, n_nr), dtype=int)

for subject_id in SUBJECT_IDS:
    _d   = _load_subject_causality(subject_id)
    _rn  = _d['region_num']
    _nte = _d['net_te_matrix']
    for _rs in np.unique(_rn):
        _sc = np.where(_rn == _rs)[0]
        _ps = rid2pos[_rs]
        for _rd in np.unique(_rn):
            if _rs == _rd: continue
            _dc  = np.where(_rn == _rd)[0]
            _pd  = rid2pos[_rd]
            _sub = _nte[np.ix_(_sc, _dc)]
            _v   = _sub[np.isfinite(_sub)]
            if len(_v):
                nte_sum[_ps, _pd] += float(np.mean(_v))
                nte_cnt[_ps, _pd] += 1

nte_mean = np.where(nte_cnt > 0,
                    nte_sum / np.maximum(nte_cnt, 1),
                    np.nan)
np.fill_diagonal(nte_mean, np.nan)

# Step 3: region metadata
net_reg_names = [region[k] for k in net_reg_indices]
net_reg_divs  = [_region_to_div(n) for n in net_reg_names]

# Step 4: per-region mean net-TE drive
net_reg_drive = np.nanmean(nte_mean, axis=1)   # positive = driver

# ── Division / L-R 색상 맵 ──
div_color_map = {
    'Tel':  fs.division_colors[2],   # '#74c476' green
    'Di':   fs.division_colors[1],   # '#fdae6b' orange
    'Mes':  fs.division_colors[3],   # '#fb6a4a' red
    'Hind': fs.division_colors[0],   # '#6baed6' blue
}

# ── 클러스터링: out_data.T = (n_regions × 5 features) ──
Z = linkage(out_data.T, method='ward')

# 덴드로그램 leaf 순서 추출
dend_info  = dendrogram(Z, no_plot=True)
leaf_order = np.array(dend_info['leaves'])

# 클러스터 경계: 4 major clusters
n_clusters = 3
clust_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
clust_ordered = clust_labels[leaf_order]
cluster_boundaries = [i - 0.5 for i in range(1, len(clust_ordered))
                      if clust_ordered[i] != clust_ordered[i - 1]]

# 클러스터링 순서로 데이터 재정렬
out_data_c = out_data[:, leaf_order]
regions_c  = regions_d[leaf_order]
divs_c     = divs_d[leaf_order]
n_regions = out_data_c.shape[1]
y_labels  = ['zFCS', 'zFCV', 'Metasta-\nbility', 'NetTE', 'Neighbor\nNetTE']

# ── Figure 레이아웃: Left=[A (dendro+divbar+heat)] | Right=[B (network)], [C D E F G] ──
_fig_w   = 16;#fs.TWO_COL_IN
_fig_h   = 9;#fs.TWO_COL_IN * 0.65
fig = plt.figure(figsize=(_fig_w, _fig_h))
gs  = GridSpec(5, 10,
               figure=fig,
               height_ratios=[1.0, 0.2, 2.5, 2.0, 2.0],
               width_ratios=[1]*10,
               left=0.08, right=0.97,
               top=0.94, bottom=0.20,
               hspace=0.20, wspace=0.30)

# ──── LEFT COLUMN: Panel A (Dendrogram, Division bar, Heatmap) ────
# Row 0: Dendrogram (left, cols 0-5)
ax_dendro = fig.add_subplot(gs[0, 0:4])

# Row 1: Division bar (left, cols 0-5)
ax_divbar_a = fig.add_subplot(gs[1, 0:4])

# Row 2: Heatmap (left, cols 0-4), Colorbar (col 4)
ax_heat   = fig.add_subplot(gs[2, 0:4])
ax_cbar   = fig.add_subplot(gs[2, 4:5])

# ──── RIGHT TOP: Panel B (Hierarchical network at top) ────
ax_hier = fig.add_subplot(gs[0:3, 5:10])

# ──── BOTTOM ROW: Panels C, D, E, F, G (각 2 columns, 동일 크기) ────
ax_c = fig.add_subplot(gs[3:5, 0:2])    # Panel C
ax_d = fig.add_subplot(gs[3:5, 2:4])    # Panel D
ax_e = fig.add_subplot(gs[3:5, 4:6])    # Panel E
ax_f = fig.add_subplot(gs[3:5, 6:8])    # Panel F
ax_g = fig.add_subplot(gs[3:5, 8:10])   # Panel G

# ════════════════════════════════════════════════════════════════
# ── Data loading & helpers for Panels C-G ──
# ════════════════════════════════════════════════════════════════

# ── figure2 helper functions ──
_div_colors_f2 = {'Tel': '#74c476', 'Di': '#fdae6b', 'Mes': '#fb6a4a', 'Hind': '#6baed6'}
_div_order_f2  = DIVISION_COLUMNS
_STAR_FS = fs.STAR_FS_2COL

def _record_division_stats(panel, df, order):
    groups = [df[k].dropna().values for k in order]
    pairs  = list(itertools.combinations(range(len(order)), 2))
    pvals  = [mannwhitneyu(groups[i], groups[j], alternative='two-sided')[1]
              for i, j in pairs]
    reject, corr, _, _ = multipletests(pvals, method='holm')
    for idx, (i, j) in enumerate(pairs):
        STATS_ROWS.append({
            'figure': 'figure9',
            'panel': panel,
            'test': 'Mann-Whitney U',
            'alternative': 'two-sided',
            'group_1': order[i],
            'group_2': order[j],
            'n_group_1': len(groups[i]),
            'n_group_2': len(groups[j]),
            'p_uncorrected': pvals[idx],
            'p_holm': corr[idx],
            'reject_holm_0.05': bool(reject[idx]),
        })
    return pairs, reject, corr

def _add_sig_bars(ax, df, order, panel):
    pairs, reject, corr = _record_division_stats(panel, df, order)
    sig = sorted([(pairs[k], corr[k]) for k in range(len(pairs)) if reject[k]],
                 key=lambda x: x[0][1] - x[0][0])
    if not sig:
        return
    y_min, y_max = ax.get_ylim()
    yr    = y_max - y_min
    step  = yr * 0.060
    bar_h = yr * 0.014
    for lvl, ((i, j), p) in enumerate(sig):
        y    = y_max + yr * 0.025 + lvl * step
        star = '***' if p < 0.001 else '**' if p < 0.01 else '*'
        ax.plot(
            [i, i, j, j],
            [y, y + bar_h, y + bar_h, y],
            lw=0.6,
            c='#333',
            clip_on=False,
        )
        ax.text((i + j) / 2, y + bar_h, star,
                ha='center', va='center', fontsize=_STAR_FS,
                clip_on=False)
    ax.set_ylim(y_min, y_max)

def _boxplot_panel(ax, df, ylabel):
    vals, labels = [], []
    for col in _div_order_f2:
        v = df[col].dropna().values
        vals.extend(v); labels.extend([col] * len(v))
    sns.boxplot(x=labels, y=vals, order=_div_order_f2,
                hue=labels, hue_order=_div_order_f2, legend=False,
                palette=[_div_colors_f2[d] for d in _div_order_f2],
                showfliers=False, width=0.45, linewidth=1.0, ax=ax)
    sns.stripplot(x=labels, y=vals, order=_div_order_f2,
                  color='black', size=3, alpha=0.2, jitter=True, ax=ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.tick_params(axis='both', which='both', direction='out',
                   bottom=True, left=True, length=4, width=1.2)
    _add_sig_bars(ax, {c: df[c] for c in _div_order_f2}, _div_order_f2, ylabel)

def _division_lists_to_df(division_data):
    max_len = max((len(v) for v in division_data), default=0)
    return pd.DataFrame({
        div: pd.Series(vals, dtype=float).reindex(range(max_len))
        for div, vals in zip(['Hind', 'Di', 'Tel', 'Mes'], division_data)
    })

def _metastability_division_df(meta_df):
    plot_df = meta_df[['Division', 'RegionwiseMetastability']].dropna().copy()
    grouped = {
        div: plot_df.loc[plot_df['Division'] == div, 'RegionwiseMetastability'].to_numpy()
        for div in _div_order_f2
    }
    max_len = max((len(v) for v in grouped.values()), default=0)
    return pd.DataFrame({
        div: pd.Series(vals, dtype=float).reindex(range(max_len))
        for div, vals in grouped.items()
    })

# ── figure2 data (Panels C, D, E) ──
_fcs = pd.read_csv(f'{DATA}/fig2_prism_A_FCS.csv')
_fcv = pd.read_csv(f'{DATA}/fig2_prism_B_FCV.csv')
_metastability = _metastability_division_df(metastability_df)

out_net_te_data_df = _division_lists_to_df(out_net_te_data)
out_neigh_net_te_data_df = _division_lists_to_df(out_neigh_net_te_data)

# ════════════════════════════════════════════════════════════════
# ── Panels C, D, E: figure2 Panel A, B, C ──
# ════════════════════════════════════════════════════════════════
_boxplot_panel(ax_c, _fcs, 'zFCS ')
_boxplot_panel(ax_d, _fcv, 'zFCV ')
_boxplot_panel(ax_e, _metastability, 'Metastability')
_boxplot_panel(ax_f, out_net_te_data_df, 'NetTE')
ax_f.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value * 1e2:g}'))
ax_f.text(
    -0.22, 1.02, r'$\times 10^{-2}$',
    transform=ax_f.transAxes,
    fontsize=fs.TICK_FS_2COL,
    ha='left',
    va='bottom',
)
_boxplot_panel(ax_g, out_neigh_net_te_data_df, 'Neighbor NetTE')
ax_g.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value * 1e3:g}'))
ax_g.text(
    -0.22, 1.02, r'$\times 10^{-3}$',
    transform=ax_g.transAxes,
    fontsize=fs.TICK_FS_2COL,
    ha='left',
    va='bottom',
)

# 패널 레터 A
ax_dendro.text(-0.05, 1.5, 'A', transform=ax_dendro.transAxes,
               fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom', ha='right')
ax_c.text(-0.31, 1.05, 'C', transform=ax_c.transAxes, fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom')
ax_d.text(-0.31, 1.05, 'D', transform=ax_d.transAxes, fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom')
ax_e.text(-0.31, 1.05, 'E', transform=ax_e.transAxes, fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom')
ax_f.text(-0.31, 1.05, 'F', transform=ax_f.transAxes, fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom')
ax_g.text(-0.31, 1.05, 'G', transform=ax_g.transAxes, fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom')
ax_hier.text(-0.02, 0.965, 'B', transform=ax_hier.transAxes,
               fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='bottom', ha='right')



# ── 1. Dendrogram ──
dendrogram(Z, ax=ax_dendro, no_labels=True,
           color_threshold=0, above_threshold_color='#333333')
ax_dendro.set_xlim(0, n_regions * 10)
# 상단 여백 제거: 실제 최대 높이 + 5%만 표시
dend_max_h = max(max(d) for d in dend_info['dcoord'])
ax_dendro.set_ylim(0, dend_max_h * 1.05)
ax_dendro.axis('off')



# ── 2. Division bar (full height) ──
for i, div in enumerate(divs_c):
    ax_divbar_a.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1,
                        color=div_color_map.get(div, 'gray'), linewidth=0))

ax_divbar_a.set_xlim(-0.5, n_regions - 0.5)
ax_divbar_a.set_ylim(0, 1)
ax_divbar_a.set_xticks([])
ax_divbar_a.set_yticks([])
for spine in ax_divbar_a.spines.values():
    spine.set_visible(False)

# 클러스터 경계선 (annotation bar)
for bnd in cluster_boundaries:
    ax_divbar_a.axvline(x=bnd, color='black', linewidth=1.5, zorder=5)

# ── 3. Heatmap ──
im = ax_heat.imshow(out_data_c, aspect='auto', cmap='PiYG_r',
                    vmax=2, vmin=-2, interpolation='nearest')
ax_heat.set_xlim(-0.5, n_regions - 0.5)

ax_heat.set_yticks(range(len(y_labels)))
ax_heat.set_yticklabels(y_labels, fontsize=fs.TICK_FS_2COL, fontweight='bold')
ax_heat.tick_params(axis='y', length=0, labelright=True, labelleft=False)

ax_heat.set_xticks(range(n_regions))
ax_heat.set_xticklabels(regions_c, rotation=90, fontsize=fs.TICK_FS_2COL - 2,
                         ha='center', fontweight='bold')
ax_heat.tick_params(axis='x', length=2, width=0.5)

# x-tick 레이블 색: division 색
for tick_label, div in zip(ax_heat.get_xticklabels(), divs_c):
    tick_label.set_color(div_color_map[div])

# 행 구분선
for y in np.arange(-0.5, len(y_labels), 1):
    ax_heat.axhline(y=y, color='white', linewidth=0.4)

# 클러스터 경계선 (heatmap)
for bnd in cluster_boundaries:
    ax_heat.axvline(x=bnd, color='black', linewidth=1.5, zorder=5)

# 외곽 테두리
for spine in ax_heat.spines.values():
    spine.set_linewidth(1.4)
    spine.set_color('black')
ax_heat.add_patch(
    mpatches.Rectangle(
        (-0.5, -0.5), n_regions, len(y_labels),
        fill=False, edgecolor='black', linewidth=1.4,
        zorder=10, clip_on=False
    )
)

# ── 4. Colorbar ──
ax_cbar.set_position([0.48, 0.585, 0.020, 0.205])
                     
cbar = plt.colorbar(im, cax=ax_cbar)
cbar.ax.set_title('Z-score', fontsize=fs.AXIS_LABEL_FS_2COL, fontweight='bold')
cbar.ax.tick_params(labelsize=fs.TICK_FS_2COL)
cbar.set_ticks([-2, -1, 0, 1, 2])


# ── 5. 통합 범례 (heatmap 하단 바깥) ──
div_handles = [mpatches.Patch(color=div_color_map[d], label=d)
               for d in ['Tel', 'Di', 'Mes', 'Hind']]
leg = ax_heat.legend(handles=div_handles,
                     loc='upper left',
                     bbox_to_anchor=(0.35, 1.95),
                     ncol=4, fontsize=fs.TICK_FS_2COL,
                     frameon=False, framealpha=0.95,
                     title_fontsize=fs.AXIS_LABEL_FS_2COL)
leg.get_title().set_fontweight('bold')
for text in leg.get_texts():
    text.set_fontweight('bold')

# ── Rank-based normalization for Panel B node size/color ──
from scipy.stats import rankdata as _rankdata
div_order_net = ['Tel', 'Di', 'Mes', 'Hind']
_valid_ks_h  = [k for k in range(n_nr) if net_reg_divs[k] is not None]
_fcv_by_region = dict(zip(regions_d, fcv_vals))
_fcv_arr     = np.array([
    _fcv_by_region.get(net_reg_names[_k], np.nan)
    for _k in _valid_ks_h
], dtype=float)
_nan_fill    = np.nanmin(_fcv_arr) if np.any(np.isfinite(_fcv_arr)) else 0.0
_fcv_arr     = np.where(np.isfinite(_fcv_arr), _fcv_arr, _nan_fill)
_ranks       = _rankdata(_fcv_arr)
_norm_arr    = (_ranks - 1) / max(len(_ranks) - 1, 1)
_k_to_norm   = {_k: float(_norm_arr[_i]) for _i, _k in enumerate(_valid_ks_h)}

_drives_arr  = np.array([net_reg_drive[_k] if np.isfinite(net_reg_drive[_k]) else np.nan
                         for _k in _valid_ks_h])
_drive_fill  = np.nanmin(_drives_arr) if np.any(np.isfinite(_drives_arr)) else 0.0
_drives_arr  = np.where(np.isfinite(_drives_arr), _drives_arr, _drive_fill)
_drive_ranks = _rankdata(_drives_arr)
_drive_norm_arr = (_drive_ranks - 1) / max(len(_drive_ranks) - 1, 1)
_k_to_drive_norm = {_k: float(_drive_norm_arr[_i]) for _i, _k in enumerate(_valid_ks_h)}
_white = np.array([1.0, 1.0, 1.0])

# ── 6. Hierarchical TE Network (Panel B) ──
# 7a. Load fc_neighbor_mask_fdr (BH-FDR q<0.001) from region_community_io
#     Region-level FC edge = significant in >= 4/7 subjects (majority rule)
fc_sig_cnt_h = np.zeros((n_nr, n_nr), dtype=int)

for _i in range(7):
    _d      = np.load(f"{BASE_NET}/subject_{_i+12}/subject_{_i+12}_causality.npz")
    _rn     = _d['region_num']
    _fc_sig = _d['fc_neighbor_mask_fdr']   # (148,148) bool, BH-FDR significant FC
    for _rs in np.unique(_rn):
        if _rs not in rid2pos: continue
        _sc = np.where(_rn == _rs)[0];  _ps = rid2pos[_rs]
        for _rd in np.unique(_rn):
            if _rs == _rd or _rd not in rid2pos: continue
            _dc = np.where(_rn == _rd)[0]; _pd = rid2pos[_rd]
            if np.any(_fc_sig[np.ix_(_sc, _dc)]):   # any community pair significant
                fc_sig_cnt_h[_ps, _pd] += 1

# FC edge exists if significant in >= 4/7 subjects
fc_edge_mask_h = fc_sig_cnt_h >= 4

# 7b. Edge list: FC mask as existence filter, positive net TE as weight
#     Show top 50% of all finite nte_mean values
_te_thresh_h = float(np.percentile(nte_mean[np.isfinite(nte_mean)], 65))

hier_edges = []
for _s in range(n_nr):
    for _t in range(n_nr):
        if _s == _t: continue
        if net_reg_divs[_s] is None or net_reg_divs[_t] is None: continue
        if not fc_edge_mask_h[_s, _t]: continue          # must have significant FC
        _te_v = nte_mean[_s, _t]
        if not (np.isfinite(_te_v) and _te_v >= _te_thresh_h): continue
        hier_edges.append((_s, _t, float(_te_v)))

hier_edges.sort(key=lambda x: x[2])   # weakest first

# 7c. Node positions: each division = one row
div_y_hier = {'Tel': 3, 'Di': 2, 'Mes': 1, 'Hind': 0}  
hier_pos   = {}

for _div in div_order_net:
    _idxs = [k for k, dv in enumerate(net_reg_divs) if dv == _div]
    _n    = len(_idxs)
    for _j, _k in enumerate(_idxs):
        _x = (_j / max(_n - 1, 1)) * 2.0 - 1.0   # -1 … +1
        hier_pos[_k] = (_x, float(div_y_hier[_div]))

# 7d. Draw panel (already created above as ax_hier)
# Faint horizontal band per division
#_band_colors = {d: mcolors.to_rgba(div_color_map[d], alpha=0.08) for d in div_order_net}
#for _div, _y in div_y_hier.items():
#    ax_hier.axhspan(_y - 0.45, _y + 0.45, color=_band_colors[_div], zorder=0)

# Edges
# Cascade edge: both source AND target are strong NetTE drivers.
_cascade_thresh = 0.5

if hier_edges:
    _te_min_h = hier_edges[0][2]
    _te_max_h = hier_edges[-1][2]
    _te_rng_h = max(_te_max_h - _te_min_h, 1e-9)

    # Draw non-cascade edges first (background), then cascade edges on top
    for _pass in ['background', 'cascade']:
        for _s, _t, _w in hier_edges:
            _s_norm = _k_to_drive_norm.get(_s, 0)
            _t_norm = _k_to_drive_norm.get(_t, 0)
            _is_cascade = (_s_norm >= _cascade_thresh) and (_t_norm >= _cascade_thresh)

            if _pass == 'background' and _is_cascade: continue
            if _pass == 'cascade'    and not _is_cascade: continue

            _x0, _y0 = hier_pos[_s]
            _x1, _y1 = hier_pos[_t]
            _norm_e   = (_w - _te_min_h) / _te_rng_h
            _div_s    = net_reg_divs[_s]

            _same_layer = (abs(_y1 - _y0) < 0.1)
            _rad = 0.15 if _same_layer else 0.04

            if _is_cascade:
                # Cascade path: source division color, thick, opaque
                _ec     = div_color_map.get(_div_s, '#aaaaaa')
                _alpha  = 0.70 + 0.30 * _norm_e
                _lw     = 1.4  + 3.2  * _norm_e
                _ms     = 16
                _zorder = 3
                
                ax_hier.annotate(
                '', xy=(_x1, _y1), xytext=(_x0, _y0),
                arrowprops=dict(arrowstyle='-|>', color=_ec, lw=_lw, alpha=_alpha,
                                mutation_scale=_ms,
                                shrinkA=8, shrinkB=8,
                                connectionstyle=f'arc3,rad={_rad}'),
                zorder=_zorder)
            '''
            else:
                # Non-cascade: faint gray background
                _ec     = '#bbbbbb'
                _alpha  = 0.08 + 0.14 * _norm_e
                _lw     = 0.3  + 0.6  * _norm_e
                _ms     = 9
                _zorder = 1
            '''
    

# Nodes: size/color intensity are proportional to regional FCV rank.
for _k in range(n_nr):
    if _k not in hier_pos: continue
    _x, _y    = hier_pos[_k]
    _div      = net_reg_divs[_k]
    _base_rgb = np.array(mcolors.to_rgb(div_color_map.get(_div, '#999999')))
    _norm_n   = _k_to_norm[_k]
    _size     = 15 + 200 * _norm_n
    _intensity = 0.10 + 0.90 * _norm_n
    _color    = np.clip(_white * (1 - _intensity) + _base_rgb * _intensity, 0, 1)
    ax_hier.scatter(_x, _y, s=_size, c=[_color],
                    edgecolors='black', linewidths=0.4, zorder=4)

# Region labels (rotated below each node)
for _k in range(n_nr):
    if _k not in hier_pos: continue
    _x, _y = hier_pos[_k]
    _div   = net_reg_divs[_k]
    ax_hier.text(_x, _y - 0.18, net_reg_names[_k],
                 fontsize=fs.TICK_FS_2COL*0.9 , ha='center', va='top', rotation=90,
                 color=div_color_map.get(_div, '#333333'), fontweight='bold', zorder=5,
                 path_effects=[pe.withStroke(linewidth=3.0, foreground='white')])

# Division labels (left margin)
for _div, _y in div_y_hier.items():
    ax_hier.text(-1.12, _y, _div, fontsize=fs.TICK_FS_2COL*1.2, ha='right', va='center',
                 color=div_color_map[_div], fontweight='bold')

# Thin dividers between layers
#for _y in [0.5, 1.5, 2.5]:
#    ax_hier.axhline(_y, color='#cccccc', lw=0.6, zorder=1)

ax_hier.set_xlim(-1.18, 1.08)
ax_hier.set_ylim(-1.20, 3.80)

pos = ax_hier.get_position()
ax_hier.set_position([pos.x0, pos.y0-0.1, pos.width , pos.height*1.45])

ax_hier.set_aspect('auto')
ax_hier.axis('off')
#ax_hier.text(-1.18, 3.65, 'B', fontsize=fs.PANEL_LABEL_FS_2COL, fontweight='bold', va='top')
#ax_hier.text(0, -0.98,
#             'Hierarchical TE Network  (colored edges: output→output cascade paths,  node size/color ∝ Net TE drive)',
#             fontsize=fs.STAT_FS_2COL, ha='center', va='top', color='#444444', fontweight='bold')



_cb_pos = ax_cbar.get_position()
ax_cbar.set_position([_cb_pos.x0, _cb_pos.y0, _cb_pos.width * 0.4, _cb_pos.height])

# 패널 C-G 크기 30% 축소 (중심 유지)
for _ax in [ax_c, ax_d, ax_e, ax_f, ax_g]:
    _p  = _ax.get_position()
    _cx = _p.x0 + _p.width  / 2
    _cy = _p.y0 + _p.height / 2
    _nw = _p.width  * 0.8
    _nh = _p.height * 0.45
    _ax.set_position([_cx - _nw/2, _cy - _nh/2, _nw, _nh])

fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches='tight', transparent=False)
fig.savefig(OUTPUT_PDF, bbox_inches='tight')
os.makedirs(STATS_DIR, exist_ok=True)
pd.DataFrame(STATS_ROWS).to_csv(STATS_CSV, index=False)
print(f"Saved {STATS_CSV}")
