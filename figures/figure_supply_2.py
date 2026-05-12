import numpy as np
import pickle
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community.quality import modularity
import itertools
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from figure_style import set_paper_style, add_panel_label_fig,add_panel_label,plot_division_box_with_stats,darw_region_bar, region as style_regions
set_paper_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_2.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_2.pdf"


def first_existing_path(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return Path(paths[0])


def load_log_outin_degree_by_region():
    region_fcv_path = first_existing_path(
        DATA_DIR / 'fig1_prism_D_FCS_FCV_bar.csv',
    )
    degree_fcv_path = first_existing_path(
        DATA_DIR / 'fig4_prism_C_degree_FCV.csv',
    )

    region_fcv_df = pd.read_csv(region_fcv_path)
    degree_fcv_df = pd.read_csv(degree_fcv_path)

    fcv_to_region = {
        round(float(fcv), 8): region
        for region, fcv in zip(region_fcv_df["Region"], region_fcv_df["FCV"])
    }

    allowed_regions = set(region_fcv_df["Region"].tolist())
    degree_by_region = {}
    for _, row in degree_fcv_df.iterrows():
        fcv_key = round(float(row["FCV"]), 8)
        region = fcv_to_region.get(fcv_key)
        if region is None:
            continue
        degree_by_region[region] = float(row["log10_OutIn_degree"])

    return degree_by_region, allowed_regions


def load_subject_log_outin_degree_data(allowed_regions):
    region_sc_path = first_existing_path(
        DATA_DIR / 'region_sc.npy',
    )
    region_sc = np.load(region_sc_path)
    out_degree = np.sum(region_sc, axis=2)
    in_degree = np.sum(region_sc, axis=1)

    log_outin_degree_data = [[] for _ in range(len(style_regions))]
    log_outin_degree_sel_n = []

    for i, region_name in enumerate(style_regions):
        if region_name not in allowed_regions:
            continue
        values = np.log10((out_degree[:, i] + 1.0) / (in_degree[:, i] + 1.0))
        values = values[np.isfinite(values)]
        if len(values) > 0:
            log_outin_degree_data[i] = values.tolist()
            log_outin_degree_sel_n.append(i)

    return log_outin_degree_data, log_outin_degree_sel_n

brain_division_list = [
    0,  # 1  Medial Octavolateral Nucleus
    0,  # 2  Cerebellum
    0,  # 3  Medulla Oblongata strip1
    0,  # 4  Medulla Oblongata strip2
    0,  # 5  Medulla Oblongata strip3
    0,  # 6  Medulla Oblongata strip4
    0,  # 7  Medulla Oblongata strip5
    0,  # 8  Interpeduncular Nucleus
    0,  # 9  Inferior Olive
    1,  # 10 Caudal Hypothalamus
    0,  # 11 Raphe Nucleus
    0,  # 12 Tegmentum
    0,  # 13 Anterior Reticular Formation
    0,  # 14 Intermediate Reticular Formation
    0,  # 15 Posterior Reticular Formation
    4,  # 16 Glossopharyngeal Ganglion
    1,  # 17 Habenula
    1,  # 18 Intermediate Hypothalamus
    1,  # 19 Rostral Hypothalamus
    4,  # 20 Octaval Ganglion
    2,  # 21 Olfactory Bulb
    2,  # 22 Olfactory Epithelium
    2,  # 23 Pallium
    1,  # 24 Pituitary
    1,  # 25 Posterior Tuberculum
    2,  # 26 Preoptic Region
    1,  # 27 Pretectum
    4,  # 28 Retina
    2,  # 29 Subpallium
    3,  # 30 Tectum
    1,  # 31 Thalamus
    3,  # 32 Torus Longitudinalis
    3,  # 33 Torus Semicircularis
    4,  # 34 Trigeminal Ganglion
    0,  # 35 Vagal Region
    0   # 36 Vagus Motor Neurons
]
brain_division_list=np.array(brain_division_list)
brain_division_list=np.concatenate((brain_division_list,brain_division_list));
'''
def fun_plot_box_scatter_statcomparison(ax,x_data,  ylabel_str):
    
    import itertools
    from scipy.stats import kruskal, mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    # division 선택 (예: 0~3, Peripheral=4 제외)
    valid_divisions = [2, 1, 3,0]

    division_names = {
        0: 'Hind',
        1: 'Dien',
        2: 'Telen',
        3: 'Mesen'
    }

    # region-level mean across files

    t_region_data = x_data        # shape [n_regions]

    new_region=[];
    new_region_data=[];

    #for i in range(72):
        #if(len(t_region_data[i])>2):
            #new_region.append(i);
            #new_region_data.append(t_region_data[i]);

    new_brain_division_list=brain_division_list[new_region];
    #new_region_data=np.array(new_region_data);
    #division_data = {}

    division_data=x_data;
    
    #for div in valid_divisions:
       # idx = np.where(new_brain_division_list == div)[0]
        #division_data[div] = new_region_data[idx]
        
     

    kw_stat, kw_p = kruskal(*[division_data[d] for d in valid_divisions])
    print(f"Kruskal–Wallis {ylabel_str}: H={kw_stat:.3f}, p={kw_p:.4g}")



    pairs = list(itertools.combinations(valid_divisions, 2))

    pvals = []
    pair_labels = []

    for d1, d2 in pairs:
        u, p = mannwhitneyu(
            division_data[d1],
            division_data[d2],
            alternative='two-sided'
        )
        pvals.append(p)
        pair_labels.append((d1, d2))

    # 다중 비교 보정 (Holm)
    reject, pvals_corr, _, _ = multipletests(pvals, method='holm')

    significant_pairs = [
        (pair_labels[i][0], pair_labels[i][1], pvals_corr[i])
        for i in range(len(pairs)) if reject[i]
    ]

    print("Significant pairs (Holm corrected):")
    for d1, d2, p in significant_pairs:
        print(f"{division_names[d1]} vs {division_names[d2]} : p={p:.4g}")
        
    import seaborn as sns

    #plt.figure(figsize=(7,5))
    #plt.rcParams['font.size'] = 14
    plot_data = []
    plot_labels = []

    for d in valid_divisions:
        plot_data.extend(division_data[d])
        plot_labels.extend([division_names[d]] * len(division_data[d]))

    sns.boxplot(x=plot_labels, ax=ax,y=plot_data, linewidth=1.2,width=0.5, showfliers=False,palette=[division_colors[d] for d in valid_divisions])
    sns.stripplot(x=plot_labels, ax=ax,y=plot_data, alpha=0.7,color='black', size=5, jitter=True)

    ax.set_ylabel(ylabel_str)
    #plt.title("Empirical FC mean across brain divisions")

    # significance bar
    y_max = max(plot_data)
    h = 0.05    * (y_max - min(plot_data))
    level = 0
 
    if p < 0.001:
            p_text = "***"
    elif p < 0.01:
            p_text = "**"
    else:
            p_text = "*"
        
        # 혹은 요청하신 대로 (p=0.000) 형식 유지
        #p_label = f"{p_text}\n(p={p:.3f})" if p >= 0.001 else f"{p_text}\n(p<0.001)"
    p_label = f"{p_text}" if p >= 0.001 else f"{p_text}"
        
    for d1, d2, p in significant_pairs:
        x1 = valid_divisions.index(d1)
        x2 = valid_divisions.index(d2)
        y = y_max + h * level

        ax.plot([x1, x1, x2, x2], [y, y+h*0.2, y+h*0.2, y], lw=1.5, c='k')
        ax.text((x1+x2)/2, y+h*0.25, f"* (p={p:.3f})",
                ha='center', va='bottom', fontsize=10)
        level += 1

    #plt.tight_layout()
    #plt.savefig(filename_str,transparent=True, dpi=300)
    #plt.close();
'''
def remove_nan_from_metric(metric_data):
    """
    metric_data: list of length 72
                 each element = list of values (subjects)
    return: same structure, NaN 제거됨
    """
    clean_data = []

    for vals in metric_data:
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        clean_data.append(arr.tolist())

    return clean_data
'''
def add_panel_label(ax, label):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes,
    fontsize=22, fontweight='bold', va='top', ha='left')
'''
# ============================================================
# Load metrics data
# ============================================================

network_metrics_path = first_existing_path(
    DATA_DIR / 'sc_original_per_area_network_metrics.pkl',
)
with open(network_metrics_path, 'rb') as f:
    data = pickle.load(f)

metrics = {
    'In-degree': data['degree_data'],
    'Clustering': data['clustering_data'],
    'Betweenness': data['BC_mean_data'],
    'GlobalEfficiency': data['Eglob_data'],
    'ModularityQ': data['q_data']
}

metrics_clean = {}
for name, data_ in metrics.items():
    metrics[name] = remove_nan_from_metric(data_)

valid_divisions = [2, 1, 3, 0]


x_cluster_data=[[] for _ in range(4)];
x_between_data=[[] for _ in range(4)];
x_GlobalEfficiency_data=[[] for _ in range(4)];
x_modularityQ_data=[[] for _ in range(4)];

sel_n=[];

for i in range(72):
    d = brain_division_list[i]

    if d <= 3:
        x_cluster_data[d].extend(metrics['Clustering'][i]);
        x_between_data[d].extend(metrics['Betweenness'][i]);
        x_GlobalEfficiency_data[d].extend(metrics['GlobalEfficiency'][i]);
        x_modularityQ_data[d].extend(metrics['ModularityQ'][i]);

sel_n=[];
clus_data=[];
gl_data=[];
mq_data=[];
dac_sel_n=[];

for i in range(72):
    clus_data.append(metrics['Clustering'][i])
    gl_data.append(metrics['GlobalEfficiency'][i])
    mq_data.append(metrics['ModularityQ'][i])
    if(len(metrics['Clustering'][i])>3):
        sel_n.append(i);
    if(len(metrics['Clustering'][i])>2):
        dac_sel_n.append(i);


dac_data_path = first_existing_path(
    DATA_DIR / 'total_selected_region_dac_data.npz',
)
load_data=np.load(dac_data_path);
cont_csel_id=load_data['arr_0'];
cont_new_total_dac_out_data=load_data['arr_1'];
cont_new_total_dac_in_data=load_data['arr_2'];
cont_sel_region=load_data['arr_3'];

for i in range(len(cont_new_total_dac_out_data)):
    if(len(cont_new_total_dac_out_data[i])>0):
        cont_new_total_dac_out_data[i]*=1;
        cont_new_total_dac_in_data[i]*=1;
        
         

        

# ============================================================
# Region names
# ============================================================
lregion = ['MON','Cb','MOS1','MOS2','MOS3','MOS4','MOS5','IPN','IO','Hc','Ra','T',
           'aRF','imRF','pRF','GG','Hb','Hi','HR','OG','OB','OE','P','Pi','PT',
           'PO','PrT','R','SP','TeO','Th','TL','TS','TG','VR','NX']

rregion = ['rMON','rCb','rMOS1','rMOS2','rMOS3','rMOS4','rMOS5','rIPN','rIO','rHc',
           'rRa','rT','raRF','rimRF','rpRF','rGG','rHb','rHi','rHR','rOG','rOB',
           'rOE','rP','rPi','rPT','rPO','rPrT','rR','rSP','rTeO','rTh','rTL',
           'rTS','rTG','rVR','rNX']

regions = lregion + rregion
N_REGION = 72
log_outin_degree_by_region, log_outin_allowed_regions = load_log_outin_degree_by_region()
log_outin_degree_data, log_outin_degree_sel_n = load_subject_log_outin_degree_data(
    log_outin_allowed_regions
)

# ============================================================
# Brain divisions
# ============================================================


division_names = {
    0: 'Hindbrain',
    1: 'Dienc',
    2: 'Telenc',
    3: 'Mesenc'
}

division_colors = {
    0: '#6baed6',  # blue - Hindbrain
    1: '#fdae6b',  # orange - Diencephalon
    2: '#74c476',  # green - Telencephalon
    3: '#fb6a4a',  # red - Mesencephalon
}


# ============================================================
# File paths for network diagrams
# ============================================================
save_file_patha_synapse_sc_aux_data = []


# ============================================================
# Create combined figure
# ============================================================

fig = plt.figure(figsize=(16, 18))

# Subplot positions for 2x3 layout
#ax1 = plt.subplot2grid((6, 6), (0, 0),rowspan=2,colspan=3)  # Network diagram Pallium
#ax2 = plt.subplot2grid((6, 6), (0, 3),rowspan=2,colspan=3)  # Network diagram Cerebellum
#ax3 = plt.subplot2grid((3, 6), (1, 0),rowspan=2,colspan=3)  # Network diagram Pallium
#ax4 = plt.subplot2grid((3, 6), (1, 3),rowspan=2,colspan=3)  # Network diagram Cerebellum

#ax3 = plt.subplot2grid((2, 6), (0, 2))  # Placeholder

#ax4 = plt.subplot2grid((6, 6), (2, 0),colspan=2)  # Clustering
#ax5 = plt.subplot2grid((6, 6), (2, 2),colspan=2)  # Modularity Q
#ax6 = plt.subplot2grid((6, 6), (2, 4),colspan=2)  # Global Efficiency

ax1 = plt.subplot2grid((6, 6), (0, 0),colspan=6)
ax2 = plt.subplot2grid((6, 6), (1, 0),colspan=6)
ax3 = plt.subplot2grid((6, 6), (2, 0),colspan=6)
ax4 = plt.subplot2grid((6, 6), (3, 0),colspan=6)
ax5 = plt.subplot2grid((6, 6), (4, 0),colspan=6)
ax6 = plt.subplot2grid((6, 6), (5, 0),colspan=6)


axs = [ax1,ax2,ax3,ax4,ax5,ax6]



for tax in axs:
    tax.tick_params(axis='both', which='both', direction='out',
                    bottom=True, left=True, length=4, width=1.2)

darw_region_bar(ax1,clus_data,sel_n);
darw_region_bar(ax2,mq_data,sel_n);
darw_region_bar(ax3,gl_data,sel_n);
darw_region_bar(ax4,cont_new_total_dac_in_data,dac_sel_n);
darw_region_bar(ax5,cont_new_total_dac_out_data,dac_sel_n);
darw_region_bar(ax6,log_outin_degree_data,log_outin_degree_sel_n);

ax1.set_ylabel('Clustering');
ax3.set_ylabel('Global Efficiency');
ax2.set_ylabel('ModularityQ');
ax4.set_ylabel('Post-DCA');
ax5.set_ylabel('Pre-DCA');
ax6.set_ylabel('log10(Out/In-degree)');


ax1.set_xlim(-1.1,len(sel_n));
ax2.set_xlim(-1.1,len(sel_n));
ax3.set_xlim(-1.1,len(sel_n));
ax4.set_xlim(-1.1,len(dac_sel_n));
ax5.set_xlim(-1.1,len(dac_sel_n));
ax6.set_xlim(-1.1,len(log_outin_degree_sel_n));
ax4.set_yticks([-0.4,-0.2, 0, 0.2]);

axes=[ax1, ax2,ax3,ax4,ax5,ax6]
for ax in axes:
    ax.yaxis.set_label_coords(-0.055, 0.5)

for ax in axes:  
    pos = ax.get_position()
    ax.set_position([pos.x0+0.02, pos.y0, pos.width*0.95, pos.height*0.90]) 
    x0=pos.width*0.20
    
add_panel_label_fig(fig, ax1, 'A', dx=-0.08, dy=0.01)
add_panel_label_fig(fig, ax2, 'B', dx=-0.08, dy=0.01)
add_panel_label_fig(fig, ax3, 'C', dx=-0.08, dy=0.01)
add_panel_label_fig(fig, ax4, 'D', dx=-0.08, dy=0.01)
add_panel_label_fig(fig, ax5, 'E', dx=-0.08, dy=0.01)
add_panel_label_fig(fig, ax6, 'F', dx=-0.08, dy=0.01)




plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.savefig(OUT_PDF, bbox_inches='tight')
