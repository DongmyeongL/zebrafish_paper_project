
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import lognorm
from scipy.optimize import curve_fit
from figure_style import set_paper_style, add_panel_label_fig, plot_division_box_with_stats

set_paper_style()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_PNG = PROJECT_ROOT / "output" / "png" / "figure_supply_1.png"
OUT_PDF = PROJECT_ROOT / "output" / "pdf" / "figure_supply_1.pdf"


# ============================================================
# Utility functions
# ============================================================

def exp_model(d, A, lamb, C):
    """Exponential decay model."""
    return A * np.exp(-d / lamb) + C

def exp_model1(d, A, lamb):
    """Exponential decay model."""
    return A * np.exp(-d / lamb);


def fit_lognorm_by_distance(dist_data, fc_data, num_bins=40, max_dist=500):
    """
    Bin distance data and fit lognormal distribution
    to FC values in each bin.

    Returns
    -------
    d_centers : ndarray
        Distance bin edges (used as x-axis)
    shape_list : ndarray
        Lognormal shape parameters
    scale_list : ndarray
        Lognormal scale parameters
    """
    bins = np.linspace(dist_data.min(), max_dist, num_bins + 1)
    bin_indices = np.digitize(dist_data, bins)

    d_centers = []
    shape_list = []
    scale_list = []

    for b in range(1, num_bins + 1):
        fc_in_bin = fc_data[bin_indices == b]

        # Skip empty bins
        if len(fc_in_bin) == 0:
            continue

        shape, loc, scale = lognorm.fit(fc_in_bin, floc=0)

        d_centers.append(bins[b])
        shape_list.append(shape)
        scale_list.append(scale)

    return (
        np.array(d_centers),
        np.array(shape_list),
        np.array(scale_list),
    )


# ============================================================
# Data loading
# ============================================================

fc_dist_data = np.load(DATA_DIR / 'figure9_distance_synapse' / 'figure_supply1_fc_dist_data.npz')
dist_data = fc_dist_data['dist_data']
fc_data = fc_dist_data['fc_data']

synapse_dist = np.load(DATA_DIR / 'figure9_distance_synapse' / 'figure_supply1_soma_dist_data.npy')

# ============================================================
# Analysis
# ============================================================

NUM_BINS = 40
MAX_DIST = 500

d, lognorm_shape, lognorm_scale = fit_lognorm_by_distance(
    dist_data,
    fc_data,
    num_bins=NUM_BINS,
    max_dist=MAX_DIST
)

# Exponential fit to scale parameter
p0 = [lognorm_scale.max(), 100.0, lognorm_scale.min()]
popt, pcov = curve_fit(exp_model, d, lognorm_scale, p0=p0, maxfev=10000)

A_hat, lambda_hat, C_hat = popt
perr = np.sqrt(np.diag(pcov))

d_fit = np.linspace(d.min(), d.max(), 300)
scale_fit = exp_model(d_fit, A_hat, lambda_hat, C_hat)

print("Mean lognormal shape parameter:", np.mean(lognorm_shape))

# ============================================================
# Plotting
# ============================================================

fig = plt.figure(figsize=(16, 4))

axA = plt.subplot2grid((1, 2), (0, 0))

axB = plt.subplot2grid((1, 2), (0, 1))


axs = [axA, axB]



for tax in axs:
    tax.tick_params(axis='both', which='both', direction='out',
                    bottom=True, left=True, length=4, width=1.2)





'''
p0 = [lognorm_scale.max(), 100.0];
popt, pcov = curve_fit(exp_model1, dist_data, fc_data, p0=p0, maxfev=10000)

A_hat, lambda_hat = popt
d_fit = np.linspace( dist_data.min(),  dist_data.max(), 300)
scale_fit = exp_model1(d_fit, A_hat, lambda_hat)
'''
'''
axC.hexbin(dist_data+0.01 ,fc_data,bins='log',xscale='log');
axC.plot(d_fit ,scale_fit);

textstr = (
    r'$y(d)=A e^{-d/\lambda}+C$' '\n'
    rf'$A = {A_hat:.3f} \pm {perr[0]:.3f}$' '\n'
    rf'$\lambda = {lambda_hat:.1f} \pm {perr[1]:.1f}$' '\n'
)

axC.text(
    0.05, 0.45, textstr,
    fontsize=11,
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)
'''

axA.axvline(
    x=220,
    linestyle='--',
    linewidth=2,
    color='k',
    alpha=0.8
)


# --- Panel A: Synapse distance distribution
axA.hist(synapse_dist, bins=40)
axA.set_xlabel('Distance')
axA.set_ylabel('Count')

shape_color = 'tab:blue'
scale_color = 'tab:red'

axB_shape = axB                      # left y-axis
axB_scale = axB.twinx()              # right y-axis

# Scatter plots
axB_shape.scatter(
    d, lognorm_shape,
    color=shape_color,
    label='Shape(FC)',
    alpha=0.8
)

axB_scale.scatter(
    d, lognorm_scale,
    color=scale_color,
    marker='s',
    label='Scale(FC)',
    alpha=0.8
)

# Exponential fit (Scale)
axB_scale.plot(
    d_fit, scale_fit,
    color=scale_color,
    linewidth=2
)

# Axis labels
axB_shape.set_xlabel('Distance')
axB_shape.set_ylabel('Lognormal shape(FC)', color=shape_color)
axB_scale.set_ylabel('Lognormal scale(FC)', color=scale_color)

axB_shape.set_ylim((1.2,1.4));

# Right y-axis spine visible for twinx
axB_scale.spines['right'].set_visible(True)
axB_scale.spines['right'].set_color(scale_color)

# Tick colors
axB_shape.tick_params(axis='y', colors=shape_color)
axB_scale.tick_params(axis='y', colors=scale_color, direction='out', length=4, width=1.2)

# Text box (keep neutral)
textstr = (
    r'$y(d)=A e^{-d/\lambda}+C$' '\n'
    rf'$A = {A_hat:.3f} \pm {perr[0]:.3f}$' '\n'
    rf'$\lambda = {lambda_hat:.1f} \pm {perr[1]:.1f}$' '\n'
    rf'$C = {C_hat:.3f} \pm {perr[2]:.3f}$'
)

axB_shape.text(
    0.65, 0.52, textstr,
    transform=axB_shape.transAxes,
    fontsize=10, fontstyle='italic',
    va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

# Legend (merged)
handles1, labels1 = axB_shape.get_legend_handles_labels()
handles2, labels2 = axB_scale.get_legend_handles_labels()
axB_shape.legend(
    handles1 + handles2,
    labels1 + labels2,
    frameon=False,
    loc='upper right',
    bbox_to_anchor=(1.0, 1.15)
)

axes=[axA, axB]
for ax in axes:  
    pos = ax.get_position()
    ax.set_position([pos.x0+0.015, pos.y0+0.11, pos.width*0.8, pos.height*0.8]) 
    x0=pos.width*0.20
    
add_panel_label_fig(fig, axA, 'A', dx=-0.09, dy=0.02)
add_panel_label_fig(fig, axB, 'B', dx=-0.09, dy=0.02)


plt.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
plt.savefig(OUT_PDF, bbox_inches='tight')
plt.close()
