import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})

EPSILON = 1e-7

def bocd_plot_dm(data, maxes, R, sparsity=5, title='', label='', ylab='', file='', show=False):
    """
    This requires a different BOCD implementation which retains the p(r_t) matrix for the entire run.
    Maintaining this matrix creates an O(n^2) memory requirement, which is not feasible for years
    of minutely data.
    """
    f = plt.figure(figsize=[12, 10])

    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05], wspace=0.05)  # create 2x2 grid, colorbar is 0.05 times the size of the plot
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[2])
    cax = plt.subplot(gs[3])

    peaks = np.where(maxes[:-1]!=maxes[1:]-1)[0]

    if len(peaks):
        for peak in peaks[:-1]:
            ax0.axvline(data.index[peak], linestyle='--', color='darkred', linewidth=0.5)
        ax0.plot([], [], color='darkred', linestyle='--', linewidth=0.5, label='Changepoints')

    ax0.plot(data.index, data, color='darkblue', linewidth=0.5, label=label)

    ax0.set_xticks([])
    ax0.set_title(title)
    ax0.set_ylabel(ylab)

    density_matrix = -np.log(R[0:-1:sparsity, 0:-1:sparsity]+EPSILON)
    mask = R[0:-1:sparsity, 0:-1:sparsity]
    ax1.pcolor(data.index[np.array(range(0, len(R[:,0]) - 1, sparsity))], 
            np.array(range(0, len(R[:,0]) - 1, sparsity)), 
            density_matrix, 
            cmap=cm.Greys, vmin=0, vmax=density_matrix.max(), shading='auto')

    ax1.set_title('Posterior Probability of Run Length')
    ax1.set_ylabel('Run Length')
    ax1.tick_params(axis='x', rotation=45)

    f.colorbar(ScalarMappable(cmap=cm.Greys_r, norm=plt.Normalize(vmin=0, vmax=mask.max())), cax=cax).set_label('Pr(run length)')
    f.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)

    if file != '':
        f.savefig(file, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def bocd_plot(data, maxes, title='', label='', ylab='', file='', show=False):
    f, ax = plt.subplots()

    peaks = np.where(maxes[:-1]!=maxes[1:]-1)[0]

    if len(peaks):
        for peak in peaks[:-1]:
            ax.axvline(data.index[peak], linestyle='--', color='darkred', linewidth=0.5)
        ax.plot([], [], color='darkred', linestyle='--', linewidth=0.5, label='Changepoints')

    ax.plot(data.index, data, color='darkblue', linewidth=0.5, label=label)

    ax.set_title(title)
    ax.set_ylabel(ylab)

    ax.tick_params(axis='x', rotation=45)

    f.legend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)
    f.savefig(file, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

def bocd_plot_comp(metric, port, true, pred):

    f, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))

    if len(true):
        for cp in true:
            axs[0].axvline(cp, linestyle='--', linewidth=0.5, color='darkgreen')
        axs[0].plot([], [], label='True CPs', color='darkgreen', linestyle='--', linewidth=0.5)

    axs[0].plot(port.index, port, linewidth=0.5, c='darkblue')
    axs[0].set_title('LP Share Price')
    axs[0].set_ylabel('Price (USD)')
    axs[0].legend()

    if len(pred):
        for p in pred:
            axs[1].axvline(p, linestyle='--', color='darkred', linewidth=0.5)
        axs[1].plot([], [], color='darkred', linestyle='--', linewidth=0.5, label='Pred CPs')

    axs[1].legend()

    axs[1].plot(metric.index, metric, linewidth=0.5, c='darkblue')
    axs[1].set_title('Entropy Log Diffs')
    axs[1].set_ylabel('Entropy Log Diff')

    axs[1].tick_params(axis='x', rotation=45)

    f.show()