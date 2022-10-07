import math
import matplotlib.pyplot as plt

def _trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def plot_dist(data, bins=10, size=(10, 8)):
    sq = math.sqrt(data.shape[1])
    d_ceil = math.ceil(sq)
    d_floor = math.floor(sq)

    if (d_ceil * d_floor) >= data.shape[1]:
        n_rows = d_floor
        n_cols = d_ceil
    else:
        n_rows = n_cols = d_ceil

    fig, axs = plt.subplots(n_rows, n_cols, figsize=size)
    axs = _trim_axs(axs, data.shape[1])

    for i, ax in enumerate(axs):
        ax.hist(data[:, i], bins=bins)
    
    fig.tight_layout()
    plt.show()