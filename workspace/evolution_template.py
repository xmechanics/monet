import os
import sys
import h5py
import logging
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

_logger = logging.getLogger(__name__)
workspace = os.path.dirname(os.path.abspath(__file__))
prj_root = os.path.dirname(workspace)
sys.path.insert(0, prj_root)


def create_output_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# get [space steps] x [time steps] mesh from the shape of G
def meshing(G):
    x_vals = np.arange(G.shape[0])
    y_vals = np.arange(G.shape[1])
    X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')
    mid = (G.max() + G.min()) / 2
    mid = 0 # not to offset
    return X, Y, G.T - mid


def draw_G(G, ax):
    X, Y, Z = meshing(G)
    # levels = MaxNLocator(nbins=20).tick_values(Z.min(), Z.max())
    levels = np.arange(-0.13, 0.14, 0.01)
    cmap = plt.get_cmap('RdBu')
    return ax.contourf(X, Y, G.T, levels=levels, cmap=cmap)


if __name__ == "__main__":
    from monet import init_logging
    from monet.io import read_center_line_h5
    
    init_logging()
    data_root = os.path.join(workspace, "data", "20210106SiliconData")
    output_root = os.path.join(workspace, "output", "20210106SiliconData")
    output_img_dir = os.path.join(output_root, "img")
    create_output_dir(output_img_dir)

    # convert folder name to strain rate
    def get_strain_rate(folder):
        r_int = int(folder)
        return "%.1f%% / sec" % (0.1 * r_int)

    # contraction start, end; expansion start, end
    frame_boundaries = {        
        "01": [210, 210 + 1250, 1050, 1050 + 1250],
        "05": [200, 200 + 250, 550, 550 + 250],
        "10": [230, 230 + 75, 425, 425 + 75],
        "25": [175, 175 + 50, 440, 440 + 50],
        "50": [144, 144 + 25, 429, 429 + 25],
        "60": [213, 213 + 21, 680, 680 + 21]
    }

    frame_ticks = {
        "01": [0, 250, 500, 750, 1000, 1250],
        "05": [0, 50, 100, 150, 200, 250],
        "10": [0, 15, 30, 45, 60, 75],
        "25": [0, 10, 20, 30, 40, 50],
        "50": [0, 5, 10, 15, 20, 25],
        "60": [0, 3, 6, 9, 12, 15, 18, 21],
    }

    for f in ["01", "05", "10", "25", "50", "60"]:
        plt_config = {
            "folder": f,   # sub-folder, also use as the name of the test case
            "cl_h5": "",   # centerline h5
            "t_dir": "y",  # time direction in the data matrix
        }
        plt_config["cl_h5"] = plt_config["folder"] + ".h5"
        plt_config["contraction_start"] = frame_boundaries[plt_config["folder"]][0]
        plt_config["contraction_end"] = frame_boundaries[plt_config["folder"]][1]
        plt_config["expansion_start"] = frame_boundaries[plt_config["folder"]][2]
        plt_config["expansion_end"] = frame_boundaries[plt_config["folder"]][3]

        centerline_h5 = os.path.join(data_root, plt_config["folder"], plt_config["cl_h5"])
        G = read_center_line_h5(centerline_h5, gx_ds="gx")

        fig = plt.figure(figsize=(6, 8)) # use (6, 7) for colorbar
        # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.gridspec.GridSpec.html
        gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=2)

        plt.title("strain rate: " + get_strain_rate(plt_config["folder"]) + "\n")
        plt.axis('off')

        axs = []

        # contraction
        ax = fig.add_subplot(gs[0, 0])
        draw_G(G[:, plt_config["contraction_start"]:plt_config["contraction_end"] + 1], ax)
        ax.set_xlabel("position (pixel)")
        ax.set_xticks([0, 200, 400, 600])
        ax.set_ylabel("time (frame)")    
        ax.set_yticks(frame_ticks[plt_config["folder"]])
        axs.append(ax)

        # expansion
        ax = fig.add_subplot(gs[0, 1])
        cs = draw_G(G[:, plt_config["expansion_start"]:plt_config["expansion_end"] + 1], ax)
        ax.set_xlabel("position (pixel)")
        ax.set_xticks([0, 200, 400, 600])
        ax.set_yticks(frame_ticks[plt_config["folder"]])
        ax.set_yticklabels([])
        axs.append(ax)

        # cbar = fig.colorbar(cs, ax=axs, shrink=0.5)
        # cbar.ax.set_xlabel("dz/dx")

        img_file = os.path.join(output_img_dir, plt_config["folder"] + ".png")
        _logger.info("Save image file %s" % img_file)
        fig.savefig(img_file, bbox_inches='tight', dpi=72) # dpi use 72 for screen, 300 for print
