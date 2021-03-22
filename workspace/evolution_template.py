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
    levels = np.arange(-0.25, 0.26, 0.01)
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
        return "%.1f%%/s" % (0.1 * r_int)

    # contraction start, end; expansion start, end
    frame_boundaries = {        
        "01": [520, 1150, 950, 750],
        "05": [250, 510, 305, 200],
        "10": [225, 390, 225, 100],
        "15": [177, 385, 160, 67],
        "20": [240, 470, 215, 50],
        "25": [183, 435, 155, 40],
        "30": [126, 300, 95, 33],
        "35": [202, 468, 165, 28],
        "40": [123, 360, 90, 25],
        "45": [183, 433, 145, 22],
        "50": [150, 425, 115, 20],
        "55": [137, 388, 100, 18],
        "60": [218, 678, 180, 16]
    }

    frame_ticks = {
        "01": [0, 250, 500, 750, 1000],
        "05": [0, 50, 100, 150, 200],
        "10": [0, 20, 40, 60, 80, 100],
        "15": [0, 15, 30, 45, 60, 67],
        "20": [0, 10, 20, 30, 40, 50],
        "25": [0, 10, 20, 30, 40],
        "30": [0, 10, 20, 30, 33],
        "35": [0, 7, 14, 21, 28],
        "40": [0, 5, 10, 15, 20, 25],
        "45": [0, 5, 10, 15, 22],
        "50": [0, 5, 10, 15, 20],
        "55": [0, 6, 12, 18],
        "60": [0, 4, 8, 12, 16],
    }

    # font = {'weight': 'light', 'size': 10}
    # matplotlib.rc('font', **font)

    rates = ["01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50"]

    fig = plt.figure(figsize=(20, 5)) # use (6, 7) for colorbar
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.gridspec.GridSpec.html
    gs = gridspec.GridSpec(figure=fig, nrows=2, ncols=len(rates), hspace=0.1, wspace=0.45)
    idx = 0
    axs = []

    for f in rates:
        plt_config = {
            "folder": f,   # sub-folder, also use as the name of the test case
            "cl_h5": "",   # centerline h5
            "t_dir": "y",  # time direction in the data matrix
        }
        plt_config["cl_h5"] = plt_config["folder"] + ".h5"
        plt_config["contraction_start"] = frame_boundaries[plt_config["folder"]][0]
        plt_config["contraction_end"] = frame_boundaries[plt_config["folder"]][0] + frame_boundaries[plt_config["folder"]][-1]
        plt_config["expansion_start"] = frame_boundaries[plt_config["folder"]][1]
        plt_config["expansion_end"] = frame_boundaries[plt_config["folder"]][1] + frame_boundaries[plt_config["folder"]][-1]
        plt_config["absolute_start"] = frame_boundaries[plt_config["folder"]][2]
        plt_config["absolute_end"] = frame_boundaries[plt_config["folder"]][2] + 200

        centerline_h5 = os.path.join(data_root, plt_config["folder"], plt_config["cl_h5"])
        G = read_center_line_h5(centerline_h5, gx_ds="gx") * 2.
        print(G.max(), G.min())

        # expansion
        ax = fig.add_subplot(gs[0, idx])        
        ax.set_title(get_strain_rate(plt_config["folder"]) + "\n")
        cs = draw_G(G[:, plt_config["expansion_start"]:plt_config["expansion_end"] + 1], ax)
        # ax.set_xlabel("position (pixel)")
        ax.set_xticks([0, 600])
        ax.set_xticklabels([])
        if idx == 0:
            ax.set_ylabel("time (frame)", fontsize=12)
            # ax.text(-500, 1100, "strain rate =", fontsize=12)        
        if idx == 0:
            ax.set_yticks([-250, 0, 250, 500, 750])
            ax.set_yticklabels([0, 250, 500, 750, 1000])
        else:
            ax.set_yticks(frame_ticks[plt_config["folder"]])
        axs.append(ax)

        # contraction
        ax = fig.add_subplot(gs[1, idx])
        draw_G(G[:, plt_config["contraction_start"]:plt_config["contraction_end"] + 1], ax)
        ax.set_xticks([0, 600])
        ax.set_xticklabels([0, 195])
        if idx == 5:
            # ax.text(500, -13, r"$x_2$ ($\mu$m)", fontsize=12)
            ax.set_xlabel(r"$x_2$ ($\mu$m)", fontsize=12)
        if idx == 0:
            ax.set_ylabel("time (frame)", fontsize=12)
        ax.set_yticks(frame_ticks[plt_config["folder"]])
        axs.append(ax)

        # # absolute time
        # ax = fig.add_subplot(gs[2, idx])
        # draw_G(G[:, plt_config["absolute_start"]:plt_config["absolute_end"] + 1], ax)
        # if idx == 6:
        #     ax.set_xlabel(r"$x_2$ ($\mu$m)")
        # ax.set_xticks([0, 600])
        # ax.set_xticklabels([0, 195])
        # if idx == 0:
        #     ax.set_ylabel("time (frame)")
        # ax.set_yticks([0, 50, 100, 150, 200])
        # axs.append(ax)

        idx += 1

    cbar = fig.colorbar(cs, ax=axs, shrink=0.6, pad=0.02)
    cbar.ax.set_xlabel(r"$\partial x_3$/$\partial x_2$", fontsize=12)

    img_file = os.path.join(output_img_dir, "normalized_time.png")
    _logger.info("Save image file %s" % img_file)
    fig.savefig(img_file, bbox_inches='tight', dpi=300) # dpi use 72 for screen, 300 for print
