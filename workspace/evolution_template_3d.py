import os
import sys
import h5py
import logging
import time

import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d

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

    # peak_origins = {
    #     "01": [(250, 573), (400, 454), (450, 376)],
    #     "20": [(15, 585), (15, 497), (16, 406)]
    # }

    peak_origins = {
        "01": [(250, 573, 120, 210), (350, 391, 100, 150), (230, 584), (290, 415)],
        "20": [(15, 585, 0, 11), (16, 406, 0, 11)],
        "50": [(7, 577, 0, 4), (6, 431, 0, 5)]
    }

    # font = {'weight': 'light', 'size': 10}
    # matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(10, 10)) # use (6, 7) for colorbar
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.gridspec.GridSpec.html
    gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=1, hspace=0.1, wspace=0.1)
    # gs = gridspec.GridSpec(figure=fig, nrows=1, ncols=2, hspace=0.1, wspace=0.7)
    # idx = 0
    # axs = []
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    # ax2 = fig.add_subplot(gs[1, 0])
    length_scale = 0.325
    fps = 1

    # for f in ["01", "05", "10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"]:
    for idx, f in enumerate(["01"]):
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

        s1 = peak_origins[plt_config["folder"]][0]
        s2 = peak_origins[plt_config["folder"]][1]       
        p1, Y1 = s1, []
        p2, Y2 = s2, []
        th = 100
        if f == "01":
            th = 10
        if f == "20":
            th = 15
        if f == "50":
            th = 100
        kfs = [1500, 1375, 1270, 1150, 1000, 875, 750, 625, 491, 386, 250, 125, 0]
        kp1 = {}
        kp2 = {}
        for t in range(plt_config["contraction_start"] - min(s1[0], s2[0]), plt_config["contraction_end"]):
            Y = G[:, t]
            Yhat = savgol_filter(Y, 51, 3)
            peaks, _ = find_peaks(Yhat - G.mean(), height=0.01, distance=50)
            if t >= plt_config["contraction_start"] + s1[0]:
                pos = np.argmin(np.abs(peaks - p1[1]))
                if len(p1) > 2 and np.abs(peaks[pos] - p1[1]) > th:
                    p1 = (t - plt_config["contraction_start"], p1[1], Yhat[p1[1]])
                else:
                    p1 = (t - plt_config["contraction_start"], peaks[pos], Yhat[peaks[pos]])
                Y1.append(p1)
                if t - plt_config["contraction_start"] in kfs:
                    kp1[t - plt_config["contraction_start"]] = p1
            if t >= plt_config["contraction_start"] + s2[0]:
                pos = np.argmin(np.abs(peaks - p2[1]))
                if len(p2) > 2 and np.abs(peaks[pos] - p2[1]) > th:
                    p2 = (t - plt_config["contraction_start"], p2[1], Yhat[p2[1]])
                else:
                    p2 = (t - plt_config["contraction_start"], peaks[pos], Yhat[peaks[pos]])
                Y2.append(p2)                
                if t - plt_config["contraction_start"] in kfs:
                    kp2[t - plt_config["contraction_start"]] = p2
        t_gap =  plt_config["expansion_start"] - plt_config["contraction_end"]
        s1 = peak_origins[plt_config["folder"]][2]
        s2 = peak_origins[plt_config["folder"]][3]
        for t in range(plt_config["expansion_start"], plt_config["expansion_end"]):
            Y = G[:, t]
            Yhat = savgol_filter(Y, 51, 3)
            peaks, _ = find_peaks(Yhat - G.mean(), height=0.01, distance=50)
            if t <= plt_config["expansion_end"] - s1[0]:
                pos = np.argmin(np.abs(peaks - p1[1]))
                if len(p1) > 2 and np.abs(peaks[pos] - p1[1]) > th:
                    p1 = (t - plt_config["contraction_start"] - t_gap, p1[1], Yhat[p1[1]])
                else:
                    p1 = (t - plt_config["contraction_start"] - t_gap, peaks[pos], Yhat[peaks[pos]])
                Y1.append(p1)
                if t - plt_config["contraction_start"] - t_gap in kfs:
                    kp1[t - plt_config["contraction_start"] - t_gap] = p1
            if t <= plt_config["expansion_end"] - s2[0]:
                pos = np.argmin(np.abs(peaks - p2[1]))
                if len(p2) > 2 and np.abs(peaks[pos] - p2[1]) > th:
                    p2 = (t - plt_config["contraction_start"] - t_gap, p2[1], Yhat[p2[1]])
                else:
                    p2 = (t - plt_config["contraction_start"] - t_gap, peaks[pos], Yhat[peaks[pos]])
                Y2.append(p2)                
                if t - plt_config["contraction_start"] - t_gap in kfs:
                    kp2[t - plt_config["contraction_start"] - t_gap] = p2     
        Y1 = np.array(Y1)
        Y2 = np.array(Y2)

        # ax.plot(Y1[:, 1] * length_scale, Y1[:, 0] / fps, Y1[:, 2], color='tab:blue')
        # ax.plot(Y2[:, 1] * length_scale, Y2[:, 0] / fps, Y2[:, 2], color='tab:green')                

        ax.plot(0 * np.ones(Y1[:, 0].shape), Y1[:, 0] / fps, Y1[:, 2], color='tab:blue', lw=1)
        ax.plot(0 * np.ones(Y2[:, 0].shape), Y2[:, 0] / fps, Y2[:, 2], color='tab:green', lw=1)
        ax.plot(Y1[:, 1] * length_scale, Y1[:, 0] / fps, -0.15 * np.ones(Y1[:, 1].shape), color='tab:blue', lw=1)
        ax.plot(Y2[:, 1] * length_scale, Y2[:, 0] / fps, -0.15 * np.ones(Y2[:, 1].shape), color='tab:green', lw=1)

        # s1 = peak_origins[plt_config["folder"]][0]
        # s2 = peak_origins[plt_config["folder"]][1]        
        # idmin, idmax = s1[2], s1[3]
        # z = np.polyfit(Y1[idmin:idmax, 0], Y1[idmin:idmax, 1], 1)
        # p = np.poly1d(z)
        # xmin, xmax = Y1[idmin, 0], Y1[idmax-1, 0]
        # xrng = xmax - xmin
        # xmin = xmin - xrng * 2.
        # xmax = xmax + xrng * 2.
        # ax.plot([p(xmin), p(xmax)], [xmin, xmax], [-0.15, -0.15], '--k', lw=1)

        A = kp1[kfs[-4]]
        B = kp2[kfs[-5]]
        print(A, B)
        ax.plot([A[1] * length_scale, B[1] * length_scale], [A[0] / fps, B[0] / fps], [-0.15, -0.15], '--', color='tab:red', lw=1)
        l = (A[2] + B[2])/2
        ax.plot([0, 0], [A[0] / fps, B[0] / fps], [l, l], '--', color='tab:red', lw=1)        
        # x1, y1, x2, y2 = 400, 479, 450, 450 
        # ax.plot([x1, x1, x2], [y1, y2, y2], [-0.15, -0.15, -0.15], 'k', lw=1)
        # ax.text(435, 425, -0.15, r'$\omega$', fontsize=12)

        # A = kp1[kfs[1]]
        # B = kp2[kfs[1]]
        # ax.plot([A[1], B[1]], [A[0], B[0]], [-0.15, -0.15], '--k', lw=1)
        # ax.text(435, 750, -0.14, r'$\lambda$', fontsize=12)

        for ik, t in enumerate(kfs):
            if t < plt_config["contraction_end"] - plt_config["contraction_start"]:
                Z = G[:, plt_config["contraction_start"] + t]
            else:
                Z = G[:, plt_config["contraction_start"] + t + t_gap]
            Y = t * np.ones(Z.shape)
            X = np.arange(600)        
            Zhat = savgol_filter(Z, 51, 3)

            tc = 'tab:gray'
            if t in [491, 386]:
                tc = 'tab:red'
            # ax.plot(X[300:], Y[300:], Z[300:], color='k', lw=0.3)
            # ax.plot(X[300:], Y[300:], Zhat[300:], color=tcs[ik], lw=1.5)
            ax.plot(X[:] * length_scale, Y[:] / fps, Z[:], color='k', lw=0.3)
            ax.plot(X[:] * length_scale, Y[:] / fps, Zhat[:], color=tc, alpha=1, lw=0.7)
            
            if t in kp1:
                kp = kp1[t]
                lww = 1
                ax.plot([kp[1] * length_scale, kp[1] * length_scale], [kp[0] / fps, kp[0] / fps], [-0.15, kp[2]], '--', color='tab:blue', lw=lww)
                ax.plot([0, kp[1] * length_scale], [kp[0] / fps, kp[0] / fps], [kp[2], kp[2]], '--', color='tab:blue', lw=lww)

            if t in kp2:
                kp = kp2[t]
                ax.plot([kp[1] * length_scale, kp[1] * length_scale], [kp[0] / fps, kp[0] / fps], [-0.15, kp[2]], '--', color='tab:green', lw=lww)
                ax.plot([0, kp[1] * length_scale], [kp[0] / fps, kp[0] / fps], [kp[2], kp[2]], '--', color='tab:green', lw=lww)

            verts = [(0, t / fps, -0.25), (600 * length_scale, t / fps, -0.25), (600 * length_scale, t / fps, 0.15), (0, t / fps, 0.15)]
            face = mp3d.art3d.Poly3DCollection([verts], color=tc, alpha=0.1, linewidth=0.5)            
            ax.add_collection3d(face)
        
        ax.set_box_aspect([1, 5, 0.5])
        ax.view_init(elev=ax.elev, azim=ax.azim + 30)

        ax.set_xlim([0, 200])
        ax.set_xticks([0, 100, 200])
        ax.set_xlabel(r'$x_2$ ($\mu$m)', fontsize=12)

        ax.set_ylim([0, 1500 / fps])
        ax.set_ylabel('\n\ntime (frame)', fontsize=12)

        ax.set_zlim([-0.25, 0.15])
        ax.set_zticks([-0.2, -0.1, 0, 0.1])
        ax.set_zlabel(r"$\partial x_3$/$\partial x_2$", fontsize=12)

    img_file = os.path.join(output_img_dir, "center_line_3d.png")
    _logger.info("Save image file %s" % img_file)
    # fig.savefig(img_file, bbox_inches='tight', dpi=72) # dpi use 72 for screen, 300 for print
    fig.savefig(img_file, bbox_inches='tight', transparent=True, dpi=300) # dpi use 72 for screen, 300 for print




