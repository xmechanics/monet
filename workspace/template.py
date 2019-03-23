import os
import sys
import h5py
import logging
import time
import numpy as np

from joblib import Parallel, delayed

_logger = logging.getLogger(__name__)
workspace = os.path.dirname(os.path.abspath(__file__))
prj_root = os.path.dirname(workspace)
sys.path.insert(0, prj_root)

from monet.io import read_h5
from monet.reconstruct import solve_compatibility, reconstruct


def flatten_frm(z_dir, z_flat_dir, frm, gX_bg, gY_bg):
        init_logging()
        t0 = time.time()
        z_h5 = os.path.join(z_dir, "z_%03d.h5" % frm)
        z_flat_h5 = os.path.join(z_flat_dir, "z_flat_%03d.h5" % frm)
        if not os.path.exists(z_flat_h5):
            gX, gY = read_h5(z_h5, gx_ds="gx", gy_ds="gy")
            gX -= gX_bg
            gY -= gY_bg
            Z = reconstruct(gX, gY)
            with h5py.File(z_flat_h5, 'w') as h5:
                h5.create_dataset("z", data=Z)
                h5.create_dataset("gx", data=gX)
                h5.create_dataset("gy", data=gY)
        _logger.info("Flattened frame %d using %.2f sec" % (frm, time.time() - t0))


def reconstruct_frm(frm):
    init_logging()
    t0 = time.time()
    grad_h5 = os.path.join(grad_dir, "gradz%04d.h5" % frm)
    z_h5 = os.path.join(z_dir, "z_%03d.h5" % frm)
    if not os.path.exists(z_h5):
        gX, gY = read_h5(grad_h5, gx_ds="gradz_x", gy_ds="gradz_y")
        gX, gY = solve_compatibility(gX, gY)
        Z = reconstruct(gX, gY)
        with h5py.File(z_h5, 'w') as h5:
            h5.create_dataset("z", data=Z)
            h5.create_dataset("gx", data=gX)
            h5.create_dataset("gy", data=gY)
    _logger.info("Reconstructed frame %d using %.2f sec" % (frm, time.time() - t0))


if __name__ == "__main__":
    from monet import init_logging
    init_logging()

    data_root = os.path.join(workspace, "data", "NiTi_Strain3E1_cyc3")
    grad_dir = os.path.join(data_root, "grad")
    n_files = len([name for name in os.listdir(grad_dir) if name.endswith(".h5")])
    _logger.info("Found %d gradient files in %s" % (n_files, grad_dir))

    # =====
    # reconstruct
    # =====
    z_dir = os.path.join(data_root, "z")

    if not os.path.exists(z_dir):
        os.makedirs(z_dir, exist_ok=True)
    
    Parallel(n_jobs=4)(delayed(reconstruct_frm)(grad_dir, z_dir, frm) for frm in range(250))

    # =====
    # flatten: remove a background gradient field
    # =====

    z0_h5 = os.path.join(z_dir, "z_001.h5")
    gX0, gY0 = read_h5(z0_h5, gx_ds="gx", gy_ds="gy")
    gX_bg = gX0.mean()
    gY_bg = gY0.mean()

    z_flat_dir = os.path.join(data_root, "z_flat")
    if not os.path.exists(z_flat_dir):
        os.makedirs(z_flat_dir, exist_ok=True)

    Parallel(n_jobs=4)(delayed(flatten_frm)(z_dir, z_flat_dir, frm, gX_bg, gY_bg) for frm in range(n_files))

