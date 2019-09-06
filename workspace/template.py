import os
import sys
import h5py
import logging
import time

from shutil import rmtree
from joblib import Parallel, delayed

_logger = logging.getLogger(__name__)
workspace = os.path.dirname(os.path.abspath(__file__))
prj_root = os.path.dirname(workspace)
sys.path.insert(0, prj_root)

from monet.io import read_h5
from monet.reconstruct import solve_compatibility, reconstruct
from monet.xdmf import z_file_to_xdmf


def reconstruct_frm(grad_dir, z_dir, frm):
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


def frm_to_xdmf(z_dir, z_prefix, xdmf_dir, frm):
    init_logging()
    z_file = os.path.join(z_dir, "%s_%03d.h5" % (z_prefix, frm))
    xdmf_file, h5_file = z_file_to_xdmf(z_file, xdmf_dir, "xdmf_%03d" % frm)
    _logger.info("Converted frame %d to XDMF format: %s, %s" % (frm, xdmf_file, h5_file))


if __name__ == "__main__":
    from monet import init_logging

    init_logging()

    data_root = os.path.join(workspace, "data", "microribbon")
    grad_dir = os.path.join(data_root, "grad")
    n_files = len([name for name in os.listdir(grad_dir) if name.endswith(".h5")])
    _logger.info("Found %d gradient files in %s" % (n_files, grad_dir))

    # =====
    # reconstruct
    # =====
    z_dir = os.path.join(data_root, "z")

    if not os.path.exists(z_dir):
        os.makedirs(z_dir, exist_ok=True)

    Parallel(n_jobs=4)(delayed(reconstruct_frm)(grad_dir, z_dir, frm) for frm in range(n_files))

    # =====
    # flatten: remove a background gradient field
    # =====

    # pick a frame as background
    z0_h5 = os.path.join(z_dir, "z_001.h5")

    gX0, gY0 = read_h5(z0_h5, gx_ds="gx", gy_ds="gy")
    gX_bg = gX0.mean()
    gY_bg = gY0.mean()

    z_flat_dir = os.path.join(data_root, "z_flat")
    if not os.path.exists(z_flat_dir):
        os.makedirs(z_flat_dir, exist_ok=True)

    Parallel(n_jobs=4)(delayed(flatten_frm)(z_dir, z_flat_dir, frm, gX_bg, gY_bg) for frm in range(n_files))

    # =====
    # xdmf: convert to ParaView format
    # =====
    xdmf_dir = os.path.join(workspace, "data", "microribbon", "xdmf")
    _logger.info("Recreating xdmf dir %s" % xdmf_dir)
    if os.path.exists(z_dir):
        rmtree(xdmf_dir)
    if not os.path.isdir(xdmf_dir):
        os.makedirs(xdmf_dir, exist_ok=True)

    # use flatten gradients?
    use_flatten = True
    # pick frame range for video
    rng = range(30, 150)
    # rng = range(n_files)

    if use_flatten:
        Parallel(n_jobs=4)(delayed(frm_to_xdmf)(z_flat_dir, "z_flat", xdmf_dir, frm) for frm in rng)
    else:
        Parallel(n_jobs=4)(delayed(frm_to_xdmf)(z_dir, "z", xdmf_dir, frm) for frm in rng)
