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
from monet.solve import solve_g, reconstruct

def reconstruct_frm(frm):
        init_logging()
        t0 = time.time()
        grad_h5 = os.path.join(grad_dir, "gradz%04d.h5" % frm)
        z_h5 = os.path.join(z_dir, "z_%03d.h5" % frm)
        if not os.path.exists(z_h5):
            gX, gY = read_h5(grad_h5)
            gX, gY = solve_g(gX, gY)
            Z = reconstruct(gX, gY)
            with h5py.File(z_h5, 'w') as h5:
                h5.create_dataset("z", data=Z)
                h5.create_dataset("gx", data=gX)
                h5.create_dataset("gy", data=gY)
        _logger.info("Reconstructed frame %d using %.2f sec" % (frm, time.time() - t0))


if __name__ == "__main__":
    from monet import init_logging
    init_logging()

    data_root = os.path.join(workspace, "data", "Au30_thermal_cycle2")
    grad_dir = os.path.join(data_root, "grad")
    z_dir = os.path.join(data_root, "z")

    if not os.path.exists(z_dir):
        os.makedirs(z_dir)

    Parallel(n_jobs=4)(delayed(reconstruct_frm)(frm) for frm in range(375))

