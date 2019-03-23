import h5py
import numpy as np


def read_h5(h5_file, gx_ds="gradz_x", gy_ds="gradz_y"):
    with h5py.File(h5_file, 'r') as gh5:
        gX = np.array(gh5[gx_ds])
        gY = np.array(gh5[gy_ds])
    return gX, gY
