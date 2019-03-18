import h5py
import numpy as np


def read_h5(h5_file):
    with h5py.File(h5_file, 'r') as gh5:
        gX = np.array(gh5["gradz_x"])
        gY = np.array(gh5["gradz_y"])
    return gX, gY
