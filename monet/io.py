import h5py
import numpy as np


def read_h5(h5_file, gx_ds="gradz_x", gy_ds="gradz_y"):
    with h5py.File(h5_file, 'r') as gh5:
        gX = np.array(gh5[gx_ds])
        gY = np.array(gh5[gy_ds])
    return gX, gY

def read_center_line_h5(h5_file, gx_ds="gx"):
    """
    the h5_file is expected to have a data set representing centerline gradient at each time step
    each row or column is a temporal snapshot, the other dimension is the "time direction"
    """
    with h5py.File(h5_file, 'r') as f:
        return np.array(f['gx'])
