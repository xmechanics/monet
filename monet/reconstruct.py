from __future__ import print_function, absolute_import
import os
import logging
import numpy as np
import scipy.sparse.linalg as sla

from scipy import sparse

_logger = logging.getLogger(__name__)


def build_b(gX, gY):
    """
    build b vector in Qx + b
    """
    M, N = gX.shape
    padding = np.zeros((M - 1) * (N - 1))
    vec_b = np.transpose(np.hstack([2 * gX.flatten(), 2 * gY.flatten(), padding]).astype(np.float))
    return vec_b


def solve_compatibility(gX, gY):
    M, N = gX.shape
    vec_b = build_b(gX, gY)

    prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    npz_file = os.path.join(prj_root, "data", "A_%d_%d.npz" % (M, N))
    if not os.path.exists(npz_file):
        raise Exception("No the coefficient matrix for M=%d N=%d, please run coef.py to generate it." % (M, N))
    mat_A = sparse.load_npz(npz_file).astype(np.float)

    _logger.info("Start solving linear system ...")
    solver = sla.splu(mat_A)
    sol = solver.solve(vec_b)

    sX = sol[: M * N].reshape(M, N).astype(np.float32)
    sY = sol[M * N: 2 * M * N].reshape(M, N).astype(np.float32)
    return sX, sY


def reconstruct(gX, gY):
    M = gX.shape[0]
    N = gY.shape[1]
    # left line
    z0 = np.hstack([np.zeros(1), gY[0, :].cumsum()])
    Z0 = np.vstack([z0] * M)[:, :N]
    # going right
    A = np.vstack([np.zeros((1, gX.shape[1])), gX.cumsum(axis=0)])[:M, :]
    return (A + Z0).astype(np.float32)


def reconstruct_t(gX, gY):
    """Reconstruct by a different integration path"""
    # switch X and Y
    gX, gY = gY.transpose(), gX.transpose()
    # normal reconstruct
    Z = reconstruct(gX, gY)
    # switch back X and Y
    return Z.transpose()


def examine(gX, gY, i, j):
    print(gX[i, j] + gY[i+1, j] - gX[i, j+1] - gY[i, j])
