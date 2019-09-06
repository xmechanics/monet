# Build pre-computed coefficient matrix
from __future__ import print_function, absolute_import
import logging
import numba
import numpy as np

from scipy import sparse
from joblib import Parallel, delayed

_logger = logging.getLogger(__name__)


def build_A(M, N):
    """
    build complete A matrix:
    [
        Q E.t
        E 0
    ]
    """
    Qmat = build_Q(M, N)
    Emat = build_E(M, N)    
    Emat_t = Emat.transpose(copy=True)
    Amat = sparse.bmat([[Qmat, Emat_t], [Emat, None]])
    return Amat.tocsc()


def build_Q(M, N):
    """build quadratic matrix for MxN pixels"""
    Q = sparse.identity(2 * M * N, dtype=np.int16)
    _logger.info("Built a Q matrix of size (%d, %d)" % (Q.shape[0], Q.shape[1]))
    return Q


def build_E(M, N):
    """build the compatibility constraint matrix"""
    Emat = None
    for i in range(M - 1):
        _logger.info("Building E matrix for i=%d" % i)
        Smat = build_E_i(i, M, N)
        if Emat is None:
            Emat = Smat
        else:
            Emat = sparse.vstack([Emat, Smat])
    return Emat


def build_E_i(I, M, N):
    """build the compatibility constraint matrix for i=I"""
    Smat = None
    grp_size = 1000
    for grp in range(np.ceil((N - 1)/grp_size).astype(np.int)):
        j_start = grp * grp_size
        j_end = min(N - 1, j_start + grp_size)
        sub_mats = Parallel(n_jobs=-1)(delayed(build_quadruple)(I, j, M, N) for j in range(j_start, j_end))
        if Smat is None:
            Smat = sparse.vstack(sub_mats)
        else:
            Smat = sparse.vstack([Smat] + sub_mats)
        _logger.info("Smat rows: %d" % (Smat.shape[0]))
    return Smat


def build_quadruple(i, j, M, N):
    row, col, data = build_quadruple_data(i, j, M, N)
    return sparse.coo_matrix((data, (row, col)), shape=(1, 2 * M * N), dtype=np.int16)


@numba.jit(nopython=True)
def build_quadruple_data(i, j, M, N):
    """
    return rows, cols, and data for one compatibility equation:
    u_i,j + v_i+1,j - u_i,j+1 - v_i,j == 0
    """
    #TODO: explore other compatibility conditions
    u1 = uij(i, j, M, N)
    # u2 = uij(i + 1, j, M, N)
    u3 = uij(i, j + 1, M, N)
    # u4 = uij(i + 1, j + 1, M, N)
    v1 = vij(i, j, M, N)
    v2 = vij(i + 1, j, M, N)
    # v3 = vij(i, j + 1, M, N)
    # v4 = vij(i + 1, j + 1, M, N)

    row = [0] * 4
    col = [u1, v2, u3, v1]
    data = [1, 1, -1, -1]

    return row, col, data


@numba.jit(nopython=True)
def uij(i, j, M, N):
    return i * M + j


@numba.jit(nopython=True)
def vij(i, j, M, N):
    return uij(i, j, M, N) + (M * N)


if __name__ == "__main__":
    import os
    import sys
    prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, prj_root)

    from monet import init_logging
    init_logging()

    data_dir = os.path.join(prj_root, "data")

    M = 1000
    N = 1000
    mat_A = build_A(M, N)

    npz_file = os.path.join(data_dir, "A.npz")
    sparse.save_npz(npz_file, mat_A)

    print(mat_A.shape, mat_A.dtype)

    print("You can rename and check in the generated coefficient matrix file for M=%d N=%d!" % (M, N))
    print(npz_file)

