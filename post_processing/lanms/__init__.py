import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile lanms: {}'.format(BASE_DIR))


def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    from .adaptor import merge_quadrangle_n9 as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32'), []
    p = polys.copy()
    p[:, :8] *= precision
    res_pair = nms_impl(p, thres)
    ret, keep = np.array(res_pair[0], dtype='float32'), np.array(res_pair[1], dtype='int64')
    ret[:, :8] /= precision
    return ret, keep
