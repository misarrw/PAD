import numpy as np

from fuse_filter import fuse_heatmap


def adaptive_thresholding(h_fuse, p):
    h_fuse_fl = h_fuse.flatten()
    n = len(h_fuse_fl)
    h_fl_sorted = h_fuse_fl.sort()
    i = np.floor(p * (n - 1))
    j = (n - 1) * p - i
    return (1 - j) * h_fl_sorted[i] + j * h_fl_sorted[i + 1]