#!/usr/bin/env python3
"""
ann_stats.py
Compute mean and std of query-to-true-NN distances and show the
top-5 ground-truth NNs for the first query.

Usage:
    python ann_stats.py sift   # for SIFT-1M
    python ann_stats.py gist   # for GIST-1M
    python ann_stats.py deep   # for Deep-1M
    python ann_stats.py glove  # for GloVe-1M
"""

import sys
import struct
import numpy as np
import h5py


# ---------- I/O helpers for .fvecs / .ivecs ----------

def _read_vecs(path, dtype, skip_first=True):
    """Load .fvecs or .ivecs: first int32 per vector is the dimension."""
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    data = np.fromfile(path, dtype=dtype)
    data = data.reshape(-1, dim + 1)
    return data[:, 1:] if skip_first else data


def read_fvecs(path):
    return _read_vecs(path, np.float32)


def read_ivecs(path):
    return _read_vecs(path, np.int32)


# ---------- distance helpers ----------

def l2_dist(a, b):
    return np.linalg.norm(a - b, axis=1)


def angular_dist(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.sum(a_norm * b_norm, axis=1)   # 1 - cosine


# ---------- per-dataset loaders ----------

def load_sift():
    q = read_fvecs("sift/sift_query.fvecs")
    base = read_fvecs("sift/sift_base.fvecs")
    gt = read_ivecs("sift/sift_groundtruth.ivecs")
    return q, base, gt, l2_dist


def load_gist():
    q = read_fvecs("gist/gist_query.fvecs")
    base = read_fvecs("gist/gist_base.fvecs")
    gt = read_ivecs("gist/gist_groundtruth.ivecs")
    return q, base, gt, l2_dist


def load_hdf5(fname):
    with h5py.File(fname, "r") as f:
        q = f["test"][:]
        base = f["train"][:]
        gt = f["neighbors"][:, :]          # ground truth indices
    return q, base, gt


def load_deep():
    q, base, gt = load_hdf5("deep1M.hdf5")
    return q, base, gt, l2_dist            # Deep-1M uses L2


def load_glove():
    q, base, gt = load_hdf5("glove1M.hdf5")
    return q, base, gt, angular_dist       # cosine distance


LOADERS = {
    "sift":  load_sift,
    "gist":  load_gist,
    "deep":  load_deep,
    "glove": load_glove,
}


# ---------- main ----------

def main():
    if len(sys.argv) != 2 or sys.argv[1].lower() not in LOADERS:
        print("Choose one dataset: sift | gist | deep | glove")
        sys.exit(1)

    name = sys.argv[1].lower()
    q, base, gt, dfunc = LOADERS[name]()

    nn_idx = gt[:, 0]                    # first neighbor is the true NN
    dists = dfunc(q, base[nn_idx])

    print(f"{name.upper()}-1M")
    print(f"mean distance to true NN : {dists.mean():.6f}")
    print(f"std  distance to true NN : {dists.std():.6f}")
    print(f"top-5 NN indices for q0  : {gt[0, :5].tolist()}")


if __name__ == "__main__":
    main()

