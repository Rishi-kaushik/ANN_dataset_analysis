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
    python ann_stats.py all    # for all datasets
"""

import sys
import struct
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime


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
    q = read_fvecs("datasets/sift/sift_query.fvecs")
    base = read_fvecs("datasets/sift/sift_base.fvecs")
    gt = read_ivecs("datasets/sift/sift_groundtruth.ivecs")
    return q, base, gt, l2_dist


def load_gist():
    q = read_fvecs("datasets/gist/gist_query.fvecs")
    base = read_fvecs("datasets/gist/gist_base.fvecs")
    gt = read_ivecs("datasets/gist/gist_groundtruth.ivecs")
    return q, base, gt, l2_dist


def load_hdf5(fname):
    with h5py.File(fname, "r") as f:
        q = f["test"][:]
        base = f["train"][:]
        gt = f["neighbors"][:, :]          # ground truth indices
    return q, base, gt


def load_deep():
    q, base, gt = load_hdf5("datasets/deep1M.hdf5")
    return q, base, gt, angular_dist       # Deep-1M uses angular


def load_glove():
    q, base, gt = load_hdf5("datasets/glove1M.hdf5")
    return q, base, gt, angular_dist       # cosine distance


LOADERS = {
    "sift":  load_sift,
    "gist":  load_gist,
    "deep":  load_deep,
    "glove": load_glove,
}

# ---------- plotting ----------
def plot_distances(all_dists, names, output_dir):
    """Create a 1xN subplot of violin plots for the distances."""
    num_datasets = len(names)
    fig, axes = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 8), squeeze=False)
    axes = axes.flatten()

    all_dists_long = []
    for name, dists in zip(names, all_dists):
        for d in dists:
            all_dists_long.append({'dataset': name.upper() + '-1M', 'distance': d})

    df = pd.DataFrame(all_dists_long)

    for i, name in enumerate(names):
        dataset_name = name.upper() + '-1M'
        dataset_df = df[df['dataset'] == dataset_name]
        
        sns.violinplot(ax=axes[i], y='distance', data=dataset_df, inner='quartile')
        axes[i].set_title(dataset_name)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Distance' if i == 0 else '')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('Distribution of True Nearest Neighbor Distances', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = os.path.join(output_dir, "distances_plot.png")
    plt.savefig(plot_filename)
    print(f"\nSaved violin plot to {plot_filename}")


# ---------- main ----------

def main():
    if len(sys.argv) != 2 or (sys.argv[1].lower() not in LOADERS and sys.argv[1].lower() != 'all'):
        print("Choose one dataset: sift | gist | deep | glove | all")
        sys.exit(1)

    arg = sys.argv[1].lower()

    # Create output directory
    output_dir_base = "outputs"
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_timestamped = os.path.join(output_dir_base, timestamp)
    os.makedirs(output_dir_timestamped)
    
    if arg == 'all':
        names_to_process = list(LOADERS.keys())
    else:
        names_to_process = [arg]

    all_distances = []

    for name in names_to_process:
        print("-" * 20)
        q, base, gt, dfunc = LOADERS[name]()

        nn_idx = gt[:, 0]                    # first neighbor is the true NN
        dists = dfunc(q, base[nn_idx])
        all_distances.append(dists)

        print(f"{name.upper()}-1M")
        print(f"mean distance to true NN : {dists.mean():.6f}")
        print(f"std  distance to true NN : {dists.std():.6f}")
        print(f"top-5 NN indices for q0  : {gt[0, :5].tolist()}")

    if all_distances:
        plot_distances(all_distances, names_to_process, output_dir_timestamped)


if __name__ == "__main__":
    main()

