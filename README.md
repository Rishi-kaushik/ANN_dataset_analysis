# ANN Datasets

This project provides scripts to download and analyze four popular Approximate Nearest Neighbor (ANN) search datasets.

## Datasets

The following datasets can be downloaded:

*   **SIFT-1M**: 1 million 128-dimensional vectors.
*   **GIST-1M**: 1 million 960-dimensional vectors.
*   **Deep-1M**: 1 million 96-dimensional vectors from a deep learning model.
*   **GloVe-1M**: 1 million 100-dimensional GloVe word vectors.

## Download

To download the datasets, run the following command:

```bash
bash download.sh
```

This will download and unpack the SIFT and GIST datasets, and download the Deep-1M and GloVe-1M datasets in HDF5 format.

## Analysis

The `ann_stats.py` script computes the mean and standard deviation of the distances between queries and their true nearest neighbors. It also shows the top-5 nearest neighbor indices for the first query vector.

### Installation

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### Usage

To run the analysis on a dataset, specify the dataset name as a command-line argument:

```bash
# For SIFT-1M
python ann_stats.py sift

# For GIST-1M
python ann_stats.py gist

# For Deep-1M
python ann_stats.py deep

# For GloVe-1M
python ann_stats.py glove
```
