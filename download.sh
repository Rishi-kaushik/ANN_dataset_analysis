#!/usr/bin/env bash
# Grab the four common 1 M ANN datasets right here.

mkdir -p datasets
cd datasets

# SIFT-1M (128-D)
wget -nc ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
rm sift.tar.gz

# GIST-1M (960-D)
wget -nc ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
tar -xzf gist.tar.gz
rm gist.tar.gz

# Deep-1M (96-D, HDF5 already)
wget -nc -O deep1M.hdf5 https://ann-benchmarks.com/deep-image-96-angular.hdf5

# GloVe-1M (100-D, HDF5 already)
wget -nc -O glove1M.hdf5 https://ann-benchmarks.com/glove-100-angular.hdf5

echo "All downloads finished."

