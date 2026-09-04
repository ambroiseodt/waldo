#!/usr/bin/bash

# This file is useful to download PCAM dataset from Zenodo.
# To do so, run the following command in the terminal
# from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/download_models.sh
# ```

DATASET_PATH="datasets/pcam/"

# Createe dataset folder
mkdir -p ${DATASET_PATH}
cd ${DATASET_PATH}

# Download dataset files
wget -c "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz?download=1" \
    -O camelyonpatch_level_2_split_train_x.h5.gz

wget -c "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz?download=1" \
    -O camelyonpatch_level_2_split_train_y.h5.gz

wget -c "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz?download=1" \
    -O camelyonpatch_level_2_split_test_x.h5.gz

wget -c "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz?download=1" \
    -O camelyonpatch_level_2_split_test_y.h5.gz

# Decompress files
gunzip camelyonpatch_level_2_split_train_x.h5.gz
gunzip camelyonpatch_level_2_split_train_y.h5.gz
gunzip camelyonpatch_level_2_split_test_x.h5.gz
gunzip camelyonpatch_level_2_split_test_y.h5.gz