#!/usr/bin/bash

# This file is useful to evalute LLMs on the MTEB benchmark. It will create 
# a dedicated tmux session on the specified device. To do so, run the following
# command in the terminal from the root directory of the project.
# ```bash
# $ bash <path_to_file_folder>/eval.sh
# ```

# FOR VISION MODELS, TRAIN ON THE FOLLOWING DATASETS:
# aircraft
# cifar10
# cifar100
# dtd
# eurosat
# flowers102
# food101
# imagenet (validation to split into train / test with 80/20 ratio)
# pet
# domainnet clipart
# cifar10-C speckle noise severity 5
# TOTAL: 11 datasets (if I want to scale, add more domainnet and cifar10-C)
