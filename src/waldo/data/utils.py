r"""
Utils functions to build dataloaders.

Notes
-----
Loading data can be slow if the num_workers variable is not properly set. Small values are not suitable for large
datasets and large values increase the CPU usage (see https://lightning.ai/docs/pytorch/stable/advanced/speed.html).
In the default DataLoader mode with num_workers=0 where the main process loads the data, we launch runs with
OMP_NUM_THREADS=1 to avoid unnecessary parallelization, following https://github.com/facebookresearch/lingua.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import numpy as np

# ------------------------------------------------------------------------------
#   Data utils
# ------------------------------------------------------------------------------


def deterministic_split(data: np.ndarray, train_size: float = 0.8, train_flag: bool = True) -> np.ndarray:
    r"""Create a deterministic split of the original data."""
    n_samples = len(data)
    n_train = int(train_size * n_samples)

    # Apply a deterministic random permutation of the data
    st0 = np.random.get_state()
    np.random.seed(42)
    indices = np.random.permutation(range(n_samples))
    np.random.set_state(st0)

    # Recover the desired subset of indices
    if train_flag:
        indices = indices[:n_train]
    else:
        indices = indices[n_train:]

    return np.sort(indices)
