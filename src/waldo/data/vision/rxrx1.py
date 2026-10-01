r"""
RxRx1 Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from wilds import get_dataset

from ...infra import DATASET_DIR


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class RxRx1DatasetConfig:
    r"""
    RxRx1 configuration file.

    Parameters
    ----------
    save_dir: Path
        Path from where to load the data or save them if it does not exist.
    mode: str
        Mode. Options are "train" and "test".
    transform: Compose | None
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "rxrx1"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        valid_modes = ["train", "test"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are {valid_modes}.")


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
class RxRx1Dataset(Dataset):
    r"""
    RxRx1 dataset from [1]_, part of the WILDS benchmark [2]_.

    RxRx1 consists of fluorescence microscopy images of human cells under one of 1,139
    genetic perturbations (siRNA knockdowns), captured across different experimental
    batches. Each image is labeled by which perturbation was applied, making this a
    1,139-way classification task.

    Parameters
    ----------
    config: configuration class with
        save_dir: Path
            Path from where to load the data or save them if it does not exist.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    -----
    RxRx1 was created by Recursion Pharmaceuticals, adapted for domain generalization
    benchmarking by the WILDS team.
    Official website: https://wilds.stanford.edu/datasets/#rxrx1

    References
    ----------
    .. [1] M. Taylor et al. RxRx1: An Image Set for Cellular Morphological Variation Across Many
        Experimental Batches. ICLR 2019 AI for Social GoodWorkshop
    .. [2] P. W. Koh et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. ICML 2021
    """

    def __init__(self, config: RxRx1DatasetConfig):
        super().__init__()
        full_dataset = get_dataset(dataset="rxrx1", download=True, root_dir=str(config.save_dir))
        assert full_dataset is not None
        self.samples = full_dataset.get_subset(config.mode)
        self.n_classes = 1139

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        # WILDS subsets return (image, label, metadata)
        sample, label = self.samples[idx][:2]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
