r"""
DTD Dataset.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass
from pathlib import Path

import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from ...infra import DATASET_DIR


# ------------------------------------------------------------------------------
#   Config
# ------------------------------------------------------------------------------
@dataclass
class DTDDatasetConfig:
    r"""
    DTD configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train", "val", and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "dtd"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are 'train', 'val' and 'test'.")


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
class DTDDataset(Dataset):
    r"""
    DTD dataset from [1]_.

    It consits of 5640 images of texture of 47 categories inspired from human perception.
    There are 120 images for each category. We make use of the default partition set at 1 following [2]_.
    Other values are possible, see the official documentation for details.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train", "val" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    ----------
    FGVCAircraft was created by Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi.
    Official website: https://www.robots.ox.ac.uk/~vgg/data/dtd/.


    References
    ----------
    .. [1] M. Cimpoi et al. Describing Textures in the Wild. In CVPR, 2014

    .. [2] T. Chen et al. A Simple Framework for Contrastive Learning of Visual Representations. In ICML 2020
    """

    def __init__(self, config: DTDDatasetConfig):
        super().__init__()
        split = config.mode
        dataset = torchvision.datasets.DTD(
            root=str(config.save_dir),
            split=split,
            partition=1,
            download=True,
        )

        # Recover images and corresponding labels
        self.data = dataset._image_files
        self.targets = dataset._labels
        self.loader = dataset.loader
        self.n_classes = 47

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.loader(self.data[idx])
        label = self.targets[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.data)} images."
