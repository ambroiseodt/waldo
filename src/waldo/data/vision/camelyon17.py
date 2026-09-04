r"""
Camelyon17 Dataset.

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
class Camelyon17DatasetConfig:
    r"""
    Camelyon17 configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exist.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "camelyon17"
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
class Camelyon17Dataset(Dataset):
    r"""
    Camelyon17 dataset from [1]_, part of the WILDS benchmark [2]_.

    Camelyon17 consists of $96 \times 96$ histopathology image patches from lymph node
    sections, collected across 5 different hospitals. Each patch is labeled with a
    binary indicator of tumor tissue presence, making this a 2-way classification
    task. The distribution shift occurs across hospitals, which differ in staining
    protocols and scanning equipment.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exist.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    ----------
    Camelyon17 was created by the CAMELYON challenge organizers, adapted for domain
    generalization benchmarking by the WILDS team.
    Official website: https://wilds.stanford.edu/datasets/#camelyon17

    References
    ----------
    .. [1] P. Bandi et al. From Detection of Individual Metastases to Classification of Lymph
        Node Status at the Patient Level: the CAMELYON17 Challenge. IEEE TMI 2018
    .. [2] P. W. Koh et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. ICML 2021
    """

    def __init__(self, config: Camelyon17DatasetConfig):
        super().__init__()
        full_dataset = get_dataset(dataset="camelyon17", download=True, root_dir=str(config.save_dir))
        self.samples = full_dataset.get_subset(config.mode)
        self.n_classes = 2

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample, label, _metadata = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
