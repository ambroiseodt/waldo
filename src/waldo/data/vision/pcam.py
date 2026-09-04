r"""
PCAM Dataset.

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
class PCAMDatasetConfig:
    r"""
    PCAM configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exit.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "pcam"
    mode: str = "train"
    transform: Compose | None = None

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.mode not in ["train", "test"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Options are 'train' and 'test'.")


# ------------------------------------------------------------------------------
#   Dataset
# ------------------------------------------------------------------------------
class PCAMDataset(Dataset):
    r"""
    PCAM dataset from [1]_.

    The PatchCamelyon dataset is a binary classification dataset with 327,680 color images (96px x 96px),
    extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label
    indicating presence of metastatic tissue.

    Parameters
    ----------
    config: configuration class with
        save_dir: str
            Path from where to load the data or save them if it does not exit.
        mode: str
            Mode. Options are "train" and "test".
        transform: Compose
            Transformation to apply to the images.

    Notes
    ----------
    PCAM was created by Bastiaan S. Veeling, Jasper Linmans, Jim Winkens, Taco Cohen, Max Welling.
    Official website: https://github.com/basveeling/pcam.

    Download
    --------
    Torchvision download can give an error because of Google Drive's rights. To avoid that, we set
    download=False and download the data manually with run the following command from the root of the repository.
    It will download the data in the default location (datasets/pcam) and extract it.
    ```bash
    $ bash scripts/
    ```
    For clipart and painting domains, run the following command;
    ```bash
    $ wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/domain.zip
    ```
    Move the downloaded files to the desired location, e.g, datasets/domainnet, and extract them with
    ```bash
    $ unzip domain.zip
    ```

    References
    ----------
    .. [1] B. S. Veeling et al. Rotation Equivariant CNNs for Digital Pathology. arXiv 2018
    """

    def __init__(self, config: PCAMDatasetConfig):
        super().__init__()
        split = config.mode
        dataset = torchvision.datasets.PCAM(
            root="./datasets/",
            split=split,
            download=False,
        )

        self.samples = dataset
        self.n_classes = 2

        # Recover transform
        self.transform = config.transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample, label = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def __repr__(self):
        return f"Dataset with {len(self.samples)} images."
