r"""
FMoW Dataset.

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
class FMoWDatasetConfig:
    r"""
    FMoW configuration file.

    Parameters
    ----------
    save_dir: str
        Path from where to load the data or save them if it does not exist.
    mode: str
        Mode. Options are "train" and "test".
    transform: Any
        Transformation to apply to the images.
    """

    save_dir: Path = DATASET_DIR / "fmow"
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
class FMoWDataset(Dataset):
    r"""
    FMoW (Functional Map of the World) dataset from [1]_, part of the WILDS
    benchmark [2]_.

    FMoW consists of RGB satellite images labeled with one of $62$ land use / building
    categories. The distribution shift occurs across both geographic region and time
    (image acquisition year), reflecting both spatial and temporal domain shift.

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
    FMoW was created by DIUx, adapted for domain generalization benchmarking by the
    WILDS team.
    Official website: https://wilds.stanford.edu/datasets/#fmow

    Downloading errors 
    -------------------
    Versions: python==3.12, wilds==2.0.0 and pandas >= 2.0.
    WILDS 2.0.0 parses FMoW timestamps with `pd.to_datetime()` without
    specifying a format. With newer pandas versions (e.g. pandas 2.3.3),
    this can fail because FMoW contains ISO-8601 timestamps with fractional
    seconds, e.g. `2011-02-07T02:48:56.643Z`.

    Fix:
    Manually edit WILDS' `fmow_dataset.py` and replace every occurrence of
    ```
    pd.to_datetime(self.metadata['timestamp'])
    ```
    with
    ```
    pd.to_datetime(self.metadata['timestamp'], format='ISO8601', utc=True)
    ```
    To locate the installed file:

    ```bash
    python -c "import wilds, os; print(os.path.join(os.path.dirname(wilds.__file__), 'datasets', 'fmow_dataset.py'))"
    ```
    Then edit that file (e.g. with VS Code):

    ```bash
    code "$(python -c "import wilds, os; print(os.path.join(os.path.dirname(wilds.__file__), 'datasets', 'fmow_dataset.py'))")"
    ```

    References
    ----------
    .. [1] G. Christie et al. Functional Map of the World. CVPR 2018
    .. [2] P. W. Koh et al. WILDS: A Benchmark of in-the-Wild Distribution Shifts. ICML 2021
    """

    def __init__(self, config: FMoWDatasetConfig):
        super().__init__()
        full_dataset = get_dataset(dataset="fmow", download=True, root_dir=str(config.save_dir))
        self.samples = full_dataset.get_subset(config.mode)
        self.n_classes = 62

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
