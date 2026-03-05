from abc import ABC, abstractmethod

import numpy as np


class BaseLanguageModel(ABC):
    r"""Abstract base class for language models."""

    @abstractmethod
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray:
        """Encode a list of sentences into embeddings."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...
