r"""
Shared fixtures for all unit tests.

Tests must never download or load pretrained models: they use tiny models built from their configuration. Loading a
pretrained model, e.g. through an unexpected code path, raises an error instead of downloading and allocating
billions of parameters.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import os

# Set before HuggingFace libraries are imported, since they read it at import time
os.environ["HF_HUB_OFFLINE"] = "1"

import pytest
from transformers import AutoTokenizer, PreTrainedModel


def _refuse_pretrained(*args: object, **kwargs: object) -> None:
    raise RuntimeError("Tests must not load pretrained models or tokenizers: patch from_pretrained explicitly.")


@pytest.fixture(autouse=True)
def no_pretrained_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Make every from_pretrained call fail unless a test patches it explicitly with a tiny model."""
    monkeypatch.setattr(PreTrainedModel, "from_pretrained", classmethod(_refuse_pretrained))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _refuse_pretrained)
