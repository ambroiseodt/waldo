r"""
Model factory unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from typing import Any
from unittest.mock import patch

import pytest
import torch.nn as nn

from waldo.models import Aimv2, Clip, Dinov2, Dinov3, Gemma3, Gemma4, Llama3, Qwen3, ViT, build_model

from .language.conftest import LANGUAGE_SPECS, WhitespaceTokenizer
from .vision.conftest import VISION_SPECS


def _build(name: str, return_config: bool = False, **config: Any) -> nn.Module | tuple[nn.Module, dict]:
    r"""Call build_model with the tiny model of a spec in place of the pretrained one."""
    spec = {**VISION_SPECS, **LANGUAGE_SPECS}[name]
    with (
        patch.object(spec.hf_class, "from_pretrained", return_value=spec.hf_class(spec.hf_config)),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()),
    ):
        return build_model({**spec.config, **config}, return_config=return_config)


# -------------------------------------------------------------------
# Build model
# -------------------------------------------------------------------
class TestBuildModel:
    r"""Tests for the dispatch of build_model."""

    @pytest.mark.parametrize(
        "name, model_type",
        [
            ("vit", ViT),
            ("dinov2", Dinov2),
            ("dinov3", Dinov3),
            ("aimv2", Aimv2),
            ("clip", Clip),
            ("gemma3", Gemma3),
            ("gemma4", Gemma4),
            ("llama3", Llama3),
            ("qwen3", Qwen3),
        ],
    )
    def test_dispatch(self, name: str, model_type: type) -> None:
        r"""Each model name should build the corresponding model."""
        assert type(_build(name)) is model_type

    def test_model_name_is_case_insensitive(self) -> None:
        r"""Model names should be matched regardless of the case."""
        assert type(_build("vit", model_name="ViT")) is ViT

    def test_returns_config(self) -> None:
        r"""With return_config, the full config of the model should also be returned as a dictionary."""
        result = _build("vit", model_size="large", return_config=True)
        assert isinstance(result, tuple)
        model, config = result
        assert type(model) is ViT
        assert config == {"model_size": "large", "patch_size": 16, "pooling": "content", "device": "cpu"}

    def test_unknown_model_raises(self) -> None:
        r"""Unknown model names should raise a ValueError."""
        with pytest.raises(ValueError, match="not found"):
            build_model({"model_name": "unexisting"})
