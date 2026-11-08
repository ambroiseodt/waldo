r"""
Gemma3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import Gemma3Config as HFGemma3Config
from transformers import Gemma3Model, Gemma3TextConfig, Gemma3TextModel

from waldo.models import Gemma3, Gemma3BlockAdapter, Gemma3Config

from .conftest import TINY, MakeLanguageModel, WhitespaceTokenizer


def _load(model_size: str) -> tuple[Gemma3, dict]:
    r"""Load Gemma3 with tiny text-only and multimodal models, and return the mocks of both loaders."""
    torch.manual_seed(0)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, patch_size=14)
    hf_config: dict[str, Any] = dict(text_config=TINY, vision_config=vision, mm_tokens_per_image=4)
    multimodal = Gemma3Model(HFGemma3Config(**hf_config))
    text_config = multimodal.config.text_config
    assert isinstance(text_config, Gemma3TextConfig)
    with (
        patch.object(Gemma3TextModel, "from_pretrained", return_value=Gemma3TextModel(text_config)) as text,
        patch.object(Gemma3Model, "from_pretrained", return_value=multimodal) as mm,
        patch("waldo.models.language.gemma3.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()) as tok,
    ):
        model = Gemma3(Gemma3Config(model_size=model_size))
    return model, {"text": text, "multimodal": mm, "tokenizer": tok, "language_model": multimodal.language_model}


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestGemma3Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be Gemma3-1b in bfloat16 on CPU."""
        config = Gemma3Config()
        assert (config.model_size, config.dtype, config.device) == ("1b", torch.bfloat16, "cpu")

    @pytest.mark.parametrize("model_size", ["2b", "1B", "270m"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Gemma3Config(model_size=model_size)

    def test_dtype_from_configuration_file(self, make_language_model: MakeLanguageModel) -> None:
        r"""The dtype given as a string in configuration files should be converted to a torch dtype."""
        model, _ = make_language_model("gemma3", dtype="float32")
        assert model.dtype is torch.float32


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestGemma3:
    r"""Tests for the loading and structure of Gemma3."""

    def test_1b_loads_text_model(self) -> None:
        r"""Gemma3-1b is text-only and should be loaded with Gemma3TextModel."""
        model, mocks = _load("1b")
        mocks["text"].assert_called_once_with("google/gemma-3-1b-pt", dtype=torch.bfloat16, device_map="cpu")
        mocks["multimodal"].assert_not_called()
        mocks["tokenizer"].assert_called_once_with("google/gemma-3-1b-pt")
        assert isinstance(model.model, Gemma3TextModel)

    @pytest.mark.parametrize("model_size", ["4b", "12b", "27b"])
    def test_multimodal_sizes_load_language_model(self, model_size: str) -> None:
        r"""Larger sizes are multimodal: they should be loaded with Gemma3Model, keeping only the language model."""
        model, mocks = _load(model_size)
        model_name = f"google/gemma-3-{model_size}-pt"
        mocks["multimodal"].assert_called_once_with(model_name, dtype=torch.bfloat16, device_map="cpu")
        mocks["text"].assert_not_called()
        assert model.model is mocks["language_model"]
        assert model.model_name == model_name

    def test_invalid_model_name_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Loading a checkpoint that is not a pretrained Gemma3 model should raise a ValueError."""
        model, _ = make_language_model("gemma3")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="google/gemma-3-1b-it", dtype=torch.float32, device="cpu")

    def test_structure(self, make_language_model: MakeLanguageModel) -> None:
        r"""Gemma3 should use its block adapter and exclude the BOS, EOS and padding tokens."""
        model, hf_model = make_language_model("gemma3")
        assert model.block_adapter is Gemma3BlockAdapter
        assert model.blocks is hf_model.layers
        assert model.special_token_ids == [1, 2, 0]

    def test_mteb_model_meta(self, make_language_model: MakeLanguageModel) -> None:
        r"""MTEB metadata should give the Gemma license."""
        model, _ = make_language_model("gemma3")
        assert model.mteb_model_meta.license == "gemma"
