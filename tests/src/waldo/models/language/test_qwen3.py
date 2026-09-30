r"""
Qwen3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Model

from waldo.models import Qwen3, Qwen3BlockAdapter, Qwen3Config

from .conftest import LANGUAGE_SPECS, MakeLanguageModel, WhitespaceTokenizer

POST_TRAINED_SIZES = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"]
PRETRAINED_SIZES = ["0.6B-Base", "1.7B-Base", "4B-Base", "8B-Base", "14B-Base"]


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestQwen3Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be the post-trained Qwen3-0.6B in bfloat16 on CPU."""
        config = Qwen3Config()
        assert (config.model_size, config.dtype, config.device) == ("0.6B", torch.bfloat16, "cpu")

    @pytest.mark.parametrize("model_size", ["32B-Base", "0.6b-base", "0.6B-Instruct"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Sizes without checkpoint, e.g. no pretrained 32B model, should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Qwen3Config(model_size=model_size)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestQwen3:
    r"""Tests for the loading and structure of Qwen3."""

    @pytest.mark.parametrize("model_size", POST_TRAINED_SIZES + PRETRAINED_SIZES)
    def test_loads_pretrained_model(self, model_size: str) -> None:
        r"""Post-trained and pretrained (Base) sizes should load the corresponding checkpoint and tokenizer."""
        spec = LANGUAGE_SPECS["qwen3"]
        with (
            patch.object(Qwen3Model, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load,
            patch(
                "waldo.models.language.qwen3.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()
            ) as tokenizer,
        ):
            model = Qwen3(Qwen3Config(model_size=model_size))
        load.assert_called_once_with(f"Qwen/Qwen3-{model_size}", dtype=torch.bfloat16, device_map="cpu")
        tokenizer.assert_called_once_with(f"Qwen/Qwen3-{model_size}")
        assert model.model_name == f"Qwen/Qwen3-{model_size}"

    def test_invalid_model_name_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Loading a checkpoint that is not a Qwen3 model should raise a ValueError."""
        model, _ = make_language_model("qwen3")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="Qwen/Qwen3-32B-Base", dtype=torch.float32, device="cpu")

    def test_structure(self, make_language_model: MakeLanguageModel) -> None:
        r"""Qwen3 should use its block adapter and recover mha after the normalization of queries and keys."""
        model, hf_model = make_language_model("qwen3")
        adapter = Qwen3BlockAdapter(model.blocks[0])
        assert model.block_adapter is Qwen3BlockAdapter
        assert model.blocks is hf_model.layers
        assert adapter.mha is model.blocks[0].self_attn

    def test_mteb_model_meta(self, make_language_model: MakeLanguageModel) -> None:
        r"""MTEB metadata should give the Apache 2.0 license of Qwen3."""
        model, _ = make_language_model("qwen3")
        assert model.mteb_model_meta.license == "apache-2.0"
