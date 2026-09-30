r"""
Llama3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import patch

import pytest
import torch
from transformers import LlamaModel

from waldo.models import Llama3, Llama3BlockAdapter, Llama3Config

from .conftest import LANGUAGE_SPECS, MakeLanguageModel, WhitespaceTokenizer


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestLlama3Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be Llama-3.2-1B in bfloat16 on CPU."""
        config = Llama3Config()
        assert (config.model_size, config.dtype, config.device) == ("1B", torch.bfloat16, "cpu")

    @pytest.mark.parametrize("model_size", ["1b", "405B"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Llama3Config(model_size=model_size)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestLlama3:
    r"""Tests for the loading and structure of Llama3."""

    @pytest.mark.parametrize(
        "model_size, model_name",
        [
            ("1B", "meta-llama/Llama-3.2-1B"),
            ("3B", "meta-llama/Llama-3.2-3B"),
            ("8B", "meta-llama/Llama-3.1-8B"),
            ("70B", "meta-llama/Llama-3.1-70B"),
        ],
    )
    def test_loads_pretrained_model(self, model_size: str, model_name: str) -> None:
        r"""1B and 3B should load Llama 3.2, 8B and 70B Llama 3.1, with the requested dtype and device."""
        spec = LANGUAGE_SPECS["llama3"]
        with (
            patch.object(LlamaModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load,
            patch("waldo.models.language.llama3.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()),
        ):
            model = Llama3(Llama3Config(model_size=model_size))
        load.assert_called_once_with(model_name, dtype=torch.bfloat16, device_map="cpu")
        assert model.model_name == model_name

    def test_padding_token_is_eos_token(self, make_language_model: MakeLanguageModel) -> None:
        r"""Llama3 has no padding token: the EOS token should be used instead, for the tokenizer and the model."""
        model, _ = make_language_model("llama3")
        assert model.tokenizer.pad_token == model.tokenizer.eos_token
        assert model.model.config.pad_token_id == model.tokenizer.eos_token_id

    def test_invalid_model_name_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Loading a checkpoint that is not a pretrained Llama3 model should raise a ValueError."""
        model, _ = make_language_model("llama3")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="meta-llama/Llama-3.2-1B-Instruct", dtype=torch.float32, device="cpu")

    def test_structure(self, make_language_model: MakeLanguageModel) -> None:
        r"""Llama3 should use its block adapter."""
        model, hf_model = make_language_model("llama3")
        assert model.block_adapter is Llama3BlockAdapter
        assert model.blocks is hf_model.layers

    @pytest.mark.parametrize(
        "model_size, release_date, version", [("1B", "2024-09-25", "llama3_2"), ("8B", "2024-07-23", "llama3_1")]
    )
    def test_mteb_model_meta(
        self, make_language_model: MakeLanguageModel, model_size: str, release_date: str, version: str
    ) -> None:
        r"""MTEB metadata should give the release date and license of Llama 3.1 or 3.2."""
        model, _ = make_language_model("llama3", model_size=model_size)
        meta = model.mteb_model_meta
        assert str(meta.release_date) == release_date
        assert version in meta.license
