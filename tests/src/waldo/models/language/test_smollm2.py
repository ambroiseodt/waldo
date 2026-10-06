r"""
SmolLM2 unit tests.

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

from waldo.models import SmolLM2, SmolLM2BlockAdapter, SmolLM2Config

from .conftest import LANGUAGE_SPECS, MakeLanguageModel, WhitespaceTokenizer


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestSmolLM2Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be SmolLM2-135M in bfloat16 on CPU."""
        config = SmolLM2Config()
        assert (config.model_size, config.dtype, config.device) == ("135M", torch.bfloat16, "cpu")

    @pytest.mark.parametrize("model_size", ["135m", "1.7B-Instruct"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            SmolLM2Config(model_size=model_size)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestSmolLM2:
    r"""Tests for the loading and structure of SmolLM2."""

    @pytest.mark.parametrize("model_size", ["135M", "360M", "1.7B"])
    def test_loads_pretrained_model(self, model_size: str) -> None:
        r"""Each size should load the corresponding base checkpoint with the requested dtype and device."""
        spec = LANGUAGE_SPECS["smollm2"]
        with (
            patch.object(LlamaModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load,
            patch("waldo.models.language.smollm2.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()),
        ):
            model = SmolLM2(SmolLM2Config(model_size=model_size))
        model_name = f"HuggingFaceTB/SmolLM2-{model_size}"
        load.assert_called_once_with(model_name, dtype=torch.bfloat16, device_map="cpu")
        assert model.model_name == model_name

    def test_padding_token_is_eos_token(self, make_language_model: MakeLanguageModel) -> None:
        r"""SmolLM2 has no padding token: the EOS token should be used instead, for the tokenizer and the model."""
        model, _ = make_language_model("smollm2")
        assert model.tokenizer.pad_token == model.tokenizer.eos_token
        assert model.model.config.pad_token_id == model.tokenizer.eos_token_id

    def test_invalid_model_name_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Loading a checkpoint that is not a base SmolLM2 model should raise a ValueError."""
        model, _ = make_language_model("smollm2")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float32, device="cpu")

    def test_structure(self, make_language_model: MakeLanguageModel) -> None:
        r"""SmolLM2 should use its block adapter."""
        model, hf_model = make_language_model("smollm2")
        assert model.block_adapter is SmolLM2BlockAdapter
        assert model.blocks is hf_model.layers

    def test_mteb_model_meta(self, make_language_model: MakeLanguageModel) -> None:
        r"""MTEB metadata should give the release date and license of SmolLM2."""
        meta = make_language_model("smollm2")[0].mteb_model_meta
        assert (str(meta.release_date), meta.license) == ("2024-10-31", "apache-2.0")
