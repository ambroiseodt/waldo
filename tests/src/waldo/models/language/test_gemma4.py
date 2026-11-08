r"""
Gemma4 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import Gemma4Config as HFGemma4Config
from transformers import Gemma4ForConditionalGeneration, Gemma4Model

from waldo.models import TRANSFORMER_MODULES, Gemma4, Gemma4BlockAdapter, Gemma4Config

from ..conftest import assert_block_identities, capture_block_inputs
from .conftest import SENTENCES, TINY, TINY_GEMMA4, MakeLanguageModel, WhitespaceTokenizer

# Gemma4 31B: no per-layer token embeddings, keys equal to values in global attention layers
TINY_GEMMA4_DENSE: dict[str, Any] = dict(
    TINY,
    layer_types=["sliding_attention", "full_attention", "full_attention"],
    sliding_window=4,
    hidden_size_per_layer_input=0,
    attention_k_eq_v=True,
    num_global_key_value_heads=1,
    global_head_dim=16,
)


def _multimodal(text_config: dict[str, Any] = TINY_GEMMA4) -> Gemma4ForConditionalGeneration:
    r"""Tiny multimodal Gemma4 model without vision and audio towers."""
    torch.manual_seed(0)
    config_kwargs: dict[str, Any] = dict(text_config=text_config, vision_config=None, audio_config=None)
    config = HFGemma4Config(**config_kwargs)
    return Gemma4ForConditionalGeneration(config).eval()


def _load(model_size: str, text_config: dict[str, Any] = TINY_GEMMA4) -> tuple[Gemma4, dict]:
    r"""Load Gemma4 with a tiny multimodal model, and return the mocks of the loaders."""
    multimodal = _multimodal(text_config)
    with (
        patch.object(Gemma4ForConditionalGeneration, "from_pretrained", return_value=multimodal) as load,
        patch("waldo.models.language.gemma4.AutoTokenizer.from_pretrained", return_value=WhitespaceTokenizer()) as tok,
    ):
        model = Gemma4(Gemma4Config(model_size=model_size, dtype=torch.float32))
    return model, {"multimodal": load, "tokenizer": tok, "language_model": multimodal.model.language_model}


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestGemma4Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be Gemma4-E2B in bfloat16 on CPU."""
        config = Gemma4Config()
        assert (config.model_size, config.dtype, config.device) == ("E2B", torch.bfloat16, "cpu")

    @pytest.mark.parametrize("model_size", ["e2b", "12B", "26B-A4B"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError, including 12B and the MoE model."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Gemma4Config(model_size=model_size)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestGemma4:
    r"""Tests for the loading and structure of Gemma4."""

    @pytest.mark.parametrize("model_size", ["E2B", "E4B", "31B"])
    def test_loads_language_model(self, model_size: str) -> None:
        r"""Each size is multimodal: it should be loaded with its checkpoint class, keeping only the language model."""
        model, mocks = _load(model_size)
        model_name = f"google/gemma-4-{model_size}"
        mocks["multimodal"].assert_called_once_with(model_name, dtype=torch.float32, device_map="cpu")
        mocks["tokenizer"].assert_called_once_with(model_name)
        assert model.model is mocks["language_model"]
        assert model.model_name == model_name

    def test_language_model_weights_are_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        r"""
        Loading a full Gemma4 checkpoint with its class should load every weight of the language model, whereas
        Gemma4Model would ignore the "model." prefix of the checkpoint and initialize them randomly.
        """
        # Allow from_pretrained for a tiny checkpoint saved locally (HF_HUB_OFFLINE stays set)
        monkeypatch.undo()
        full = _multimodal()
        full.save_pretrained(tmp_path)
        result = Gemma4ForConditionalGeneration.from_pretrained(tmp_path, output_loading_info=True)
        assert isinstance(result, tuple)
        loaded, info = result
        assert not info["missing_keys"]
        expected = full.model.language_model.state_dict()
        for name, parameter in loaded.model.language_model.state_dict().items():
            torch.testing.assert_close(parameter, expected[name])
        result = Gemma4Model.from_pretrained(tmp_path, output_loading_info=True)
        assert isinstance(result, tuple)
        _, info = result
        assert any(key.startswith("language_model.") for key in info["missing_keys"])

    def test_text_model_matches_multimodal_model(self) -> None:
        r"""The language model alone should give the hidden states of the multimodal model on text inputs."""
        model, _ = _load("E2B")
        reference = _multimodal().model
        encoding = WhitespaceTokenizer()(SENTENCES)
        with torch.inference_mode():
            expected = reference(**encoding).last_hidden_state
            output = model.model(**encoding).last_hidden_state
        torch.testing.assert_close(output, expected)

    def test_dense_block_identities(self) -> None:
        r"""Without per-layer token embeddings (31B), rc2 should be the scaled sum of rc1 and the FFN output."""
        model, _ = _load("31B", text_config=TINY_GEMMA4_DENSE)
        for block in model.blocks:
            block.layer_scalar.fill_(0.5)
        encoding = WhitespaceTokenizer()(SENTENCES)
        outputs = {}
        with torch.inference_mode(), capture_block_inputs(model.blocks) as inputs:
            for module in TRANSFORMER_MODULES:
                model.set_probing(layer=1, module=module)
                outputs[module] = model._get_hidden_states(**encoding)
            assert_block_identities(model.blocks[1], model.block_adapter(model.blocks[1]), inputs[1], outputs)

    def test_invalid_model_name_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Loading a checkpoint that is not a pretrained Gemma4 model should raise a ValueError."""
        model, _ = make_language_model("gemma4")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="google/gemma-4-E2B-it", dtype=torch.float32, device="cpu")

    def test_structure(self, make_language_model: MakeLanguageModel) -> None:
        r"""Gemma4 should use its block adapter and exclude the BOS, EOS and padding tokens."""
        model, hf_model = make_language_model("gemma4")
        assert model.block_adapter is Gemma4BlockAdapter
        assert model.blocks is hf_model.layers
        assert model.special_token_ids == [1, 2, 0]

    @pytest.mark.parametrize("model_size, release_date", [("E2B", "2026-03-02"), ("31B", "2026-03-12")])
    def test_mteb_model_meta(self, model_size: str, release_date: str) -> None:
        r"""MTEB metadata should give the release date of the size and the Apache 2.0 license."""
        meta = _load(model_size)[0].mteb_model_meta
        assert (str(meta.release_date), meta.license) == (release_date, "apache-2.0")
