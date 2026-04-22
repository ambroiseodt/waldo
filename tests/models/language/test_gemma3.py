r"""
Gemma3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from typing import Any, cast

import pytest
import torch
from torch.nn import ModuleList

from waldo.models.language.gemma3 import Gemma3, Gemma3BlockAdapter, Gemma3Config
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockGemma3Block,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestGemma3Config:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize(
        "model_size",
        [
            "1b",
            "4b",
            "12b",
            "27b",
        ],
    )
    def test_valid_size_combinations_accepted(self, model_size: str) -> None:
        r"""Valid model size should instantiate without errors."""
        cfg = Gemma3Config(model_size=model_size)
        assert cfg.model_size == model_size

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Gemma3Config(model_size="7b")

    def test_defaults(self) -> None:
        r"""Default config should be model_size as '1b' and device as 'cpu'."""
        cfg = Gemma3Config()
        assert cfg.model_size == "1b"
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------
def _make_adapter() -> tuple[Gemma3BlockAdapter, MockGemma3Block]:
    r"""Helper to instantiate Gemma3BlockAdapter."""
    block = MockGemma3Block()
    return Gemma3BlockAdapter(block), block


class TestGemma3BlockAdapter:
    r"""Tests for correct instantiation and default values."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default Gemma3BlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_pre_hooks_match_spec(self) -> None:
        r"""Gemma3BlockAdapter apply pre-hooks on 'rc1' and 'act' due to GeGLU design."""
        adapter, _ = _make_adapter()
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_module_instantiation(self) -> None:
        r"""Gemma3BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter()
        assert adapter.ln1 is block.input_layernorm
        assert adapter.mha is block.self_attn
        assert adapter.ln2 is block.pre_feedforward_layernorm
        assert adapter.rc1 is block.pre_feedforward_layernorm
        assert adapter.fc1 is block.mlp.gate_proj
        assert adapter.act is block.mlp.down_proj
        assert adapter.fc2 is block.mlp.down_proj
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestGemma3Model:
    r"""Tests for model class valid entries, correct instantiation, and default values."""

    def test_invalid_model_name_raises(self, mock_gemma3_model: Gemma3) -> None:
        r"""Invalid model name should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="is not valid"):
            mock_gemma3_model._load_model("google/gemma-3-fake-pt", torch.float32, "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_gemma3_model: Gemma3) -> None:
        r"""Eval mode should be set after loading the model."""
        cast(Any, mock_gemma3_model.model.eval).assert_called()

    def test_blocks_returns_module_list(self, mock_gemma3_model: Gemma3) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_gemma3_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_gemma3_model: Gemma3) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_gemma3_model.blocks) == N_LAYERS

    def test_block_adapter_is_gemma3_adapter(self, mock_gemma3_model: Gemma3) -> None:
        r"""Block adapter property should return a Gemma3BlockAdapter object."""
        assert mock_gemma3_model.block_adapter is Gemma3BlockAdapter

    def test_pooling_returns_n_e_shape(self, mock_gemma3_model: Gemma3) -> None:
        r"""Pooling method should return a tensor of dimension batch size x embedding dimension."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_gemma3_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_gemma3_model: Gemma3) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_gemma3_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_special_tokens(self, mock_gemma3_model: Gemma3, input_ids: torch.Tensor) -> None:
        r"""Pooling should exclude special tokens: setting it to large value should make this clear."""
        torch.manual_seed(42)
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        input_ids_copy = input_ids.clone()
        special_id = mock_gemma3_model.special_token_ids[0]
        input_ids_copy[:, 0] = special_id
        hidden[:, 0, :] = 100
        out_excl = mock_gemma3_model.pooling(
            hidden, input_ids=input_ids_copy, special_token_ids=mock_gemma3_model.special_token_ids
        )
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_encode_returns_correct_shape(
        self, mock_gemma3_model: Gemma3, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        r"""The internal _get_hidden_states should return a tensor of dimension batch x seq x hidden."""
        hidden_states = mock_gemma3_model._get_hidden_states(input_ids, attention_mask)
        assert hidden_states.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_get_hidden_states_pre_hook_capture(
        self, mock_gemma3_model: Gemma3, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> None:
        r"""Probing a pre-hook module (like 'rc1') should successfully return hidden states."""
        mock_gemma3_model.set_probe(layer=0, module="rc1")
        hidden_states = mock_gemma3_model._get_hidden_states(input_ids, attention_mask)
        assert hidden_states.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)

    def test_mteb_meta_name_format(self, mock_gemma3_model: Gemma3) -> None:
        r"""MTEB metadata name should correctly reflect the current probe state."""
        mock_gemma3_model.set_probe(layer=1, module="mha")
        meta = mock_gemma3_model.mteb_model_meta
        assert "layer_1_module_mha" in str(meta.name)
