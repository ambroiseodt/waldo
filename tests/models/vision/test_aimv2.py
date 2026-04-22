r"""
AIMv2 unit tests.

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

from waldo.models import Aimv2, Aimv2BlockAdapter, Aimv2Config
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockAimv2Block,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestAimv2Config:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize(
        "model_size,resolution",
        [
            ("large", 224),
            ("huge", 336),
            ("1B", 448),
            ("3B", 224),
        ],
    )
    def test_valid_size_res_combinations_accepted(self, model_size: str, resolution: int) -> None:
        r"""Valid model size and resolution pairs should instantiate without errors."""
        cfg = Aimv2Config(model_size=model_size, resolution=resolution)
        assert cfg.model_size == model_size
        assert cfg.resolution == resolution

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Aimv2Config(model_size="small", resolution=224)

    def test_invalid_resolution_raises(self) -> None:
        r"""Invalid resolution should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            Aimv2Config(model_size="large", resolution=128)

    def test_defaults(self) -> None:
        r"""Default config should be model_size as 'large', resolution as 224, and device as 'cpu'."""
        cfg = Aimv2Config()
        assert cfg.model_size == "large"
        assert cfg.resolution == 224
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------
def _make_adapter() -> tuple[Aimv2BlockAdapter, MockAimv2Block]:
    r"""Helper to instantiate Aimv2BlockAdapter."""
    block = MockAimv2Block()
    return Aimv2BlockAdapter(block), block


class TestAimv2BlockAdapter:
    r"""Tests for correct instantiation and default values."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default Aimv2BlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_pre_hooks_match_spec(self) -> None:
        r"""Aimv2BlockAdapter apply pre-hook on 'rc1' and 'act'."""
        adapter, _ = _make_adapter()
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_module_instantiation(self) -> None:
        r"""Default Aimv2BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter()
        assert adapter.ln1 is block.rms_norm1
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.rms_norm2
        assert adapter.rc1 is block.rms_norm2
        assert adapter.fc1 is block.ffn.gate_proj
        assert adapter.act is block.ffn.down_proj
        assert adapter.fc2 is block.ffn.down_proj
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestAimv2Model:
    r"""Tests for model class valid entries, correct instantiation, and default values."""

    def test_invalid_model_name_raises(self, mock_aimv2_model: Aimv2) -> None:
        r"""Invalid model name should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="not valid"):
            mock_aimv2_model._load_model("apple/aimv2-tiny-patch14-224", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_aimv2_model: Aimv2) -> None:
        r"""Eval mode should be set after loading the model."""
        cast(Any, mock_aimv2_model.model.eval).assert_called()

    def test_blocks_returns_module_list(self, mock_aimv2_model: Aimv2) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_aimv2_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_aimv2_model: Aimv2) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_aimv2_model.blocks) == N_LAYERS

    def test_block_adapter_is_aimv2_adapter(self, mock_aimv2_model: Aimv2) -> None:
        r"""Block adapter property should return the Aimv2BlockAdapter class."""
        assert mock_aimv2_model.block_adapter is Aimv2BlockAdapter

    def test_pooling_returns_n_e_shape(self, mock_aimv2_model: Aimv2) -> None:
        r"""Pooling method should return a tensor of dimension batch size x embedding dimension."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_aimv2_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_aimv2_model: Aimv2) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_aimv2_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_includes_all_tokens(self, mock_aimv2_model: Aimv2) -> None:
        r"""In AIMv2, pooling should include all tokens as there is no CLS token."""
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        out = mock_aimv2_model.pooling(hidden)
        expected = hidden.mean(dim=1).cpu()
        assert torch.allclose(out, expected)

    def test_returns_all_keys(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""In default setting, embeddings dict should contain all pairs of layers and modules."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be a tensor of dimension batch size x embedding dimension."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values)
        for _, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_embeddings_are_on_cpu(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be on CPU."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values)
        for _, tensor in embeddings.items():
            assert tensor.device.type == "cpu"

    def test_no_extra_keys_produced(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Embedding dict should contain only pairs of layer and module as keys."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Only specified single layer in get_embeddings should be returned."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers in get_embeddings should be returned."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Only specified single module in get_embeddings should be returned."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, modules=["mha"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha"}

    def test_module_filter_subset(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Only specified modules in get_embeddings should be returned."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, modules=["mha", "rc2"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha", "rc2"}

    def test_module_filter_all_layers(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for the specified module."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, modules=["mha"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_mha") for k in embeddings)

    def test_embeddings_rc1_act_capture(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""In standard model, rc1 and act modules (captured via pre-hook) should be present in results."""
        for module_name in ["rc1", "act"]:
            embeddings = mock_aimv2_model.get_embeddings(pixel_values, modules=[module_name])
            assert len(embeddings) == N_LAYERS
            assert all(k.endswith(f"_{module_name}") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Empty specified layers should returned empty dict."""
        assert mock_aimv2_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Empty specified modules should returned empty dict."""
        assert mock_aimv2_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_aimv2_model: Aimv2, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers and modules in get_embeddings should be returned."""
        embeddings = mock_aimv2_model.get_embeddings(pixel_values, layers=[1], modules=["mha", "act"])
        assert set(embeddings.keys()) == {"block_1_mha", "block_1_act"}
