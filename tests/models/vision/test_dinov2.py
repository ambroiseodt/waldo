r"""
DINOv2 unit tests.

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

from waldo.models import Dinov2, Dinov2BlockAdapter, Dinov2Config
from waldo.models.transformer_block import TRANSFORMER_MODULES, SwiGLUGateProjection

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockDinov2Block,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestDinov2Config:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize("model_size", ["small", "base", "large", "giant"])
    def test_valid_model_sizes_accepted(self, model_size: str) -> None:
        r"""Valid model sizes should instantiate without errors."""
        cfg = Dinov2Config(model_size=model_size)
        assert cfg.model_size == model_size

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Dinov2Config(model_size="tiny")

    def test_defaults(self) -> None:
        r"""Default config should be model_size 'base' and device 'cpu'."""
        cfg = Dinov2Config()
        assert cfg.model_size == "base"
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------
def _make_adapter(gated: bool = False) -> tuple[Dinov2BlockAdapter, MockDinov2Block]:
    r"""Helper to instantiate Dinov3BlockAdapter."""
    block = MockDinov2Block(gated=gated)
    return Dinov2BlockAdapter(block), block


class TestDinov2BlockAdapter:
    r"""Tests for correct instantiation and module mapping."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default Aimv2BlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_pre_hooks_match_spec_standard(self) -> None:
        r"""Default Dinov2BlockAdapter apply pre-hook only on 'rc1' for standard MLP."""
        adapter, _ = _make_adapter(gated=False)
        assert adapter.pre_hooks == {"rc1"}

    def test_pre_hooks_match_spec_gated(self) -> None:
        r"""Gated Dinov2BlockAdapter apply pre-hooks on 'rc1' and 'act'."""
        adapter, _ = _make_adapter(gated=True)
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_module_instantiation(self) -> None:
        r"""Default Dinov2BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter(gated=False)
        assert adapter.ln1 is block.norm1
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.norm2
        assert adapter.rc1 is block.norm2
        assert adapter.fc1 is block.mlp.fc1
        assert adapter.act is block.mlp.activation
        assert adapter.fc2 is block.mlp.fc2
        assert adapter.rc2 is block

    def test_module_instantiation_gated_mlp(self) -> None:
        r"""Gated Dinov2BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter(gated=True)
        assert adapter.ln1 is block.norm1
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.norm2
        assert adapter.rc1 is block.norm2
        assert isinstance(adapter.fc1, SwiGLUGateProjection)
        assert adapter.fc1.linear is block.mlp.weights_in
        assert adapter.act is block.mlp.weights_out
        assert adapter.fc2 is block.mlp.weights_out
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestDinov2Model:
    r"""Tests for model class implementation."""

    def test_invalid_model_name_raises(self, mock_dinov2_model: Dinov2) -> None:
        r"""Invalid model name should raise a ValueError."""
        with pytest.raises(ValueError, match="not valid"):
            mock_dinov2_model._load_model("facebook/dinov2-micro", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_dinov2_model: Dinov2) -> None:
        r"""Eval mode should be set after loading the model."""
        cast(Any, mock_dinov2_model.model.eval).assert_called()

    def test_blocks_returns_module_list(self, mock_dinov2_model: Dinov2) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_dinov2_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_dinov2_model: Dinov2) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_dinov2_model.blocks) == N_LAYERS

    def test_block_adapter_is_dinov2_adapter(self, mock_dinov2_model: Dinov2) -> None:
        r"""Block adapter property should return a Dinov3BlockAdapter object."""
        assert mock_dinov2_model.block_adapter is Dinov2BlockAdapter

    def test_pooling_returns_n_e_shape(self, mock_dinov2_model: Dinov2) -> None:
        r"""Pooling method should return a tensor of dimension batch size x embedding dimension."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_dinov2_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_dinov2_model: Dinov2) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_dinov2_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_cls_token(self, mock_dinov2_model: Dinov2) -> None:
        r"""Pooling should exclude the CLS token: setting it to large value should make this clear."""
        torch.manual_seed(42)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        hidden[:, 0, :] = 100
        out_excl = mock_dinov2_model.pooling(hidden)
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_returns_all_keys(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""In default setting, embeddings dict should contain all the pairs of layers and modules."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be a tensor of dimension batch size x embedding dimension."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE), (
                f"{key}: expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {tensor.shape}"
            )

    def test_embeddings_are_on_cpu(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be on CPU."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

    def test_no_extra_keys_produced(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Embedding dict should contain only pairs of layer and module as keys."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Only specified single layer in get_embeddings should be returned."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers in get_embeddings should be returned."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Only specified single module in get_embeddings should be returned."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, modules=["mha"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha"}

    def test_module_filter_subset(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Only specified modules in get_embeddings should be returned."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, modules=["mha", "rc2"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha", "rc2"}

    def test_module_filter_all_layers(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for the specified module."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, modules=["mha"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_mha") for k in embeddings)

    def test_embeddings_rc1_capture(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""In standard model, rc1 module (captured via pre-hook) should be present in results."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, modules=["rc1"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_rc1") for k in embeddings)

    def test_gated_embeddings_rc1_act_capture(
        self, mock_dinov2_gated_model: Dinov2, pixel_values: torch.Tensor
    ) -> None:
        r"""In gated model, rc1 and act modules (captured via pre-hook) should be present in results."""
        for module_name in ["rc1", "act"]:
            embeddings = mock_dinov2_gated_model.get_embeddings(pixel_values, modules=[module_name])
            assert len(embeddings) == N_LAYERS
            assert all(k.endswith(f"_{module_name}") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Empty specified layers should returned empty dict."""
        assert mock_dinov2_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Empty specified modules should returned empty dict."""
        assert mock_dinov2_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_dinov2_model: Dinov2, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers and modules in get_embeddings should be returned."""
        embeddings = mock_dinov2_model.get_embeddings(pixel_values, layers=[2], modules=["mha", "rc2"])
        assert set(embeddings.keys()) == {"block_2_mha", "block_2_rc2"}
