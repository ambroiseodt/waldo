r"""
DINOv3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch
from torch.nn import ModuleList

from waldo.models import Dinov3, Dinov3BlockAdapter, Dinov3Config
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockDinov3Block,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestDinov3Config:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize(
        "model_size",
        [
            "vits16",
            "vitb16",
            "vitl16",
            "vit7b16",
            "vits16plus",
            "vith16plus",
        ],
    )
    def test_valid_size_combinations_accepted(self, model_size: str) -> None:
        r"""Valid model size should instantiate without errors."""
        cfg = Dinov3Config(model_size=model_size)
        assert cfg.model_size == model_size

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Dinov3Config(model_size="vit-giant")

    def test_defaults(self) -> None:
        r"""Default config should be model_size as 'vitb16' and device as 'cpu'."""
        cfg = Dinov3Config()
        assert cfg.model_size == "vitb16"
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------


def _make_adapter(gated: bool = False) -> tuple[Dinov3BlockAdapter, MockDinov3Block]:
    r"""Helper to instantiate Dinov3BlockAdapter."""
    block = MockDinov3Block(gated=gated)
    return Dinov3BlockAdapter(block), block


class TestDinov3BlockAdapter:
    r"""Tests for correct instantiation and default values."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default Dinov3BlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_only_rc1_in_pre_hooks_standard(self) -> None:
        r"""Default Dinov3BlockAdapter apply pre-hook only on 'rc1' for standard MLP."""
        adapter, _ = _make_adapter(gated=False)
        assert adapter.pre_hooks == {"rc1"}

    def test_rc1_and_act_in_pre_hooks_gated(self) -> None:
        r"""Gated Dinov3BlockAdapter apply pre-hooks on 'rc1' and 'act'."""
        adapter, _ = _make_adapter(gated=True)
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_module_instantiation(self) -> None:
        r"""Default Dinov3BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter(gated=False)
        assert adapter.ln1 is block.norm1
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.norm2
        assert adapter.rc1 is block.norm2
        assert adapter.fc1 is block.mlp.up_proj
        assert adapter.act is block.mlp.act_fn
        assert adapter.fc2 is block.mlp.down_proj
        assert adapter.rc2 is block

    def test_module_instantiation_gated_mlp(self) -> None:
        r"""Gated Dinov3BlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter(gated=True)
        assert adapter.ln1 is block.norm1
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.norm2
        assert adapter.rc1 is block.norm2
        assert adapter.fc1 is block.mlp.gate_proj
        assert adapter.act is block.mlp.down_proj
        assert adapter.fc2 is block.mlp.down_proj
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestDinov3Model:
    r"""Tests for model class valid entries, correct instantiation, and default values."""

    def test_invalid_model_name_raises(self, mock_dinov3_model: Dinov3) -> None:
        r"""Invalid model name should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="is not valid"):
            mock_dinov3_model._load_model("facebook/dinov3-fake-pretrain", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_dinov3_model: Dinov3) -> None:
        r"""Eval mode should be set after loading the model."""
        mock_dinov3_model.model.eval.assert_called()

    def test_blocks_returns_module_list(self, mock_dinov3_model: Dinov3) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_dinov3_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_dinov3_model: Dinov3) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_dinov3_model.blocks) == N_LAYERS

    def test_block_adapter_is_dinov3_adapter(self, mock_dinov3_model: Dinov3) -> None:
        r"""Block adapter property should return a Dinov3BlockAdapter object."""
        assert mock_dinov3_model.block_adapter is Dinov3BlockAdapter

    def test_pooling_returns_n_e_shape(self, mock_dinov3_model: Dinov3) -> None:
        r"""Pooling method should return a tensor of dimension batch size x embedding dimension."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_dinov3_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_dinov3_model: Dinov3) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_dinov3_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_cls_token(self, mock_dinov3_model: Dinov3) -> None:
        r"""Pooling should exclude the CLS token: setting it to large value should make this clear."""
        torch.manual_seed(42)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        hidden[:, 0, :] = 100.0
        out_excl = mock_dinov3_model.pooling(hidden)
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_returns_all_keys(self, mock_dinov3_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""In default setting, embeddings dict should contain all the pairs of layers and modules."""
        embeddings = mock_dinov3_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_dinov3_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be a tensor of dimension batch size x embedding dimension."""
        embeddings = mock_dinov3_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE), (
                f"{key}: expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {tensor.shape}"
            )

    def test_embeddings_are_on_cpu(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be on CPU."""
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

    def test_no_extra_keys_produced(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Embedding dict should contain only pairs of layer and module as keys."""
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Only specified single layer in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_dinov3_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers in get_embeddings should be returned."""
        embeddings = mock_dinov3_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Only specified single module in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["mha"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha"}

    def test_module_filter_subset(self, mock_dinov3_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Only specified modules in get_embeddings should be returned."""
        embeddings = mock_dinov3_model.get_embeddings(pixel_values, modules=["mha", "rc2"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha", "rc2"}

    def test_module_filter_all_layers(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for the specified module."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["mha"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_mha") for k in embeddings)

    def test_rc1_filter_all_layer(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for rc1 even as a pre-hook module."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["rc1"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_rc1") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Empty specified layers should returned empty dict."""
        assert mock_vit_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Empty specified modules should returned empty dict."""
        assert mock_vit_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_vit_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers and modules in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[2], modules=["mha", "rc2"])
        assert set(embeddings.keys()) == {"block_2_mha", "block_2_rc2"}

    def test_gated_embeddings_act_capture(self, mock_dinov3_gated_model: Dinov3, pixel_values: torch.Tensor) -> None:
        r"""In gated model, act module (captured via pre-hook) should be present in results."""
        embeddings = mock_dinov3_gated_model.get_embeddings(pixel_values, modules=["act"])
        assert all(k.endswith("_act") for k in embeddings)
        assert len(embeddings) == N_LAYERS
