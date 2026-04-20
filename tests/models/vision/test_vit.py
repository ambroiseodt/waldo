r"""
ViT unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch
from torch.nn import ModuleList

from waldo.models import ViT, ViTBlockAdapter, ViTConfig
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockViTBlock,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestViTConfig:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize(
        "model_size,patch_size",
        [
            ("base", 16),
            ("base", 32),
            ("large", 16),
            ("large", 32),
            ("huge", 14),
        ],
    )
    def test_valid_size_patch_combinations_accepted(self, model_size: str, patch_size: int) -> None:
        r"""Valid model size and patch size pairs should instantite without errors."""
        cfg = ViTConfig(model_size=model_size, patch_size=patch_size)
        assert cfg.model_size == model_size
        assert cfg.patch_size == patch_size

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid model size"):
            ViTConfig(model_size="xlarge", patch_size=16)

    def test_invalid_patch_size_raises(self) -> None:
        r"""Invalid patch size should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="Invalid patch size"):
            ViTConfig(model_size="base", patch_size=8)

    def test_defaults(self) -> None:
        r"""Default config should be model_size as 'base', patch_size as 16 and device as 'cpu'."""
        cfg = ViTConfig()
        assert cfg.model_size == "base"
        assert cfg.patch_size == 16
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------


def _make_adapter() -> tuple[ViTBlockAdapter, MockViTBlock]:
    r"""Helper to instantiate ViTBlockAdapter."""
    block = MockViTBlock()
    return ViTBlockAdapter(block), block


class TestViTBlockAdapter:
    r"""Tests for correct instantiation and default values."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default ViTBlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_only_rc1_in_pre_hooks(self) -> None:
        r"""Default ViTBlockAdapter apply pre-hook only on 'rc1'."""
        adapter, _ = _make_adapter()
        assert adapter.pre_hooks == {"rc1"}

    def test_module_instantiation(self) -> None:
        r"""Default ViTBlockAdapter should correctly each transformer module."""
        adapter, block = _make_adapter()
        assert adapter.ln1 is not adapter.ln2
        assert adapter.ln1 is block.layernorm_before
        assert adapter.mha is block.attention
        assert adapter.ln2 is block.layernorm_after
        assert adapter.rc1 is block.layernorm_after
        assert adapter.fc1 is block.intermediate.dense
        assert adapter.act is block.intermediate.intermediate_act_fn
        assert adapter.fc2 is block.output.dense
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestViTModel:
    r"""Tests for model class valid entries, correct instantiation, and default values."""

    def test_invalid_model_name_raises(self, mock_vit_model: ViT) -> None:
        r"""Invalid model name should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="not valid"):
            mock_vit_model._load_model("google/vit-giant-patch16-224-in21k", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_vit_model: ViT) -> None:
        r"""Eval mode should be set after loading the model."""
        mock_vit_model.model.eval.assert_called()

    def test_blocks_returns_module_list(self, mock_vit_model: ViT) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_vit_model.blocks, ModuleList)

    def test_blocks_length_equals_n_layers(self, mock_vit_model: ViT) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_vit_model.blocks) == N_LAYERS

    def test_block_adapter_is_vit_adapter(self, mock_vit_model: ViT) -> None:
        r"""Block adapter property should return a ViTBlockAdapter object."""
        assert mock_vit_model.block_adapter is ViTBlockAdapter

    def test_embedding_dim_equals_hidden_size(self, mock_vit_model: ViT) -> None:
        r"""Embedding dimension should match the model's hidden size."""
        assert mock_vit_model.embedding_dim == HIDDEN_SIZE

    def test_n_layers_equals_n_layers_constant(self, mock_vit_model: ViT) -> None:
        r"""n_layer property should match the number of layers in the model."""
        assert mock_vit_model.n_layers == N_LAYERS

    def test_pooling_returns_n_e_shape(self, mock_vit_model: ViT) -> None:
        r"""Pooling method should return a tensor of dimension batch size x embedding dimension."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_vit_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)

    def test_pooling_result_is_on_cpu(self, mock_vit_model: ViT) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_vit_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_cls_token(self, mock_vit_model: ViT) -> None:
        r"""Pooling should exclude the CLS token: setting it to large value should make this clear."""
        torch.manual_seed(42)
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        hidden[:, 0, :] = 100.0
        out_excl = mock_vit_model.pooling(hidden)
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_returns_all_keys(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""In default setting, embeddings dict should contain all the pairss of layers and modules."""
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""
        Each embedding should be a tensor of dimension batch size x embedding dimension. In the mock setting,
        The feedforward keeps the hidden size fixed but in practice, it increases (typically 4 times larger).
        """
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE), (
                f"{key}: expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {tensor.shape}"
            )

    def test_embeddings_are_on_cpu(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be on CPU."""
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

    def test_no_extra_keys_produced(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Embedding dict should contain only pairs of layer and module as keys."""
        embeddings = mock_vit_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Only specified single layer in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Only specified single module in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["mha"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha"}

    def test_module_filter_subset(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Only specified modules in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["mha", "rc2"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha", "rc2"}

    def test_module_filter_all_layers(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for the specified module."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["mha"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_mha") for k in embeddings)

    def test_rc1_filter_all_layer(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for rc1 even as a pre-hook module."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, modules=["rc1"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_rc1") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Empty specified layers should returned empty dict."""
        assert mock_vit_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Empty specified modules should returned empty dict."""
        assert mock_vit_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_vit_model: ViT, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers and modules in get_embeddings should be returned."""
        embeddings = mock_vit_model.get_embeddings(pixel_values, layers=[2], modules=["mha", "rc2"])
        assert set(embeddings.keys()) == {"block_2_mha", "block_2_rc2"}
