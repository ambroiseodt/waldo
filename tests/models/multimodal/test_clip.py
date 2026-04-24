r"""
Clip unit tests.

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

from waldo.models import Clip, ClipBlockAdapter, ClipConfig
from waldo.models.transformer_block import TRANSFORMER_MODULES

from .conftest import (
    BATCH_SIZE,
    HIDDEN_SIZE,
    N_LAYERS,
    SEQ_LEN,
    MockClipBlock,
)


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestClipConfig:
    r"""Tests for config class valid entries and default values."""

    @pytest.mark.parametrize(
        "model_size,patch_size",
        [
            ("base", 16),
            ("base", 32),
            ("large", 14),
        ],
    )
    def test_valid_size_combinations_accepted(self, model_size: str, patch_size: int) -> None:
        r"""Valid model size and patch size should instantiate without errors."""
        cfg = ClipConfig(model_size=model_size, patch_size=patch_size)
        assert cfg.model_size == model_size
        assert cfg.patch_size == patch_size

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model size should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            ClipConfig(model_size="huge")

    def test_defaults(self) -> None:
        r"""Default config should have model_size as 'base' and device as 'cpu'."""
        cfg = ClipConfig()
        assert cfg.model_size == "base"
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# BlockAdapter class
# ---------------------------------------------------------------------------
def _make_adapter() -> tuple[ClipBlockAdapter, MockClipBlock]:
    r"""Helper to instantiate ClipBlockAdapter."""
    block = MockClipBlock()
    return ClipBlockAdapter(block), block


class TestClipBlockAdapter:
    r"""Tests for correct instantiation and default values."""

    def test_module_map_has_all_eight_keys(self) -> None:
        r"""Default ClipBlockAdapter has the 8 transformer modules."""
        adapter, _ = _make_adapter()
        assert set(adapter.module_map().keys()) == set(TRANSFORMER_MODULES)

    def test_pre_hooks_match_spec(self) -> None:
        r"""ClipBlockAdapter applies pre-hook only on 'rc1'."""
        adapter, _ = _make_adapter()
        assert adapter.pre_hooks == {"rc1"}

    def test_module_instantiation(self) -> None:
        r"""Default ClipBlockAdapter should correctly map each transformer module."""
        adapter, block = _make_adapter()
        assert adapter.ln1 is block.layer_norm1
        assert adapter.mha is block.self_attn
        assert adapter.ln2 is block.layer_norm2
        assert adapter.rc1 is block.layer_norm2
        assert adapter.fc1 is block.mlp.fc1
        assert adapter.act is block.mlp.activation_fn
        assert adapter.fc2 is block.mlp.fc2
        assert adapter.rc2 is block


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------
class TestClipModel:
    r"""Tests for model class valid entries, correct instantiation, and default values."""

    def test_invalid_model_name_raises(self, mock_clip_model: Clip) -> None:
        r"""Invalid model name should raise a ValueError with an informative message."""
        with pytest.raises(ValueError, match="is not valid"):
            mock_clip_model._load_vision_model("openai/clip-fake", "cpu")
        with pytest.raises(ValueError, match="is not valid"):
            mock_clip_model._load_text_model("openai/clip-fake", "cpu")

    def test_model_in_eval_mode_after_loading(self, mock_clip_model: Clip) -> None:
        r"""Both vision and text models should be in eval mode."""
        cast(Any, mock_clip_model.vision_model.eval).assert_called()
        cast(Any, mock_clip_model.text_model.eval).assert_called()

    def test_blocks_returns_module_list(self, mock_clip_model: Clip) -> None:
        r"""Blocks property should return a module list object."""
        assert isinstance(mock_clip_model.blocks, ModuleList)
        assert len(mock_clip_model.blocks) == N_LAYERS

    def test_blocks_length_equals_n_layers(self, mock_clip_model: Clip) -> None:
        r"""Blocks length should match the number of layers in the model."""
        assert len(mock_clip_model.blocks) == N_LAYERS

    def test_block_adapter_is_clip_adapter(self, mock_clip_model: Clip) -> None:
        r"""Block adapter property should return ClipBlockAdapter."""
        assert mock_clip_model.block_adapter is ClipBlockAdapter

    def test_pooling_returns_n_e_shape(self, mock_clip_model: Clip) -> None:
        r"""Pooling method should return batch size x embedding dimension on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_clip_model.pooling(hidden)
        assert out.shape == (BATCH_SIZE, HIDDEN_SIZE)
        assert out.device.type == "cpu"

    def test_pooling_result_is_on_cpu(self, mock_clip_model: Clip) -> None:
        r"""Pooling method output should be on CPU."""
        hidden = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        out = mock_clip_model.pooling(hidden)
        assert out.device.type == "cpu"

    def test_pooling_excludes_cls_token(self, mock_clip_model: Clip) -> None:
        r"""Clip pooling must exclude the CLS token (index 0)."""
        hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE)
        # Create a detectable difference at index 0
        hidden[:, 0, :] = 100.0
        out_excl = mock_clip_model.pooling(hidden)
        out_incl = hidden.mean(dim=1).cpu()
        assert not torch.allclose(out_excl, out_incl)

    def test_get_class_embeddings_alignment(self, mock_clip_model: Clip) -> None:
        r"""Verify text labels are processed and return pooled embeddings."""
        labels = ["a photo of a cat", "a photo of a dog"]
        out = mock_clip_model.get_class_embeddings(labels)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (len(labels), HIDDEN_SIZE)

    def test_get_embeddings_uses_processor(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""VLM must delegate vision processing to self.processor."""
        _ = mock_clip_model.get_embeddings(pixel_values, layers=[0])
        cast(Any, mock_clip_model.processor).assert_called()

    def test_returns_all_keys(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Embeddings dict should contain all requested pairs of layers and modules."""
        embeddings = mock_clip_model.get_embeddings(pixel_values)
        expected = {f"block_{layer}_{module}" for layer in range(N_LAYERS) for module in TRANSFORMER_MODULES}
        assert set(embeddings.keys()) == expected

    def test_embedding_shapes_are_n_e(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be a tensor of dimension batch size x embedding dimension."""
        embeddings = mock_clip_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.shape == (BATCH_SIZE, HIDDEN_SIZE), (
                f"{key}: expected ({BATCH_SIZE}, {HIDDEN_SIZE}), got {tensor.shape}"
            )

    def test_embeddings_are_on_cpu(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Each embedding should be on CPU."""
        embeddings = mock_clip_model.get_embeddings(pixel_values)
        for key, tensor in embeddings.items():
            assert tensor.device.type == "cpu", f"{key} is on {tensor.device}"

    def test_no_extra_keys_produced(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Embedding dict should contain only pairs of layer and module as keys."""
        embeddings = mock_clip_model.get_embeddings(pixel_values)
        for key in embeddings:
            parts = key.split("_")
            assert parts[0] == "block"
            assert int(parts[1]) < N_LAYERS
            assert "_".join(parts[2:]) in TRANSFORMER_MODULES

    def test_layer_filter_single(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Only specified single layer in get_embeddings should be returned."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, layers=[0])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0}

    def test_layer_filter_subset(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers in get_embeddings should be returned."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, layers=[0, 2])
        layers_found = {int(k.split("_")[1]) for k in embeddings}
        assert layers_found == {0, 2}

    def test_module_filter_single(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Only specified single module in get_embeddings should be returned."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, modules=["mha"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha"}

    def test_module_filter_subset(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Only specified modules in get_embeddings should be returned."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, modules=["mha", "rc2"])
        modules_found = {k.split("_")[2] for k in embeddings}
        assert modules_found == {"mha", "rc2"}

    def test_module_filter_all_layers(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""In default setting, all layers should be return for the specified module."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, modules=["mha"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_mha") for k in embeddings)

    def test_embeddings_rc1_capture(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""The rc1 module (captured via pre-hook) should be present in results."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, modules=["rc1"])
        assert len(embeddings) == N_LAYERS
        assert all(k.endswith("_rc1") for k in embeddings)

    def test_empty_layers_returns_empty(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Empty specified layers should return empty dict."""
        assert mock_clip_model.get_embeddings(pixel_values, layers=[]) == {}

    def test_empty_modules_returns_empty(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Empty specified modules should returned empty dict."""
        assert mock_clip_model.get_embeddings(pixel_values, modules=[]) == {}

    def test_combined_layer_and_module_filter(self, mock_clip_model: Clip, pixel_values: torch.Tensor) -> None:
        r"""Only specified layers and modules in get_embeddings should be returned."""
        embeddings = mock_clip_model.get_embeddings(pixel_values, layers=[1], modules=["mha", "fc2"])
        assert set(embeddings.keys()) == {"block_1_mha", "block_1_fc2"}
