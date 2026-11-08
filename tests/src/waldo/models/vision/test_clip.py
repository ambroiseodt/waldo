r"""
CLIP unit tests.

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
from transformers import CLIPConfig, CLIPModel, CLIPVisionModel

from waldo.models import Clip, ClipBlockAdapter, ClipConfig
from waldo.models.transformer_block import RC2NORM

from .conftest import N_LAYERS, TINY, TINY_IMAGE, VISION_SPECS, MakeVisionModel


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestClipConfig:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be CLIP ViT-B/16 at 224 px with content pooling on CPU."""
        config = ClipConfig()
        assert (config.model_size, config.patch_size, config.resolution, config.pooling, config.device) == (
            "base",
            16,
            224,
            "content",
            "cpu",
        )

    @pytest.mark.parametrize("model_size", ["huge", "small"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            ClipConfig(model_size=model_size)

    @pytest.mark.parametrize("model_size, patch_size", [("base", 14), ("large", 16), ("large", 32)])
    def test_invalid_patch_size_raises(self, model_size: str, patch_size: int) -> None:
        r"""Patch sizes without checkpoint for the model size should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid patch size"):
            ClipConfig(model_size=model_size, patch_size=patch_size)

    @pytest.mark.parametrize("model_size, patch_size, resolution", [("base", 16, 336), ("large", 14, 448)])
    def test_invalid_resolution_raises(self, model_size: str, patch_size: int, resolution: int) -> None:
        r"""Resolutions without checkpoint for the model size should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            ClipConfig(model_size=model_size, patch_size=patch_size, resolution=resolution)

    def test_invalid_pooling_raises(self) -> None:
        r"""Invalid poolings should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid pooling"):
            ClipConfig(pooling="max")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        assert not hasattr(ClipConfig(unknown_arg=42), "unknown_arg")


# -------------------------------------------------------------------
# Transformer block adapter
# -------------------------------------------------------------------
class TestClipBlockAdapter:
    r"""Tests for the mapping of the modules of a CLIP block."""

    def test_modules(self, make_vision_model: MakeVisionModel) -> None:
        r"""The modules should be the layernorms, the attention and the MLP of the block, with rc1 as a pre-hook."""
        model, _ = make_vision_model("clip")
        block = model.blocks[0]
        adapter = ClipBlockAdapter(block)
        assert (adapter.ln1, adapter.mha, adapter.ln2) == (block.layer_norm1, block.self_attn, block.layer_norm2)
        assert (adapter.fc1, adapter.act, adapter.fc2) == (block.mlp.fc1, block.mlp.activation_fn, block.mlp.fc2)
        assert adapter.pre_hooks == {"rc1"}


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestClip:
    r"""Tests for the loading and structure of CLIP."""

    @pytest.mark.parametrize(
        "model_size, patch_size, resolution, model_name",
        [
            ("base", 16, 224, "openai/clip-vit-base-patch16"),
            ("base", 32, 224, "openai/clip-vit-base-patch32"),
            ("large", 14, 224, "openai/clip-vit-large-patch14"),
            ("large", 14, 336, "openai/clip-vit-large-patch14-336"),
        ],
    )
    def test_loads_pretrained_model(self, model_size: str, patch_size: int, resolution: int, model_name: str) -> None:
        r"""Each size should load the vision tower of the corresponding checkpoint on the requested device."""
        spec = VISION_SPECS["clip"]
        with patch.object(CLIPVisionModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load:
            model = Clip(ClipConfig(model_size=model_size, patch_size=patch_size, resolution=resolution))
        load.assert_called_once_with(model_name, device_map="cpu")
        assert model.model_name == model_name

    def test_vision_tower_weights_are_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        r"""Loading a full CLIP checkpoint should load every weight of the vision tower and ignore the text tower."""
        # Allow from_pretrained for a tiny checkpoint saved locally (HF_HUB_OFFLINE stays set)
        monkeypatch.undo()
        torch.manual_seed(0)
        text = dict(TINY, vocab_size=100, max_position_embeddings=16)
        config: dict[str, Any] = dict(text_config=text, vision_config=dict(TINY, **TINY_IMAGE))
        clip = CLIPModel(CLIPConfig(**config))
        clip.save_pretrained(tmp_path)
        result = CLIPVisionModel.from_pretrained(tmp_path, output_loading_info=True)
        assert isinstance(result, tuple)
        vision, info = result
        assert not info["missing_keys"]
        assert all(not key.startswith("vision_model") for key in info["unexpected_keys"])
        for name, parameter in vision.vision_model.state_dict().items():
            torch.testing.assert_close(parameter, clip.vision_model.state_dict()[name])

    def test_invalid_model_name_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""Loading a checkpoint that is not a CLIP model should raise a ValueError."""
        model, _ = make_vision_model("clip")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="openai/clip-vit-huge-patch14", device="cpu")

    def test_structure(self, make_vision_model: MakeVisionModel) -> None:
        r"""CLIP should use its block adapter, the post layernorm as final norm and exclude the CLS token."""
        model, hf_model = make_vision_model("clip")
        assert model.block_adapter is ClipBlockAdapter
        assert model.final_norm is hf_model.get_submodule("vision_model.post_layernorm")
        assert model.n_prefix_tokens == 1
        assert model.blocks is hf_model.get_submodule("vision_model.encoder.layers")

    @pytest.mark.parametrize("pooling", ["content", "all", "cls"])
    def test_rc2norm_is_pooler_output(
        self, make_vision_model: MakeVisionModel, images: torch.Tensor, pooling: str
    ) -> None:
        r"""Whatever the pooling, rc2norm should be the pooler output, i.e. the CLS token after the final norm."""
        model, hf_model = make_vision_model("clip", pooling=pooling)
        embeddings = model.get_embeddings(images, layers=[0, -1], modules=["rc2", RC2NORM])
        assert set(embeddings) == {"block_0_rc2", f"block_{N_LAYERS - 1}_rc2", f"block_{N_LAYERS - 1}_{RC2NORM}"}
        with torch.inference_mode():
            expected = hf_model(pixel_values=images).pooler_output
        torch.testing.assert_close(embeddings[f"block_{N_LAYERS - 1}_{RC2NORM}"], expected)

    def test_rc2norm_without_last_layer_raises(self, make_vision_model: MakeVisionModel, images: torch.Tensor) -> None:
        r"""rc2norm should require the last layer, and leave no hook on the model when raising."""
        model, _ = make_vision_model("clip")
        with pytest.raises(ValueError, match="only defined for the last layer"):
            model.get_embeddings(images, layers=[0], modules=[RC2NORM])
        for module in model.model.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks
