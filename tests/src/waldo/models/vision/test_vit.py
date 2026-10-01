r"""
ViT unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import patch

import pytest
from transformers import ViTModel

from waldo.models import ViT, ViTBlockAdapter, ViTConfig

from .conftest import VISION_SPECS, MakeVisionModel


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestViTConfig:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be ViT-B/16 with content pooling on CPU."""
        config = ViTConfig()
        assert (config.model_size, config.patch_size, config.pooling, config.device) == ("base", 16, "content", "cpu")

    @pytest.mark.parametrize("model_size", ["unexisting", "small"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            ViTConfig(model_size=model_size)

    def test_invalid_patch_size_raises(self) -> None:
        r"""Invalid patch sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid patch size"):
            ViTConfig(patch_size=8)

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        assert not hasattr(ViTConfig(unknown_arg=42), "unknown_arg")


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestViT:
    r"""Tests for the loading and structure of ViT."""

    @pytest.mark.parametrize(
        "model_size, patch_size, model_name",
        [
            ("base", 16, "google/vit-base-patch16-224-in21k"),
            ("base", 32, "google/vit-base-patch32-224-in21k"),
            ("large", 16, "google/vit-large-patch16-224-in21k"),
            ("large", 32, "google/vit-large-patch32-224-in21k"),
            ("huge", 14, "google/vit-huge-patch14-224-in21k"),
        ],
    )
    def test_loads_pretrained_model(self, model_size: str, patch_size: int, model_name: str) -> None:
        r"""Each size should load the corresponding ImageNet-21k checkpoint on the requested device."""
        spec = VISION_SPECS["vit"]
        with patch.object(ViTModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load:
            model = ViT(ViTConfig(model_size=model_size, patch_size=patch_size))
        load.assert_called_once_with(model_name, device_map="cpu")
        assert model.model_name == model_name

    def test_unavailable_checkpoint_raises(self) -> None:
        r"""Size and patch size combinations without checkpoint should raise a ValueError."""
        with pytest.raises(ValueError, match="is not valid"):
            ViT(ViTConfig(model_size="huge", patch_size=16))

    def test_structure(self, make_vision_model: MakeVisionModel) -> None:
        r"""ViT should use its block adapter, the final layernorm and exclude the CLS token."""
        model, hf_model = make_vision_model("vit")
        assert model.block_adapter is ViTBlockAdapter
        assert model.final_norm is hf_model.layernorm
        assert model.n_prefix_tokens == 1
        assert model.blocks is hf_model.get_submodule("encoder.layer")
