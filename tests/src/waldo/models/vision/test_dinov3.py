r"""
DINOv3 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import patch

import pytest
import torch.nn as nn
from transformers import DINOv3ViTModel

from waldo.models import Dinov3, Dinov3BlockAdapter, Dinov3Config

from .conftest import VISION_SPECS, MakeVisionModel

MODEL_SIZES = ["vits16", "vitb16", "vitl16", "vit7b16", "vits16plus", "vith16plus"]


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestDinov3Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be DINOv3 ViT-B/16 with content pooling on CPU."""
        config = Dinov3Config()
        assert (config.model_size, config.pooling, config.device) == ("vitb16", "content", "cpu")

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Dinov3Config(model_size="base")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        assert not hasattr(Dinov3Config(unknown_arg=42), "unknown_arg")


# -------------------------------------------------------------------
# Transformer block adapter
# -------------------------------------------------------------------
class TestDinov3BlockAdapter:
    r"""Tests for the mapping of the FFN modules, with an MLP or a gated MLP."""

    def test_mlp_modules(self, make_vision_model: MakeVisionModel) -> None:
        r"""With an MLP, fc1, act and fc2 should be the up projection, the activation and the down projection."""
        model, _ = make_vision_model("dinov3")
        mlp = model.blocks[0].mlp
        adapter = Dinov3BlockAdapter(model.blocks[0])
        assert (adapter.fc1, adapter.act, adapter.fc2) == (mlp.up_proj, mlp.act_fn, mlp.down_proj)
        assert adapter.pre_hooks == {"rc1"}

    def test_gated_mlp_modules(self, make_vision_model: MakeVisionModel) -> None:
        r"""With a gated MLP, fc1 should be the gate projection and act the input of the down projection."""
        model, _ = make_vision_model("dinov3-gated")
        mlp = model.blocks[0].mlp
        adapter = Dinov3BlockAdapter(model.blocks[0])
        assert (adapter.fc1, adapter.act, adapter.fc2) == (mlp.gate_proj, mlp.down_proj, mlp.down_proj)
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_unsupported_ffn_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""FFN types other than an MLP or a gated MLP should raise a ValueError."""
        model, _ = make_vision_model("dinov3")
        block = model.blocks[0]
        block.mlp = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="Unsupported FFN type"):
            Dinov3BlockAdapter(block)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestDinov3:
    r"""Tests for the loading and structure of DINOv3."""

    @pytest.mark.parametrize("model_size", MODEL_SIZES)
    def test_loads_pretrained_model(self, model_size: str) -> None:
        r"""Each size should load the corresponding LVD-1689M checkpoint on the requested device."""
        spec = VISION_SPECS["dinov3"]
        with patch.object(DINOv3ViTModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load:
            model = Dinov3(Dinov3Config(model_size=model_size))
        model_name = f"facebook/dinov3-{model_size}-pretrain-lvd1689m"
        load.assert_called_once_with(model_name, device_map="cpu")
        assert model.model_name == model_name

    def test_invalid_model_name_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""Loading a checkpoint that is not a DINOv3 model should raise a ValueError."""
        model, _ = make_vision_model("dinov3")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="facebook/dinov3-vitg16-pretrain-lvd1689m", device="cpu")

    def test_structure(self, make_vision_model: MakeVisionModel) -> None:
        r"""DINOv3 should use its block adapter, the final norm and exclude the CLS and the 4 register tokens."""
        model, hf_model = make_vision_model("dinov3")
        assert model.block_adapter is Dinov3BlockAdapter
        assert model.final_norm is hf_model.norm
        assert model.n_prefix_tokens == 5
        assert model.blocks is hf_model.model.layer
