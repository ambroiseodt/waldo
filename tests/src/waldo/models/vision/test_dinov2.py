r"""
DINOv2 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2WithRegistersModel

from waldo.models import Dinov2, Dinov2BlockAdapter, Dinov2Config

from .conftest import VISION_SPECS, MakeVisionModel


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestDinov2Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be DINOv2-B without registers with content pooling on CPU."""
        config = Dinov2Config()
        assert (config.model_size, config.pooling, config.device) == ("base", "content", "cpu")

    @pytest.mark.parametrize("model_size", ["huge", "base-register", "registers"])
    def test_invalid_model_size_raises(self, model_size: str) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Dinov2Config(model_size=model_size)

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        assert not hasattr(Dinov2Config(unknown_arg=42), "unknown_arg")


# -------------------------------------------------------------------
# Transformer block adapter
# -------------------------------------------------------------------
class TestDinov2BlockAdapter:
    r"""Tests for the mapping of the FFN modules, with an MLP or a SwiGLU FFN."""

    def test_mlp_modules(self, make_vision_model: MakeVisionModel) -> None:
        r"""With an MLP, fc1, act and fc2 should be the two linear layers and the activation."""
        model, _ = make_vision_model("dinov2")
        mlp = model.blocks[0].mlp
        adapter = Dinov2BlockAdapter(model.blocks[0])
        assert (adapter.fc1, adapter.act, adapter.fc2) == (mlp.fc1, mlp.activation, mlp.fc2)
        assert adapter.pre_hooks == {"rc1"}

    def test_swiglu_modules(self, make_vision_model: MakeVisionModel) -> None:
        r"""With a SwiGLU FFN, fc1 should be recovered from weights_in and act as the input of weights_out."""
        model, _ = make_vision_model("dinov2-swiglu")
        mlp = model.blocks[0].mlp
        adapter = Dinov2BlockAdapter(model.blocks[0])
        assert (adapter.fc1, adapter.act, adapter.fc2) == (mlp.weights_in, mlp.weights_out, mlp.weights_out)
        assert adapter.pre_hooks == {"rc1", "act"}

    def test_swiglu_fc1_is_gate_projection(self, make_vision_model: MakeVisionModel) -> None:
        r"""For SwiGLU, process_output should keep the gate projection, i.e. the first half of weights_in."""
        model, _ = make_vision_model("dinov2-swiglu")
        adapter = Dinov2BlockAdapter(model.blocks[0])
        output = torch.randn(2, 5, 8)
        torch.testing.assert_close(adapter.process_output("fc1", output), output[..., :4])
        torch.testing.assert_close(adapter.process_output("fc2", output), output)

    def test_unsupported_ffn_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""FFN types other than an MLP or a SwiGLU FFN should raise a ValueError."""
        model, _ = make_vision_model("dinov2")
        block = model.blocks[0]
        block.mlp = nn.Linear(2, 2)
        with pytest.raises(ValueError, match="Unsupported FFN type"):
            Dinov2BlockAdapter(block)


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestDinov2:
    r"""Tests for the loading and structure of DINOv2, with and without registers."""

    @pytest.mark.parametrize("model_size", ["small", "base", "large", "giant"])
    def test_loads_pretrained_model(self, model_size: str) -> None:
        r"""Each size should load the corresponding checkpoint on the requested device."""
        spec = VISION_SPECS["dinov2"]
        with patch.object(Dinov2Model, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load:
            model = Dinov2(Dinov2Config(model_size=model_size))
        load.assert_called_once_with(f"facebook/dinov2-{model_size}", device_map="cpu")
        assert model.model_name == f"facebook/dinov2-{model_size}"

    @pytest.mark.parametrize("model_size", ["small", "base", "large", "giant"])
    def test_loads_pretrained_model_with_registers(self, model_size: str) -> None:
        r"""Sizes with registers should load the checkpoint with registers through Dinov2WithRegistersModel."""
        spec = VISION_SPECS["dinov2-registers"]
        hf_model = spec.hf_class(spec.hf_config)
        with patch.object(Dinov2WithRegistersModel, "from_pretrained", return_value=hf_model) as load:
            model = Dinov2(Dinov2Config(model_size=f"{model_size}-registers"))
        load.assert_called_once_with(f"facebook/dinov2-with-registers-{model_size}", device_map="cpu")
        assert model.n_prefix_tokens == 1 + hf_model.config.num_register_tokens

    def test_invalid_model_name_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""Loading a checkpoint that is not a DINOv2 model should raise a ValueError."""
        model, _ = make_vision_model("dinov2")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="facebook/dinov2-huge", device="cpu")

    @pytest.mark.parametrize("name", ["dinov2", "dinov2-registers"])
    def test_structure(self, make_vision_model: MakeVisionModel, name: str) -> None:
        r"""DINOv2 should use its block adapter, the final layernorm and exclude the CLS and register tokens."""
        model, hf_model = make_vision_model(name)
        assert model.block_adapter is Dinov2BlockAdapter
        assert model.final_norm is hf_model.layernorm
        assert model.n_prefix_tokens == VISION_SPECS[name].n_prefix_tokens
        assert model.blocks is hf_model.get_submodule("encoder.layer")
