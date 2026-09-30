r"""
AIMv2 unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import itertools
from unittest.mock import patch

import pytest
from transformers import Aimv2VisionModel

from waldo.models import Aimv2, Aimv2BlockAdapter, Aimv2Config

from .conftest import VISION_SPECS, MakeVisionModel


# -------------------------------------------------------------------
# Config class
# -------------------------------------------------------------------
class TestAimv2Config:
    r"""Tests for config class valid entries and default values."""

    def test_defaults(self) -> None:
        r"""Default config should be AIMv2-L at 224 px with content pooling on CPU."""
        config = Aimv2Config()
        assert (config.model_size, config.resolution, config.pooling, config.device) == (
            "large",
            224,
            "content",
            "cpu",
        )

    def test_invalid_model_size_raises(self) -> None:
        r"""Invalid model sizes should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid model size"):
            Aimv2Config(model_size="base")

    def test_invalid_resolution_raises(self) -> None:
        r"""Invalid resolutions should raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid resolution"):
            Aimv2Config(resolution=512)

    def test_all_pooling_raises(self) -> None:
        r"""All pooling needs a CLS token, which AIMv2 does not have."""
        with pytest.raises(ValueError, match="no CLS token"):
            Aimv2Config(pooling="all")

    def test_unknown_kwargs_are_ignored(self) -> None:
        r"""Unknown kwargs should be silently ignored and not set as attributes."""
        assert not hasattr(Aimv2Config(unknown_arg=42), "unknown_arg")


# -------------------------------------------------------------------
# Model class
# -------------------------------------------------------------------
class TestAimv2:
    r"""Tests for the loading and structure of AIMv2."""

    @pytest.mark.parametrize(
        "model_size, resolution", itertools.product(["large", "huge", "1B", "3B"], [224, 336, 448])
    )
    def test_loads_pretrained_model(self, model_size: str, resolution: int) -> None:
        r"""Each size and resolution should load the corresponding checkpoint on the requested device."""
        spec = VISION_SPECS["aimv2"]
        with patch.object(Aimv2VisionModel, "from_pretrained", return_value=spec.hf_class(spec.hf_config)) as load:
            model = Aimv2(Aimv2Config(model_size=model_size, resolution=resolution))
        model_name = f"apple/aimv2-{model_size}-patch14-{resolution}"
        load.assert_called_once_with(model_name, device_map="cpu")
        assert model.model_name == model_name

    def test_invalid_model_name_raises(self, make_vision_model: MakeVisionModel) -> None:
        r"""Loading a checkpoint that is not an AIMv2 model should raise a ValueError."""
        model, _ = make_vision_model("aimv2")
        with pytest.raises(ValueError, match="is not valid"):
            model._load_model(model_name="apple/aimv2-large-patch14-native", device="cpu")

    def test_structure(self, make_vision_model: MakeVisionModel) -> None:
        r"""AIMv2 should use its block adapter and final RMSNorm, and pool over all tokens."""
        model, hf_model = make_vision_model("aimv2")
        adapter = Aimv2BlockAdapter(model.blocks[0])
        ffn = model.blocks[0].ffn
        assert model.block_adapter is Aimv2BlockAdapter
        assert model.final_norm is hf_model.rms_norm
        assert model.n_prefix_tokens == 0
        assert model.blocks is hf_model.encoder.layers
        assert (adapter.fc1, adapter.act, adapter.fc2) == (ffn.gate_proj, ffn.down_proj, ffn.down_proj)
