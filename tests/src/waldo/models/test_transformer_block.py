r"""
Transformer block adapter unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch
import torch.nn as nn

from waldo.models import TRANSFORMER_MODULES, TransformerBlockAdapter
from waldo.models.transformer_block import RC2NORM


class Block(nn.Module):
    r"""A minimal pre-norm transformer block with one submodule per standard module."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1, self.attention, self.norm2 = nn.LayerNorm(4), nn.Identity(), nn.LayerNorm(4)
        self.fc1, self.act, self.fc2 = nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)


class BlockAdapter(TransformerBlockAdapter):
    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("norm1")

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("attention")

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("norm2")

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("norm2")

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("fc1")

    def _get_act(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("act")

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule("fc2")

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        return {"rc1"}


# -------------------------------------------------------------------
# Standard modules
# -------------------------------------------------------------------
class TestTransformerModules:
    r"""Tests for the standard module names."""

    def test_modules_follow_the_block(self) -> None:
        r"""The 8 modules should follow the order of the computations of a pre-norm block."""
        assert TRANSFORMER_MODULES == ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]

    def test_rc2norm_is_not_a_block_module(self) -> None:
        r"""rc2norm should only be probed when requested, and its name should not contain underscores."""
        assert RC2NORM not in TRANSFORMER_MODULES
        assert "_" not in RC2NORM


# -------------------------------------------------------------------
# Adapter
# -------------------------------------------------------------------
class TestTransformerBlockAdapter:
    r"""Tests for the mapping between standard names and submodules."""

    def test_module_map(self) -> None:
        r"""module_map should map the standard names, in order, to the modules returned by the subclass."""
        block = Block()
        module_map = BlockAdapter(block).module_map()
        assert list(module_map) == TRANSFORMER_MODULES
        assert module_map["fc1"] is block.fc1
        assert module_map["rc1"] is module_map["ln2"] is block.norm2
        assert module_map["rc2"] is block

    def test_pre_hooks(self) -> None:
        r"""The modules recovered with a pre-hook should be the ones declared by the subclass."""
        assert BlockAdapter(Block()).pre_hooks == {"rc1"}

    def test_process_output_is_identity_by_default(self) -> None:
        r"""By default, the outputs of the hooked modules should be kept unchanged."""
        output = torch.randn(2, 3, 4)
        assert BlockAdapter(Block()).process_output("fc1", output) is output

    def test_abstract_adapter_cannot_be_instantiated(self) -> None:
        r"""An adapter that does not implement the mapping should not be instantiable."""
        with pytest.raises(TypeError):
            TransformerBlockAdapter(Block())
