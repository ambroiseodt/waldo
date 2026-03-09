r"""
Abstract model for large vision models (LVMs). It allows to study specific layer and modules following [1, 2]_.

Reference
---------
.. [1] A. Odonnat et al. Vision Transformer Finetuning Benefits from Smooth Components. arXiv 2026

.. [2] A. Odonnat et al. Layer by layer, module by module: Choose both for optimal
       OOD probing of ViT. In ICLR 2026 CAO Workshop

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from abc import ABC, abstractmethod

import torch.nn as nn

# -------------------------------------------------------------------
# Block adapter
# -------------------------------------------------------------------


class TransformerBlockAdapter(ABC):
    r"""
    Adapter that maps standard module names to a transformer block's submodules:
        ln1: first layer norm
        attn:  multi-head attention
        ln2: second layer norm
        ffn_fc1: first feedforward linear layer
        ffn_act: feedforward activation function
        ffn_fc2: second feedforward linear layer

    Parameters
    ----------
    layer: nn.Module
        A single transformer block.

    Usage
    -----
    Create a subclass that implement the six abstract methods for a specific architecture:
    >>> class ViTBlockAdapter(TransformerBlockAdapter):
    ...     def _get_ln1(self, layer):     return layer.layernorm_before
    ...     def _get_attn(self, layer):    return layer.attention
    ...     def _get_ln2(self, layer):     return layer.layernorm_after
    ...     def _get_fc1(self, layer): return layer.intermediate
    ...     def _get_act(self, layer): return layer.intermediate.intermediate_act_fn
    ...     def _get_fc2(self, layer): return layer.output
    """

    def __init__(self, layer: nn.Module):
        self.ln1 = self._get_ln1(layer)
        self.attn = self._get_attn(layer)
        self.ln2 = self._get_ln2(layer)
        self.fc1 = self._get_fc1(layer)
        self.act = self._get_act(layer)
        self.fc2 = self._get_fc2(layer)

    @abstractmethod
    def _get_ln1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_attn(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_ln2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_act(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc2(self, layer: nn.Module) -> nn.Module: ...

    def module_map(self) -> dict[str, nn.Module]:
        return {
            "ln1": self.ln1,
            "attn": self.attn,
            "ln2": self.ln2,
            "fc1": self.fc1,
            "act": self.act,
            "fc2": self.fc2,
        }
