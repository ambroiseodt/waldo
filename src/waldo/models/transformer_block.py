r"""
Transformer primitives shared across VFMs, VLMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

TRANSFORMER_MODULES = ["ln1", "mha", "rc1", "ln2", "fc1", "act", "fc2", "rc2"]


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class TransformerBlockAdapter(ABC):
    r"""
    Adapter that maps standard module names to a transformer block's submodules.

    Subclasses need to implement the 8 methods that outputs the following modules:
        ln1: first layer norm
        mha: multi-head attention
        rc1: first residual connection
        ln2: second layer norm
        fc1: first feedforward linear layer
        act: feedforward activation function
        fc2: second feedforward linear layer
        rc2: second residual connection

    Parameters
    ----------
    layer: nn.Module
        A single transformer block.

    Notes
    -----
    Our implementation assumes that pre-norm is used, as is the case in modern transformer models like ViT, DINO,
    or Llama, i.e., normalization layers are applied before the attention and feedforward blocks (see [1, 2]). It
    allows us to recover all modules using a single forward pass by using register_forward_hook, except for rc1,
    which can be seen as the input to ln2 and is recovered using register_forward_pre_hook.  We advise users that
    would like to use a post-norm standard to make sure their implementation matches our design. For generality, the
    modules that require a pre_hook can be specified in the _pre_hook_modules method. This is notably needed for
    modern models that rely on a gated activation like SwiGLU in the FFN (see [3, 4]_ for details). For consistency,
    we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = swish(gate_proj(x)) * up_proj(x)   -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    This ensures that fc1 and fc2 remain pure linear operations, and the gating mechanism is absorbed into act. The
    up_proj can be interpreted as part of the activation definition rather than a separate linear layer, making act
    the complete sequence-to-sequence non-linearity between fc1 and fc2. Subclasses must reflect this convention in
    their _get_fc1, _get_act, and _get_fc2 implementations, and declare act as a pre-hook module in _pre_hook_modules
    when SwiGLU is used, since the gated activation output is recovered as the input to down_proj.

    References
    ----------
    .. [1] R. Child et al. Generating Long Sequences with Sparse Transformers. arXiv 2019

    .. [2] R. Xiong et al. On Layer Normalization in the Transformer Architecture. In ICML 2020

    .. [3] Y. Dauphin et al. Language Modeling with Gated Convolutional Networks. In ICML 2017

    .. [4] N. Shazeer. GLU Variants Improve Transformer. arXiv 2020

    Usage
    -----
    >>> class ViTBlockAdapter(TransformerBlockAdapter):
    ...     def _get_ln1(self, layer):  return layer.layernorm_before
    ...     def _get_mha(self, layer):  return layer.attention
    ...     def _get_rc1(self, layer):  return layer.layer_norm_after  # pre_hook captures rc1
    ...     def _get_ln2(self, layer):  return layer.layernorm_after
    ...     def _get_fc1(self, layer):  return layer.intermediate.dense
    ...     def _get_act(self, layer):  return layer.intermediate.intermediate_act_fn
    ...     def _get_fc2(self, layer):  return layer.output.dense
    ...     def _get_rc2(self, layer):  return layer.output
    """

    def __init__(self, layer: nn.Module):
        self.ln1 = self._get_ln1(layer)
        self.mha = self._get_mha(layer)
        self.rc1 = self._get_rc1(layer)
        self.ln2 = self._get_ln2(layer)
        self.fc1 = self._get_fc1(layer)
        self.act = self._get_act(layer)
        self.fc2 = self._get_fc2(layer)
        self.rc2 = self._get_rc2(layer)
        self.pre_hooks = self._get_pre_hooks_modules(layer)

    @abstractmethod
    def _get_ln1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_mha(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_rc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_ln2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc1(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_act(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_fc2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_rc2(self, layer: nn.Module) -> nn.Module: ...

    @abstractmethod
    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        r"""
        Returns the names of the module for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of the outputs.

        """
        ...

    def module_map(self) -> dict[str, nn.Module]:
        return {
            "ln1": self.ln1,
            "mha": self.mha,
            "rc1": self.rc1,
            "ln2": self.ln2,
            "fc1": self.fc1,
            "act": self.act,
            "fc2": self.fc2,
            "rc2": self.rc2,
        }


# -------------------------------------------------------------------
#   Gated activation wrapper
# -------------------------------------------------------------------
class SwiGLUGateProjection(nn.Module):
    r"""
    Wraps a linear layer and returns only the first half of its output.
    It can be used to recover the gate projection of the Dinov2 SwiGLU feedforward network.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        x1, _ = out.chunk(2, dim=-1)
        return x1
