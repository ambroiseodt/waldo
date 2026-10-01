r"""
Transformer primitives shared across VFMs and LLMs.

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

# Output of the final norm applied to the residual stream of the last transformer block (rc2 of the last layer).
# It corresponds to the standard output of the model (e.g., last_hidden_state in HuggingFace) and is only
# defined for the last layer. It is not part of TRANSFORMER_MODULES, so it is only probed when explicitly requested.
RC2NORM = "rc2norm"


# -------------------------------------------------------------------
#   Transformer block adapter
# -------------------------------------------------------------------
class TransformerBlockAdapter(ABC):
    r"""
    Adapter mapping the standard module names to the submodules of a transformer block.

    Subclasses implement the 8 methods returning the modules whose output (or input for pre-hooks) gives:
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
    or Llama, i.e., normalization layers are applied before the attention and feedforward blocks (see [1, 2]_). It
    allows us to recover all modules in a single forward pass with register_forward_hook, except for rc1, which is
    the input of ln2 and is recovered with register_forward_pre_hook. Users who would like to use a post-norm model
    should make sure that their implementation matches this design. The modules recovered with a pre-hook are
    returned by _get_pre_hooks_modules. This is notably needed for models that rely on a gated activation such as
    SwiGLU in the FFN (see [3, 4]_ for details), for which we decompose the FFN block as follows:

        fc1  = gate_proj(x)                      -> linear layer which feeds the activation
        act  = sigma(gate_proj(x)) * up_proj(x)  -> non-linearity (which is a function of the input too)
        fc2  = down_proj(act)                    -> linear layer which gives the output of the FFN

    This ensures that fc1 and fc2 remain pure linear operations, and the gating mechanism is absorbed into act. The
    up_proj can be interpreted as part of the activation definition rather than a separate linear layer, making act
    the complete sequence-to-sequence non-linearity between fc1 and fc2. Subclasses must reflect this convention in
    their _get_fc1, _get_act, and _get_fc2 implementations, and declare act as a pre-hook module in
    _get_pre_hooks_modules, since the gated activation output is recovered as the input to down_proj. The
    process_output method allows to post-process a module output, for instance to recover fc1 from the fused gate
    and up projections of DINOv2-g.

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
    ...     def _get_rc1(self, layer):  return layer.layernorm_after  # pre-hook captures rc1
    ...     def _get_ln2(self, layer):  return layer.layernorm_after
    ...     def _get_fc1(self, layer):  return layer.intermediate.dense
    ...     def _get_act(self, layer):  return layer.intermediate.intermediate_act_fn
    ...     def _get_fc2(self, layer):  return layer.output.dense
    ...     def _get_rc2(self, layer):  return layer
    ...     def _get_pre_hooks_modules(self, layer):  return {"rc1"}
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
        Returns the names of the modules for which register_forward_pre_hook should be used
        in place of register_forward_hook to access their inputs instead of their outputs.
        """
        ...

    def module_map(self) -> dict[str, nn.Module]:
        r"""Return the modules of the block indexed by their standard names, in the order of TRANSFORMER_MODULES."""
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

    def process_output(self, module_name: str, output: torch.Tensor) -> torch.Tensor:
        r"""
        Optionally post-process a hooked module output before storing it.
        """
        return output
