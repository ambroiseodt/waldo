r"""
Gemma3 language model.

Comments abbreviations
----------------------
    N: batch size
    L: sequence length
    E: embedding dimension

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import ModuleList
from transformers import AutoTokenizer, Gemma3ForCausalLM, PreTrainedModel, PreTrainedTokenizerBase

from ..base import TRANSFORMER_MODULES, TransformerBlockAdapter
from ..base_language import BaseLanguageModel


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
@dataclass
class Gemma3Config:
    r"""
    Gemma3 configuration file.

    Parameters
    ----------
    model_size: str, default="1b"
        Size of the model. Options are "1b", "4b", "12b", "27b".
    layer: int, default=-1
        Layer index from which to extract embeddings. Defaults to the last layer.
    module: str, default="rc2"
        Module name from TRANSFORMER_MODULES to extract embeddings from.
    device: str, default="cpu"
        Device to load the model on.
    """

    model_size: str = "1b"
    layer: int = -1
    module: str = "rc2"
    device: str = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__annotations__)
        self.__post_init__()

    def __post_init__(self):
        if self.model_size not in ["1b", "4b", "12b", "27b"]:
            raise ValueError(f"Invalid model size '{self.model_size}'. Options are '1b', '4b', '12b', '27b'.")
        if self.module not in TRANSFORMER_MODULES:
            raise ValueError(f"Invalid module '{self.module}'. Options are {TRANSFORMER_MODULES}.")


# -------------------------------------------------------------------
# Block adapter
# -------------------------------------------------------------------
class Gemma3BlockAdapter(TransformerBlockAdapter):
    r"""
    Adapter mapping standard module names to Gemma3DecoderLayer submodules.

    Gemma3 uses SwiGLU in the FFN, so act is declared as a pre-hook module
    following the convention in TransformerBlockAdapter.

        ln1  -> input_layernorm
        mha  -> self_attn
        rc1  -> pre_feedforward_layernorm  (pre-hook: captures residual after attention)
        ln2  -> pre_feedforward_layernorm
        fc1  -> mlp.gate_proj
        act  -> mlp.down_proj              (pre-hook: captures gated activation output)
        fc2  -> mlp.down_proj
        rc2  -> post_feedforward_layernorm (pre-hook: captures residual after FFN)
    """

    def _get_ln1(self, layer: nn.Module) -> nn.Module:
        return layer.input_layernorm

    def _get_mha(self, layer: nn.Module) -> nn.Module:
        return layer.self_attn

    def _get_rc1(self, layer: nn.Module) -> nn.Module:
        return layer.pre_feedforward_layernorm

    def _get_ln2(self, layer: nn.Module) -> nn.Module:
        return layer.pre_feedforward_layernorm

    def _get_fc1(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.gate_proj

    def _get_act(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.down_proj

    def _get_fc2(self, layer: nn.Module) -> nn.Module:
        return layer.mlp.down_proj

    def _get_rc2(self, layer: nn.Module) -> nn.Module:
        return layer.post_feedforward_layernorm

    def _get_pre_hooks_modules(self, layer: nn.Module) -> set[str]:
        return {"rc1", "act", "rc2"}


# -------------------------------------------------------------------
# Gemma3
# -------------------------------------------------------------------
class Gemma3(BaseLanguageModel):
    r"""
    Gemma3 language model wrapped for MTEB evaluation.

    Parameters
    ----------
    config: Gemma3Config
        Configuration object.

    References
    ----------
    .. [1] Gemma Team. Gemma 3 Technical Report. 2025
    """

    def __init__(self, config: Gemma3Config):
        super().__init__()
        self.device = config.device
        self.layer = config.layer
        self.module = config.module
        self.model_name = f"google/gemma-3-{config.model_size}-pt"
        self.model = self._load_from_huggingface(
            model_name=self.model_name,
            device=self.device,
        )
        self.tokenizer = self._load_tokenizer(model_name=self.model_name)

    def _load_from_huggingface(self, model_name: str, device: str) -> PreTrainedModel:
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        return model

    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(model_name)

    @property
    def block_adapter(self) -> type[TransformerBlockAdapter]:
        return Gemma3BlockAdapter

    @property
    def blocks(self) -> ModuleList:
        return self.model.model.layers
