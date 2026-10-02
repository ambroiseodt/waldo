r"""
Shared helpers for models unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from waldo.models.transformer_block import TransformerBlockAdapter


# -------------------------------------------------------------------
# Block inputs
# -------------------------------------------------------------------
@contextmanager
def capture_block_inputs(blocks: nn.ModuleList) -> Iterator[dict[int, torch.Tensor]]:
    r"""Record the input hidden states of each transformer block during the forward passes of the context."""
    inputs: dict[int, torch.Tensor] = {}
    handles = []
    for idx, block in enumerate(blocks):

        def hook(module: nn.Module, args: tuple, kwargs: dict, idx: int = idx) -> None:
            inputs[idx] = args[0] if args else kwargs["hidden_states"]

        handles.append(block.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield inputs
    finally:
        for handle in handles:
            handle.remove()


# -------------------------------------------------------------------
# Block identities
# -------------------------------------------------------------------
def _ffn_parts(block: nn.Module) -> tuple[Callable | None, Callable | None, Callable, Callable]:
    r"""
    Return the activation and up projection of gated FFNs (None otherwise), and the operations applied to the
    outputs of the attention and FFN blocks before the residual connections (LayerScale or post-norms).
    """
    identity = nn.Identity()
    post_attn = getattr(block, "layer_scale1", None) or identity
    post_ffn = getattr(block, "layer_scale2", None) or getattr(block, "post_feedforward_layernorm", None) or identity
    if hasattr(block, "pre_feedforward_layernorm"):
        post_attn = block.get_submodule("post_attention_layernorm")
    mlp: Any = getattr(block, "mlp", None) or getattr(block, "ffn", None)
    if hasattr(mlp, "weights_in"):
        return F.silu, lambda x: mlp.weights_in(x).chunk(2, dim=-1)[1], post_attn, post_ffn
    if hasattr(mlp, "gate_proj"):
        return mlp.act_fn, mlp.up_proj, post_attn, post_ffn
    return None, None, post_attn, post_ffn


def assert_block_identities(
    block: nn.Module,
    adapter: TransformerBlockAdapter,
    block_input: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> None:
    r"""
    Check that the 8 recovered representations of a transformer block are consistent with each other:
    ln1 = norm1(input), rc1 = input + mha, ln2 = norm2(rc1), fc1 = W1(ln2), act = act(fc1) (times up(ln2) for gated
    FFNs), fc2 = W2(act) and rc2 = rc1 + fc2, up to LayerScale and post-norms applied before residual connections.
    """
    act_fn, up, post_attn, post_ffn = _ffn_parts(block)
    if act_fn is None or up is None:
        expected_act = adapter.act(outputs["fc1"])
    else:
        expected_act = act_fn(outputs["fc1"]) * up(outputs["ln2"])
    expected = {
        "ln1": adapter.ln1(block_input),
        "rc1": block_input + post_attn(outputs["mha"]),
        "ln2": adapter.ln2(outputs["rc1"]),
        "fc1": adapter.process_output("fc1", adapter.fc1(outputs["ln2"])),
        "act": expected_act,
        "fc2": adapter.fc2(outputs["act"]),
        "rc2": outputs["rc1"] + post_ffn(outputs["fc2"]),
    }
    for name, value in expected.items():
        torch.testing.assert_close(outputs[name], value, msg=f"Inconsistent {name}.")
