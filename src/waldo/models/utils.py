r"""
Utils functions for foundation models.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import torch
import torch.nn as nn


# -------------------------------------------------------------------
#   Pooling
# -------------------------------------------------------------------
def mean_pooling(
    x: torch.Tensor,
    exclude_cls: bool = False,
    attention_mask: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    special_token_ids: list[int | None] | None = None,
) -> torch.Tensor:
    r"""
    Mean pooling over context tokens with flexible exclusion of non-content tokens. For vision models
    where a CLS token is defined, we exclude it from the pooling. For language models, special tokens
    such as eos, bos and padding tokens are also excluded and the same follows for vision-language models.

    Parameters
    ----------
    x: torch.Tensor of shape (N, L, E)
        Sequence of tokens to pool.
    exclude_cls: bool, default=False
        If True, excludes the CLS token (position 0).
    attention_mask: torch.Tensor of shape (N, L) | None, default=None
        Padding mask (1 for real tokens, 0 for padding). If None, no padding tokens are excluded.
    input_ids: torch.Tensor of shape (N, L) | None, default=None
        Integer tensor mapping each token to its vocabulary index, as returned
        by the tokenizer. Used to identify and exclude special tokens by id.
        Required when special_token_ids is set. For instance, the sentence "Hello world"
        writes with a Llama tokenizer as:
            input_ids = tensor([[1, 15043, 3186, 2]])
                                 ↑               ↑
                                BOS             EOS
        The BOS (1) and EOS (2) ids would be passed in special_token_ids to exclude them from pooling.
    special_token_ids: list[int | None] | None, default=None
        Token ids to exclude from pooling, e.g. [bos_token_id, eos_token_id].
        None values are ignored to handle tokenizers that miss some special tokens).
        When defined, input_ids must be provided.

    Returns
    -------
    x_pooled: torch.Tensor of shape (N, E)
        Global sequence representation after pooling.

    Raises
    ------
    ValueError
        If special_token_ids is provided but input_ids is None.
    ValueError
        If all tokens are masked out for at least one sequence in the batch.

    Notes
    -----
    In computer vision, it is a common practice to probe patch tokens that contain valuable information
    about the input images. In [1]_, the authors show that it leads to similar performance than probing the
    CLS token for image classification. In [2, 3]_, probing for semantic segmentation or depth estimation is
    done over the patch tokens. In our setting, it allows us to have a consistent experimental setup for models
    that do not have CLS tokens. Given that we probe across transformer blocks and modules, mean pooling over
    patch tokens ensure a fair comparison. We follow the same logic for vision-language and language models by applying
    mean pooling over context tokens, exclusing separations and padding tokens. Note that for language models, the
    attention_mask, input_ids and special_token_ids should be recovered via the tokenizer.

    References
    ----------
    .. [1] K. He et al. Masked Autoencoders Are Scalable Vision Learners. In CVPR 2022

    .. [2] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. In TMLR 2024

    .. [3] O. Simeoni et al. DINOv3. arXiv 2025

    Usage
    -----
    >>> # ViT: exclude the CLS token, pool over patch tokens
    >>> mean_pooling(x, exclude_cls=True)

    >>> # AIM (autoregressive vision models): there is no CLS token, pool over all tokens
    >>> mean_pooling(x, exclude_cls=False)

    >>> # Llama: exclude BOS, EOS, PAD tokens
    >>> mean_pooling(
    ...     x,
    ...     attention_mask=attention_mask,
    ...     input_ids=input_ids,
    ...     special_token_ids=[tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id],
    ... )
    """
    if special_token_ids is not None and input_ids is None:
        raise ValueError("special_token_ids requires input_ids to be provided.")

    N, L, _ = x.shape
    mask = torch.ones(N, L, dtype=torch.bool, device=x.device)

    # Exclude CLS token (always at position 0)
    if exclude_cls:
        mask[:, 0] = False

    # Exclude padding tokens
    if attention_mask is not None:
        mask = mask & attention_mask.bool()

    # Exclude special tokens by id
    if special_token_ids is not None and input_ids is not None:
        valid_ids = [sid for sid in special_token_ids if sid is not None]
        if valid_ids:
            special_ids_tensor = torch.tensor(valid_ids, device=x.device)
            mask = mask & ~torch.isin(input_ids, special_ids_tensor)

    # Check that no sequence has all tokens masked
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("All tokens are masked out for at least one sequence. Check the masking arguments.")

    # Apply mask and pool
    mask = mask[:, :, None].float()
    x_pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).float()

    return x_pooled


# -------------------------------------------------------------------
#   Gated activation wrapper
# -------------------------------------------------------------------
class SwiGLUGateProjection(nn.Module):
    r"""
    Wraps a linear layer and returns only the first half of its output.
    It can be used to recover the gate projection of a feedforward network using SwiGLU like
    in Dinov2, Dinov3 or Llama model.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        x1, _ = out.chunk(2, dim=-1)
        return x1
