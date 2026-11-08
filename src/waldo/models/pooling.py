r"""
Pooling functions for VFMs and LLMs.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import torch


# -------------------------------------------------------------------
#   Mean pooling
# -------------------------------------------------------------------
def mean_pooling(
    hidden_states: torch.Tensor,
    n_prefix_tokens: int = 0,
    attention_mask: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    special_token_ids: list[int | None] | None = None,
) -> torch.Tensor:
    r"""
    Mean pooling over content tokens with flexible exclusion of non-content tokens. For vision models, the
    auxiliary tokens prepended to the patch tokens (CLS and register tokens) are excluded from the pooling.
    For language models, special tokens such as BOS, EOS and padding tokens are excluded.

    Parameters
    ----------
    hidden_states: torch.Tensor of shape (N, L, E)
        Sequence of tokens to pool.
    n_prefix_tokens: int, default=0
        Number of auxiliary tokens at the start of the sequence to exclude, e.g., 1 for the CLS token
        or 1 + n_registers for models with a CLS token followed by register tokens [4]_.
    attention_mask: torch.Tensor of shape (N, L) | None, default=None
        Padding mask (1 for real tokens, 0 for padding). If None, no padding tokens are excluded.
    input_ids: torch.Tensor of shape (N, L) | None, default=None
        Integer tensor mapping each token to its vocabulary index, as returned by the tokenizer.
        Used to identify and exclude special tokens by id. Required when special_token_ids is set.
    special_token_ids: list[int | None] | None, default=None
        Token ids to exclude from pooling, e.g. [bos_token_id, eos_token_id]. None values are ignored
        to handle tokenizers that miss some special tokens. When defined, input_ids must be provided.

    Returns
    -------
    emb: torch.Tensor of shape (N, E)
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
    patch tokens ensure a fair comparison. Register tokens [4]_ are excluded for the same reason as the CLS token:
    they are learned auxiliary tokens that do not correspond to any image patch and are discarded at the output
    of the model by design. We follow the same logic for language models by applying mean pooling over content
    tokens, notably excluding separation and padding tokens. Note that for language models, the attention_mask,
    input_ids and special_token_ids should be recovered via the tokenizer.

    References
    ----------
    .. [1] K. He et al. Masked Autoencoders Are Scalable Vision Learners. In CVPR 2022

    .. [2] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. In TMLR 2024

    .. [3] O. Simeoni et al. DINOv3. arXiv 2025

    .. [4] T. Darcet et al. Vision Transformers Need Registers. In ICLR 2024

    Usage
    -----
    >>> # ViT: exclude the CLS token, pool over patch tokens
    >>> mean_pooling(x, n_prefix_tokens=1)

    >>> # DINOv3: exclude the CLS token and the 4 register tokens, pool over patch tokens
    >>> mean_pooling(x, n_prefix_tokens=5)

    >>> # AIMv2: there is no CLS token, pool over all tokens
    >>> mean_pooling(x, n_prefix_tokens=0)

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

    N, L, _ = hidden_states.shape
    mask = torch.ones(N, L, dtype=torch.bool, device=hidden_states.device)

    # Exclude auxiliary tokens (CLS and registers) at the start of the sequence
    mask[:, :n_prefix_tokens] = False

    # Exclude padding tokens
    if attention_mask is not None:
        mask = mask & attention_mask.bool()

    # Exclude special tokens by id
    if special_token_ids is not None and input_ids is not None:
        valid_ids = [sid for sid in special_token_ids if sid is not None]
        if valid_ids:
            special_ids_tensor = torch.tensor(valid_ids, device=hidden_states.device)
            mask = mask & ~torch.isin(input_ids, special_ids_tensor)

    # Check that no sequence has all tokens masked
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("All tokens are masked out for at least one sequence. Check the masking arguments.")

    # Apply mask and pool
    mask = mask[:, :, None].float()
    emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).float()

    return emb
