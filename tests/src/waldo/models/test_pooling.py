r"""
Pooling unit tests.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import pytest
import torch

from waldo.models import mean_pooling


@pytest.fixture
def hidden_states() -> torch.Tensor:
    r"""A batch of 2 sequences of 5 tokens with embedding dimension 3."""
    return torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)


# -------------------------------------------------------------------
# Mean pooling
# -------------------------------------------------------------------
class TestMeanPooling:
    r"""Tests for mean pooling over content tokens."""

    def test_mean_over_all_tokens(self, hidden_states: torch.Tensor) -> None:
        r"""Without exclusion, the pooling should be the mean over all tokens."""
        torch.testing.assert_close(mean_pooling(hidden_states), hidden_states.mean(dim=1))

    @pytest.mark.parametrize("n_prefix_tokens", [1, 3])
    def test_prefix_tokens_are_excluded(self, hidden_states: torch.Tensor, n_prefix_tokens: int) -> None:
        r"""The first n_prefix_tokens tokens (CLS and registers) should be excluded."""
        pooled = mean_pooling(hidden_states, n_prefix_tokens=n_prefix_tokens)
        torch.testing.assert_close(pooled, hidden_states[:, n_prefix_tokens:].mean(dim=1))

    def test_padding_tokens_are_excluded(self, hidden_states: torch.Tensor) -> None:
        r"""Tokens with a zero attention mask should be excluded, per sequence."""
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]])
        pooled = mean_pooling(hidden_states, attention_mask=attention_mask)
        torch.testing.assert_close(pooled[1], hidden_states[1, :2].mean(dim=0))

    def test_special_tokens_are_excluded(self, hidden_states: torch.Tensor) -> None:
        r"""Special tokens should be excluded by id, and None ids ignored."""
        input_ids = torch.tensor([[1, 5, 6, 7, 2], [1, 8, 9, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        pooled = mean_pooling(
            hidden_states, attention_mask=attention_mask, input_ids=input_ids, special_token_ids=[1, 2, None]
        )
        torch.testing.assert_close(pooled[0], hidden_states[0, 1:4].mean(dim=0))
        torch.testing.assert_close(pooled[1], hidden_states[1, 1:3].mean(dim=0))

    def test_special_tokens_require_input_ids(self, hidden_states: torch.Tensor) -> None:
        r"""Special token ids without input ids should raise a ValueError."""
        with pytest.raises(ValueError, match="requires input_ids"):
            mean_pooling(hidden_states, special_token_ids=[1])

    def test_fully_masked_sequence_raises(self, hidden_states: torch.Tensor) -> None:
        r"""A sequence whose tokens are all excluded should raise a ValueError instead of returning NaN."""
        attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
        with pytest.raises(ValueError, match="All tokens are masked out"):
            mean_pooling(hidden_states, n_prefix_tokens=1, attention_mask=attention_mask)
