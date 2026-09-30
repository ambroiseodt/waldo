r"""
Base language model unit tests, run on every architecture.

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2026, Ambroise Odonnat
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from waldo.models import TRANSFORMER_MODULES, mean_pooling
from waldo.models.transformer_block import RC2NORM

from ..conftest import assert_block_identities, capture_block_inputs
from .conftest import HIDDEN_SIZE, N_LAYERS, SENTENCES, MakeLanguageModel, WhitespaceTokenizer


def _encode_inputs(sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return the input ids and attention mask of the whitespace tokenizer."""
    encoding = WhitespaceTokenizer()(sentences)
    return encoding.input_ids, encoding.attention_mask


def _encode(model: object, sentences: list[str] | DataLoader) -> torch.Tensor:
    r"""Encode sentences with the MTEB interface, whose task arguments are not used by the models."""
    return model.encode(sentences, task_metadata=None, hf_split="test", hf_subset="default")


# -------------------------------------------------------------------
# Hidden states
# -------------------------------------------------------------------
class TestHiddenStates:
    r"""Tests for the representations recovered at a given layer and module."""

    @pytest.mark.parametrize("layer", [0, N_LAYERS - 1])
    def test_block_identities(self, make_language_model: MakeLanguageModel, language_spec: str, layer: int) -> None:
        r"""The 8 recovered modules of a block should be consistent with the block computations."""
        model, _ = make_language_model(language_spec)
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        outputs = {}
        with torch.inference_mode(), capture_block_inputs(model.blocks) as inputs:
            for module in TRANSFORMER_MODULES:
                model.set_probing(layer=layer, module=module)
                outputs[module] = model._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
            block = model.blocks[layer]
            assert_block_identities(block, model.block_adapter(block), inputs[layer], outputs)

    def test_rc2norm_is_standard_output(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""rc2norm should be the output of the final norm, i.e. the last_hidden_state of HuggingFace."""
        model, hf_model = make_language_model(language_spec)
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        model.set_probing(layer=-1, module=RC2NORM)
        hidden_states = model._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        with torch.inference_mode():
            expected = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        torch.testing.assert_close(hidden_states, expected)

    def test_last_rc2_is_last_block_output(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""rc2 of the last layer should be the output of the last block, before the final norm."""
        model, hf_model = make_language_model(language_spec)
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        hidden_states = model._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        with torch.inference_mode():
            expected = hf_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            torch.testing.assert_close(model.final_norm(hidden_states), expected)

    def test_hooks_are_removed(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""No hook should remain on the model after the forward pass."""
        model, _ = make_language_model(language_spec)
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        for layer, module in [(0, "rc1"), (-1, "fc2"), (-1, RC2NORM)]:
            model.set_probing(layer=layer, module=module)
            model._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        for module in model.model.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks


# -------------------------------------------------------------------
# Probing location
# -------------------------------------------------------------------
class TestSetProbing:
    r"""Tests for the selection of the layer and module to probe."""

    def test_default_is_last_residual_stream(self, make_language_model: MakeLanguageModel) -> None:
        r"""By default, the output of the last transformer block should be probed."""
        model, _ = make_language_model("qwen3")
        assert (model.layer, model.module) == (-1, "rc2")

    def test_none_keeps_current_value(self, make_language_model: MakeLanguageModel) -> None:
        r"""Passing None should keep the current layer or module."""
        model, _ = make_language_model("qwen3")
        model.set_probing(layer=1, module="ln2")
        model.set_probing(layer=None, module="act")
        assert (model.layer, model.module) == (1, "act")
        model.set_probing(layer=0, module=None)
        assert (model.layer, model.module) == (0, "act")

    @pytest.mark.parametrize("layer", [N_LAYERS, -N_LAYERS - 1])
    def test_out_of_range_layer_raises(self, make_language_model: MakeLanguageModel, layer: int) -> None:
        r"""Out of range layer indices should raise a ValueError."""
        model, _ = make_language_model("qwen3")
        with pytest.raises(ValueError, match="out of range"):
            model.set_probing(layer=layer, module=None)

    def test_invalid_module_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Unknown module names should raise a ValueError."""
        model, _ = make_language_model("qwen3")
        with pytest.raises(ValueError, match="Invalid module"):
            model.set_probing(layer=None, module="unexisting")

    @pytest.mark.parametrize("layer", [-1, N_LAYERS - 1])
    def test_rc2norm_at_last_layer(self, make_language_model: MakeLanguageModel, layer: int) -> None:
        r"""rc2norm should be accepted for the last layer, with a negative or nonnegative index."""
        model, _ = make_language_model("qwen3")
        model.set_probing(layer=layer, module=RC2NORM)
        assert model.module == RC2NORM

    def test_rc2norm_at_other_layer_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""rc2norm should be rejected for any other layer."""
        model, _ = make_language_model("qwen3")
        with pytest.raises(ValueError, match="only defined for the last layer"):
            model.set_probing(layer=0, module=RC2NORM)


# -------------------------------------------------------------------
# Encoding
# -------------------------------------------------------------------
class TestEncode:
    r"""Tests for the sentence embeddings used by MTEB."""

    def test_embeddings_are_mean_pooled_content_tokens(
        self, make_language_model: MakeLanguageModel, language_spec: str
    ) -> None:
        r"""Sentence embeddings should be the mean of the hidden states over non-special tokens."""
        model, _ = make_language_model(language_spec)
        model.set_probing(layer=1, module="ln2")
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        hidden_states = model._get_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        content = (attention_mask.bool() & (input_ids > WhitespaceTokenizer.eos_token_id)).float()[..., None]
        expected = (hidden_states * content).sum(dim=1) / content.sum(dim=1)
        torch.testing.assert_close(_encode(model, SENTENCES), expected)

    def test_embeddings_do_not_depend_on_padding(
        self, make_language_model: MakeLanguageModel, language_spec: str
    ) -> None:
        r"""The embedding of a sentence should not change when it is padded in a batch of longer sentences."""
        model, _ = make_language_model(language_spec)
        alone = _encode(model, SENTENCES[1:2])
        batched = _encode(model, SENTENCES)
        torch.testing.assert_close(batched[1:2], alone)

    def test_dataloader_and_list_inputs_match(self, make_language_model: MakeLanguageModel) -> None:
        r"""MTEB DataLoader inputs should give the same embeddings as a list of sentences."""
        model, _ = make_language_model("qwen3")
        loader = DataLoader([{"text": s} for s in SENTENCES], batch_size=2)
        embeddings = _encode(model, loader)
        assert embeddings.shape == (len(SENTENCES), HIDDEN_SIZE)
        torch.testing.assert_close(embeddings, _encode(model, SENTENCES))

    def test_unsupported_input_raises(self, make_language_model: MakeLanguageModel) -> None:
        r"""Inputs that are neither a DataLoader nor a list should raise a ValueError."""
        model, _ = make_language_model("qwen3")
        with pytest.raises(ValueError, match="Unsupported input type"):
            _encode(model, "a sentence")

    def test_pooling_excludes_special_tokens(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""The pooling of each model should exclude its BOS, EOS and padding tokens."""
        model, _ = make_language_model(language_spec)
        input_ids, attention_mask = _encode_inputs(SENTENCES)
        hidden_states = torch.randn(*input_ids.shape, HIDDEN_SIZE)
        expected = mean_pooling(
            hidden_states, attention_mask=attention_mask, input_ids=input_ids, special_token_ids=[1]
        )
        pooled = model.pooling(hidden_states, attention_mask, input_ids, model.special_token_ids)
        torch.testing.assert_close(pooled, expected)


# -------------------------------------------------------------------
# Similarity
# -------------------------------------------------------------------
class TestSimilarity:
    r"""Tests for the cosine similarities required by MTEB."""

    def test_similarity_matrix(self, make_language_model: MakeLanguageModel) -> None:
        r"""similarity should return the (N, M) cosine similarity matrix, for tensors and arrays."""
        model, _ = make_language_model("qwen3")
        a, b = torch.randn(3, HIDDEN_SIZE), torch.randn(4, HIDDEN_SIZE)
        expected = F.cosine_similarity(a[:, None], b[None], dim=-1)
        torch.testing.assert_close(model.similarity(a, b), expected)
        torch.testing.assert_close(model.similarity(a.numpy(), b.numpy()), expected)

    def test_pairwise_similarity(self, make_language_model: MakeLanguageModel) -> None:
        r"""similarity_pairwise should return the cosine similarity of each pair, for tensors and arrays."""
        model, _ = make_language_model("qwen3")
        a, b = torch.randn(3, HIDDEN_SIZE), torch.randn(3, HIDDEN_SIZE)
        expected = F.cosine_similarity(a, b, dim=-1)
        torch.testing.assert_close(model.similarity_pairwise(a, b), expected)
        torch.testing.assert_close(model.similarity_pairwise(np.asarray(a), np.asarray(b)), expected)


# -------------------------------------------------------------------
# Properties
# -------------------------------------------------------------------
class TestProperties:
    r"""Tests for the properties of the model."""

    def test_model_in_eval_mode(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""The model should be loaded in evaluation mode."""
        model, _ = make_language_model(language_spec)
        assert not model.model.training

    def test_n_layers_and_embedding_dim(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""n_layers and embedding_dim should match the configuration of the HuggingFace model."""
        model, _ = make_language_model(language_spec)
        assert (model.n_layers, model.embedding_dim) == (N_LAYERS, HIDDEN_SIZE)

    def test_gated_ffn_act_is_pre_hook(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""All LLMs rely on gated FFNs, so act should be recovered as the input of the down projection."""
        model, _ = make_language_model(language_spec)
        assert model.block_adapter(model.blocks[0]).pre_hooks == {"rc1", "act"}

    def test_final_norm(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""The final norm should be the norm of the HuggingFace model."""
        model, hf_model = make_language_model(language_spec)
        assert model.final_norm is hf_model.norm

    def test_mteb_model_meta(self, make_language_model: MakeLanguageModel, language_spec: str) -> None:
        r"""MTEB metadata should name the probed layer (nonnegative) and module, and describe the model."""
        model, hf_model = make_language_model(language_spec)
        model.set_probing(layer=-1, module="ln2")
        meta = model.mteb_model_meta
        assert meta.name == f"{model.model_name}_layer_{N_LAYERS - 1}_module_ln2"
        assert meta.embed_dim == HIDDEN_SIZE
        assert meta.n_parameters == sum(p.numel() for p in hf_model.parameters())
        assert meta.open_weights and not meta.use_instructions

    def test_repr_counts_parameters(self, make_language_model: MakeLanguageModel) -> None:
        r"""__repr__ should give the class name and the number of trainable parameters."""
        model, hf_model = make_language_model("qwen3")
        n_params = sum(p.numel() for p in hf_model.parameters())
        assert repr(model) == f"Qwen3 model with {n_params:,} trainable parameters."
