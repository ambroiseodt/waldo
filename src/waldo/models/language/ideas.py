from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


class DecoderSentenceTransformer(SentenceTransformer):
    r"""
    Wrapper around a decoder-only LLM (e.g. Llama3) to make it compatible
    with the SentenceTransformer interface expected by MTEB.

    The sentence embedding is obtained by mean-pooling the last hidden states
    of the decoder over non-padding tokens.

    Parameters
    ----------
    model_name_or_path: str
        HuggingFace model name or local path to the decoder model.
    device: str, default="cuda"
        Device to run the model on.
    """

    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        # Do NOT call super().__init__() — it would try to load a
        # SentenceTransformer model from the path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        self.model.eval()
        self._device = device

        # SentenceTransformer expects a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        **kwargs,  # absorb any extra args MTEB passes
    ) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(
                    **encoded,
                    output_hidden_states=True,
                )

            # mean pool last hidden state over non-padding tokens
            last_hidden = outputs.hidden_states[-1]  # (B, T, D)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            embeddings = (last_hidden * attention_mask).sum(dim=1)
            embeddings = embeddings / attention_mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)


def encode(
    self,
    sentences: list[str],
    batch_size: int = 32,
    show_progress_bar: bool = False,
    **kwargs,
) -> np.ndarray:
    all_embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        encoded = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
            )

        last_hidden = outputs.hidden_states[-1]  # (B, T, D)

        # Build mask that excludes special tokens
        special_token_ids = set(
            filter(
                None,
                [
                    self.tokenizer.bos_token_id,
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                    self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id,
                ],
            )
        )

        # (B, T) — True for real content tokens only
        special_mask = torch.isin(
            encoded["input_ids"],
            torch.tensor(list(special_token_ids), device=self._device),
        )
        content_mask = (encoded["attention_mask"].bool() & ~special_mask).unsqueeze(-1)  # (B, T, 1)

        # Mean pool over content tokens only
        embeddings = (last_hidden * content_mask).sum(dim=1)
        embeddings = embeddings / content_mask.sum(dim=1).clamp(min=1)

        all_embeddings.append(embeddings.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)