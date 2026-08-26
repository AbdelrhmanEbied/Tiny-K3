from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenizerManager:
    """Thin wrapper around a HF tokenizer for the training pipeline.

    Handles loading, batch encoding (truncation + padding to max_seq_len)
    and decoding. The underlying tokenizer is exposed via `.tok` so any
    HF-specific functionality remains reachable.
    """

    def __init__(self, name_or_path: str, max_seq_len: int):
        self.tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(name_or_path)
        self.max_seq_len = max_seq_len
        self.tok.model_max_length = int(1e12)

    @classmethod
    def from_tokenizer(
        cls, tok: PreTrainedTokenizerBase, max_seq_len: int
    ) -> TokenizerManager:
        """Wrap an already-loaded tokenizer instead of loading from a path."""
        manager = cls.__new__(cls)
        manager.tok = tok
        manager.max_seq_len = max_seq_len
        return manager

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    @property
    def pad_token_id(self) -> int | None:
        return self.tok.pad_token_id

    @property
    def eos_token_id(self) -> int | None:
        return self.tok.eos_token_id

    def encode(
        self,
        texts: str | list[str],
        *,
        padding: bool | str = False,
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        """Encode text(s) into token ids, truncated to max_seq_len."""
        if isinstance(texts, list) and not texts:
            raise ValueError("texts must not be empty")
        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.max_seq_len,
            padding=padding,
            return_tensors=return_tensors,
        )
        return dict(enc)

    def encode_batch(
        self,
        texts: list[str],
        *,
        pad_to_multiple_of: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts into padded (input_ids, attention_mask)."""
        if not texts:
            raise ValueError("texts must not be empty")
        enc = self.tok(
            texts,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length" if pad_to_multiple_of else True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def decode(
        self,
        token_ids: int | list[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        sequences: list[list[int]] | torch.Tensor,
        *,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        return self.tok.batch_decode(sequences, skip_special_tokens=skip_special_tokens)

    def save_pretrained(self, path: str) -> None:
        self.tok.save_pretrained(path)
