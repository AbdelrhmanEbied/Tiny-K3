from __future__ import annotations

import pytest
import torch
from datasets import Dataset  # noqa: F401  (ensures datasets imports cleanly first)
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from training.tokenizer import TokenizerManager

MAX_LEN = 16

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "to be or not to be that is the question",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
] * 20


@pytest.fixture(scope="module")
def fast_tok() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=False,
    )
    tok.train_from_iterator(_CORPUS, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )


@pytest.fixture(scope="module")
def manager(fast_tok) -> TokenizerManager:
    return TokenizerManager.from_tokenizer(fast_tok, MAX_LEN)


class TestConstruction:
    def test_wraps_underlying_tokenizer(self, manager, fast_tok):
        assert manager.tok is fast_tok
        assert manager.max_seq_len == MAX_LEN

    def test_properties_exposed(self, manager):
        assert manager.eos_token_id is not None
        assert manager.pad_token_id is not None
        assert isinstance(manager.vocab_size, int) and manager.vocab_size > 0


class TestEncode:
    def test_encode_single_text(self, manager):
        out = manager.encode("hello world")
        assert isinstance(out["input_ids"], list)
        assert len(out["input_ids"]) > 0

    def test_encode_truncates_to_max_seq_len(self, manager):
        long_text = "word " * 100
        assert len(manager.encode(long_text)["input_ids"]) == MAX_LEN

    def test_encode_empty_list_raises(self, manager):
        with pytest.raises(ValueError, match="empty"):
            manager.encode([])


class TestEncodeBatch:
    def test_returns_padded_tensors(self, manager):
        texts = ["short", "a much longer sentence with many tokens inside it"]
        input_ids, attention_mask = manager.encode_batch(texts)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape[0] == 2
        assert input_ids.shape[1] == attention_mask.shape[1]
        assert input_ids.shape[1] <= MAX_LEN

        lengths = attention_mask.sum(dim=1)
        assert lengths[1] > lengths[0]
        # padding only at the tail
        assert attention_mask[0][0] == 1
        assert attention_mask[0][-1] == 0

    def test_pad_to_multiple_of(self, manager):
        _, mask = manager.encode_batch(["hi", "there"], pad_to_multiple_of=8)
        assert mask.shape[1] % 8 == 0

    def test_truncates_long_inputs(self, manager):
        input_ids, _ = manager.encode_batch(["word " * 100])
        assert input_ids.shape[1] == MAX_LEN

    def test_empty_batch_raises(self, manager):
        with pytest.raises(ValueError, match="empty"):
            manager.encode_batch([])

    def test_padded_positions_do_not_affect_masked_lengths(self, manager):
        texts = ["abc", "abc"]
        ids_a, mask_a = manager.encode_batch(texts)
        ids_b, mask_b = manager.encode_batch(texts[:1])
        assert torch.equal(ids_a[0][: mask_a[0].sum()], ids_b[0][: mask_b[0].sum()])


class TestDecode:
    def test_round_trip_preserves_text(self, manager):
        text = "the quick brown fox"
        ids = manager.encode(text)["input_ids"]
        assert manager.decode(ids).strip() == text

    def test_decode_tensor_input(self, manager):
        ids = torch.tensor(manager.encode("hello there")["input_ids"])
        assert isinstance(manager.decode(ids), str)

    def test_batch_decode(self, manager):
        texts = ["first line", "second line"]
        encoded = [manager.encode(t)["input_ids"] for t in texts]
        decoded = manager.batch_decode(encoded)
        assert [d.strip() for d in decoded] == texts


class TestSaveLoad:
    def test_save_and_reload_round_trip(self, manager, tmp_path):
        path = str(tmp_path / "tok")
        manager.save_pretrained(path)

        reloaded = TokenizerManager(path, MAX_LEN)
        text = "to be or not to be"
        assert reloaded.encode(text)["input_ids"] == manager.encode(text)["input_ids"]
