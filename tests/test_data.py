from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from configs.trainer_config import TrainConfig
from training.data import (
    DatasetSpec,
    PackedTextDataset,
    _extract_text,
    build_dataloader,
    collate_fn,
)

SEQ_LEN = 8


class StubTokenizer:
    """Whitespace tokenizer with a growing vocab; id 0 reserved for EOS."""

    def __init__(self, eos_token_id: int | None = 0):
        self.eos_token_id = eos_token_id
        self._vocab: dict[str, int] = {}

    def __call__(self, texts, truncation=False, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = len(self._vocab) + 1
            ids.append(self._vocab[word])
        return ids


def make_cfg(**overrides) -> TrainConfig:
    defaults: dict = {
        "lr": 1e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "num_train_steps": 100,
        "warmup_steps": 10,
        "optimizer_type": "AdamW",
        "micro_batch_size": 2,
        "grad_accum_steps": 1,
        "max_seq_len": SEQ_LEN,
        "mixed_precision": "no",
        "compile_model": False,
        "compile_mode": "default",
        "enable_gradient_checkpointing": False,
        "save_interval": 50,
        "num_eval_steps": 4,
        "eval_interval": 10,
        "log_interval": 5,
        "out_dir": "/tmp/opencode/tiny-k3-out",
        "project_name": "tiny-k3-test",
        "checkpoint_path": "",
        "deepspeed_enabled": False,
        "wall_clock_breakdown": False,
        "zero_stage": 0,
        "overlap_comm": False,
        "contiguous_gradients": False,
        "reduce_bucket_size": 1,
        "allgather_bucket_size": 1,
        "allgather_partitions": False,
        "reduce_scatter": False,
        "offload_optimizer": False,
        "offload_param": False,
        "partition_activations": False,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": False,
        "num_checkpoints": 0,
        "loss_scale_window": 1000,
        "moe_enabled": False,
        "ep_size": 1,
        "moe_param_group": False,
        "use_residual": False,
        "drop_last": True,
        "num_workers": 0,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "pin_memory": False,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def make_docs(*docs: str, column: str = "text") -> Dataset:
    return Dataset.from_dict({column: list(docs)})


def patch_sources(monkeypatch, sources: dict[str, Dataset]):
    def fake_load(path, name=None, split="train", streaming=False, data_files=None):
        ds = sources[path]
        return ds.to_iterable_dataset() if streaming else ds

    monkeypatch.setattr("training.data.load_dataset", fake_load)


def make_dataset(docs, monkeypatch, **kwargs):
    patch_sources(monkeypatch, {"fake/docs": make_docs(*docs)})
    spec = DatasetSpec("fake/docs")
    tok = kwargs.pop("tokenizer", None) or StubTokenizer()
    ds = PackedTextDataset(spec, tok, kwargs.pop("seq_len", SEQ_LEN), **kwargs)
    return ds, tok


class TestPacking:
    def test_windows_are_exact_length_and_contiguous(self, monkeypatch):
        docs = ["a b c d e f g h i j k l m n o p q r s t u v w x y z"]
        ds, tok = make_dataset(docs, monkeypatch)
        windows = [s["input_ids"] for s in ds]

        assert all(w.shape == (SEQ_LEN,) for w in windows)
        assert all(w.dtype == torch.long for w in windows)
        assert (
            torch.cat(windows).tolist() == tok.encode(docs[0])[: SEQ_LEN * len(windows)]
        )

    def test_eos_separates_documents(self, monkeypatch):
        docs = ["a b c", "d e f"]
        ds, tok = make_dataset(
            docs * 4, monkeypatch, shuffle=False, seed=0, shuffle_buffer_size=2
        )
        # every window comes from one continuous stream; verify against the
        # expected packed stream built independently
        expected: list[int] = []
        for d in docs * 4:
            expected.extend(tok.encode(d))
            expected.append(0)  # eos
        windows = torch.cat([s["input_ids"] for s in ds]).tolist()
        assert windows == expected[: len(windows)]
        assert len(windows) % SEQ_LEN == 0

    def test_no_eos_when_disabled(self, monkeypatch):
        docs = ["a b c"] * 6
        ds, tok = make_dataset(docs, monkeypatch, add_eos=False)
        expected: list[int] = []
        for d in docs:
            expected.extend(tok.encode(d))
        windows = torch.cat([s["input_ids"] for s in ds]).tolist()
        assert windows == expected[: len(windows)]

    def test_empty_documents_skipped(self, monkeypatch):
        ds, tok = make_dataset(["", "   ", "a b c d e f g h i j"], monkeypatch)
        expected: list[int] = [*tok.encode("a b c d e f g h i j"), 0]
        windows = torch.cat([s["input_ids"] for s in ds]).tolist()
        assert windows == expected[: len(windows)]

    def test_max_batches_caps_output(self, monkeypatch):
        ds, _ = make_dataset(
            ["a b c d e f g h i j k l"] * 20, monkeypatch, max_batches=3
        )
        assert len(list(ds)) == 3


class TestColumnHandling:
    def test_explicit_text_column(self, monkeypatch):
        patch_sources(
            monkeypatch, {"fake/docs": make_docs("a b c d e f g h i", column="content")}
        )
        ds = PackedTextDataset(
            DatasetSpec("fake/docs", text_column="content"), StubTokenizer(), SEQ_LEN
        )
        assert len(list(ds)) > 0

    @pytest.mark.parametrize("column", ["text", "content", "article"])
    def test_auto_detection(self, monkeypatch, column):
        patch_sources(
            monkeypatch, {"fake/docs": make_docs("a b c d e f g h i", column=column)}
        )
        ds = PackedTextDataset(DatasetSpec("fake/docs"), StubTokenizer(), SEQ_LEN)
        assert len(list(ds)) > 0

    def test_missing_column_raises(self, monkeypatch):
        patch_sources(monkeypatch, {"fake/docs": make_docs("a b c", column="body")})
        ds = PackedTextDataset(DatasetSpec("fake/docs"), StubTokenizer(), SEQ_LEN)
        with pytest.raises(ValueError, match="text_column"):
            list(ds)

    def test_non_string_column_raises(self, monkeypatch):
        patch_sources(
            monkeypatch, {"fake/docs": Dataset.from_dict({"text": [123, 456]})}
        )
        ds = PackedTextDataset(DatasetSpec("fake/docs"), StubTokenizer(), SEQ_LEN)
        # auto-detection skips non-str values, then reports available columns
        with pytest.raises(ValueError, match="No text column"):
            list(ds)


class TestConstruction:
    def test_empty_specs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            PackedTextDataset([], StubTokenizer(), SEQ_LEN)

    def test_single_spec_accepted_directly(self, monkeypatch):
        patch_sources(monkeypatch, {"fake/docs": make_docs("a b c d e f g h i")})
        ds = PackedTextDataset(DatasetSpec("fake/docs"), StubTokenizer(), SEQ_LEN)
        assert len(list(ds)) > 0

    def test_probabilities_must_sum_to_one(self, monkeypatch):
        patch_sources(
            monkeypatch,
            {
                "fake/a": make_docs("a b c"),
                "fake/b": make_docs("d e f"),
            },
        )
        specs = [
            DatasetSpec("fake/a", probability=0.5),
            DatasetSpec("fake/b", probability=0.7),
        ]
        with pytest.raises(ValueError, match="sum to 1"):
            PackedTextDataset(specs, StubTokenizer(), SEQ_LEN)

    def test_interleaved_mix_produces_windows(self, monkeypatch):
        patch_sources(
            monkeypatch,
            {
                "fake/a": make_docs(*(["x y z w"] * 20)),
                "fake/b": make_docs(*(["p q r s"] * 20)),
            },
        )
        specs = [
            DatasetSpec("fake/a", probability=0.5),
            DatasetSpec("fake/b", probability=0.5),
        ]
        ds = PackedTextDataset(specs, StubTokenizer(), SEQ_LEN, shuffle=False)
        for sample in ds:
            assert sample["input_ids"].shape == (SEQ_LEN,)
            break

    def test_requires_eos_when_adding(self, monkeypatch):
        patch_sources(monkeypatch, {"fake/docs": make_docs("a b c")})
        tok = StubTokenizer(eos_token_id=None)
        with pytest.raises(ValueError, match="eos_token_id"):
            PackedTextDataset(DatasetSpec("fake/docs"), tok, SEQ_LEN)


class TestStreamingAndSharding:
    def test_streaming_source(self, monkeypatch):
        patch_sources(
            monkeypatch, {"fake/docs": make_docs(*(["a b c d e f g h i j k l"] * 5))}
        )
        ds = PackedTextDataset(
            DatasetSpec("fake/docs", streaming=True), StubTokenizer(), SEQ_LEN
        )
        assert len(list(ds)) > 0

    def test_nondeterministic_without_seed_is_still_valid(self, monkeypatch):
        patch_sources(
            monkeypatch, {"fake/docs": make_docs(*(["a b c d e f g h i j"] * 10))}
        )
        ds = PackedTextDataset(DatasetSpec("fake/docs"), StubTokenizer(), SEQ_LEN)
        for sample in ds:
            assert sample["input_ids"].shape == (SEQ_LEN,)


class TestCollateAndLoader:
    def test_collate_labels_mirror_input_ids(self):
        batch = [
            {"input_ids": torch.arange(SEQ_LEN)},
            {"input_ids": torch.ones(SEQ_LEN, dtype=torch.long)},
        ]
        out = collate_fn(batch)
        assert out["input_ids"].shape == (2, SEQ_LEN)
        assert torch.equal(out["labels"], out["input_ids"])

    def test_build_dataloader_uses_cfg(self, monkeypatch):
        ds, _ = make_dataset(["a b c d e f g h i j k l m n o p"] * 5, monkeypatch)
        cfg = make_cfg(micro_batch_size=2, num_workers=0)
        dl = build_dataloader(ds, cfg)
        batch = next(iter(dl))
        assert set(batch) == {"input_ids", "labels"}
        assert batch["input_ids"].shape == (2, SEQ_LEN)
        assert batch["labels"].shape == (2, SEQ_LEN)

    def test_extract_text_fallback_order(self):
        assert _extract_text({"content": "c"}, None) == "c"
        assert _extract_text({"text": "t", "content": "c"}, None) == "t"
        assert _extract_text({"article": "a", "other": "x"}, None) == "a"
