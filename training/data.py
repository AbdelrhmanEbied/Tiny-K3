from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.distributed as dist
from datasets import interleave_datasets, load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase

from configs.trainer_config import TrainConfig

_TEXT_COLUMN_CANDIDATES = ("text", "content", "article")

_TOKENIZE_CHUNK = 64


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None = None
    split: str = "train"
    streaming: bool = True
    probability: float = 1.0
    text_column: str | None = None
    # passed to load_dataset(); enables local files / "text" builder,
    # e.g. DatasetSpec("text", data_files={"train": "data/file.txt"})
    data_files: dict[str, Any] | str | None = None


FINWEB_MIX: list[DatasetSpec] = [
    DatasetSpec("HuggingFaceFW/fineweb-edu", name="sample-10BT", probability=0.60),
    DatasetSpec("HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", probability=0.25),
    DatasetSpec("open-web-math/open-web-math", probability=0.15),
]


def _extract_text(example: dict[str, Any], text_column: str | None) -> str:
    if text_column is not None:
        if text_column not in example:
            raise KeyError(
                f"text_column={text_column!r} not found; available: {list(example)}"
            )
        value = example[text_column]
    else:
        for col in _TEXT_COLUMN_CANDIDATES:
            if col in example and isinstance(example[col], str):
                value = example[col]
                break
        else:
            raise ValueError(
                f"No text column found; available: {list(example)}. Set DatasetSpec.text_column explicitly."
            )
    if not isinstance(value, str):
        raise TypeError(f"Column value must be str, got {type(value)}")
    return value


class PackedTextDataset(IterableDataset):
    def __init__(
        self,
        specs: DatasetSpec | list[DatasetSpec],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
        shuffle_buffer_size: int = 50_000,
        add_eos: bool = True,
        max_batches: int | None = None,
    ):
        super().__init__()
        if isinstance(specs, DatasetSpec):
            specs = [specs]
        if not specs:
            raise ValueError("specs must contain at least one DatasetSpec")

        self.specs = specs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.add_eos = add_eos
        self.max_batches = max_batches

        eos_id = tokenizer.eos_token_id
        if add_eos and eos_id is None:
            raise ValueError("tokenizer.eos_token_id must be defined when add_eos=True")

        sources = [
            load_dataset(
                spec.path,
                name=spec.name,
                split=spec.split,
                streaming=spec.streaming,
                data_files=spec.data_files,
            )
            for spec in specs
        ]

        if len(sources) == 1:
            stream = sources[0]
        else:
            probs = [spec.probability for spec in specs]
            if abs(sum(probs) - 1.0) > 1e-6:
                raise ValueError(f"probabilities must sum to 1.0, got {probs}")
            stream = interleave_datasets(
                sources,
                probabilities=probs,
                stopping_strategy="all_exhausted",
                seed=seed,
            )

        if shuffle:
            stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

        self.stream = stream
        self.text_column = specs[0].text_column if len(specs) == 1 else None

    def _tokenize(
        self, iterator: Iterator[dict], text_column: str | None
    ) -> Iterator[list[int]]:
        eos_id = self.tokenizer.eos_token_id
        while True:
            chunk = list(islice(iterator, _TOKENIZE_CHUNK))
            if not chunk:
                return
            texts = [_extract_text(ex, text_column) for ex in chunk]
            batch_ids = self.tokenizer(
                texts,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            for ids in batch_ids:
                if not ids:
                    continue
                if self.add_eos:
                    ids = [*ids, eos_id]
                yield ids

    @staticmethod
    def _shard_index() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            rank, world_size = dist.get_rank(), dist.get_world_size()
        else:
            rank, world_size = 0, 1
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        global_worker_id = rank * num_workers + worker_id
        return global_worker_id, world_size * num_workers

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        index, num_shards = self._shard_index()

        iterator = iter(self.stream)
        if num_shards > 1:
            iterator = islice(iterator, index, None, num_shards)

        token_buffer: list[int] = []
        yielded = 0

        for ids in self._tokenize(iterator, self.text_column):
            token_buffer.extend(ids)
            while len(token_buffer) >= self.max_seq_len:
                window = token_buffer[: self.max_seq_len]
                yield {
                    "input_ids": torch.tensor(window, dtype=torch.long),
                    "position_ids": torch.arange(self.max_seq_len, dtype=torch.long),
                }
                token_buffer = token_buffer[self.max_seq_len :]

                yielded += 1
                if self.max_batches is not None and yielded >= self.max_batches:
                    return


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def build_dataloader(
    dataset: IterableDataset,
    cfg: TrainConfig,
    *,
    batch_size: int | None = None,
) -> DataLoader:
    num_workers = cfg.num_workers
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size if batch_size is not None else cfg.micro_batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last,
        "persistent_workers": cfg.persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(**kwargs)


def build_train_dataset(
    cfg: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    specs: list[DatasetSpec] | None = None,
    seed: int = 42,
) -> PackedTextDataset:
    return PackedTextDataset(
        specs if specs is not None else FINWEB_MIX,
        tokenizer,
        cfg.max_seq_len,
        seed=seed,
        shuffle=True,
    )


def build_eval_dataset(
    cfg: TrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    specs: DatasetSpec | list[DatasetSpec],
    num_eval_steps: int | None = None,
    seed: int = 1234,
) -> PackedTextDataset:
    if num_eval_steps is None:
        num_eval_steps = cfg.num_eval_steps
    max_batches = (
        num_eval_steps * cfg.grad_accum_steps * cfg.micro_batch_size
        if num_eval_steps is not None
        else None
    )
    return PackedTextDataset(
        specs,
        tokenizer,
        cfg.max_seq_len,
        seed=seed,
        shuffle=False,
        max_batches=max_batches,
    )
