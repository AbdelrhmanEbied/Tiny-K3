"""Launch TinyK3 pretraining on fineweb-edu with `python -m training.train`.

Optional overrides: put a JSON dict of TrainConfig fields in `overrides.json`
(or point $TRAIN_OVERRIDES at one) and it is applied on top of the defaults.
"""

from __future__ import annotations

import json
import os
from typing import Any

from configs.model_config import ModelConfig
from configs.trainer_config import TrainConfig
from model.model import TinyK3Model
from training.data import (
    DatasetSpec,
    build_dataloader,
    build_eval_dataset,
    build_train_dataset,
)
from training.tokenizer import TokenizerManager
from training.trainer import train

TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"

FINWEB_EDU = DatasetSpec(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    streaming=True,
)


def load_overrides(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        overrides = json.load(f)
    print(f"Loaded {len(overrides)} override(s) from {path}")
    return overrides


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig(
        **load_overrides(os.environ.get("TRAIN_OVERRIDES", "overrides.json"))
    )

    tokenizer = TokenizerManager(TOKENIZER_NAME, cfg.max_seq_len)

    model_kwargs = load_overrides(
        os.environ.get("MODEL_OVERRIDES", "model_overrides.json")
    )
    # explicit overrides take precedence; otherwise follow the trainer's seq len
    model_kwargs.setdefault("max_seq_len", cfg.max_seq_len)
    model_kwargs.setdefault("original_max_seq_len", cfg.max_seq_len)

    model_cfg = ModelConfig(**model_kwargs)
    model = TinyK3Model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    accelerator_hint = f"{cfg.micro_batch_size}x{cfg.grad_accum_steps} micro/accum"
    print(
        f"TinyK3 | {n_params:,} params | seq {cfg.max_seq_len} | "
        f"batch {accelerator_hint} | {cfg.num_train_steps} steps | {cfg.optimizer_type}"
    )

    train_ds = build_train_dataset(cfg, tokenizer.tok, specs=[FINWEB_EDU])
    eval_ds = build_eval_dataset(cfg, tokenizer.tok, specs=[FINWEB_EDU])

    train(
        model,
        build_dataloader(train_ds, cfg),
        build_dataloader(eval_ds, cfg),
        cfg,
        qb_update_interval=cfg.qb_update_interval,
        tokenizer=tokenizer.tok,
    )


if __name__ == "__main__":
    main()
