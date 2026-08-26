"""End-to-end pipeline test: overfit a tiny TinyK3 model on tiny shakespeare.

Trains a small model (32K Llama-2 vocab, 4 layers, hidden 64) on
test_data/tiny_shakespeare.txt, exercising: tokenizer -> packed streaming data ->
MoE/QB routing -> eval loop -> generation sampling -> checkpointing.
"""

from __future__ import annotations

import os

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

DATA_FILE = "test_data/tiny_shakespeare.txt"
TOKENIZER_NAME = "hf-internal-testing/llama-tokenizer"
OUT_DIR = "runs/pipeline_test"


def build_model_config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        num_layers=4,
        num_attention_heads=4,
        kv_lora_rank=16,
        qk_nope_dim=8,
        qk_rope_dim=8,
        num_experts=4,
        num_experts_per_token=2,
        moe_intermediate_size=64,
        moe_latent_dim=32,
        first_k_dense_replace=1,
        attnres_block_layers=2,
        max_seq_len=256,
    )


def build_cfg() -> TrainConfig:
    return TrainConfig(
        lr=3e-3,
        betas=[0.9, 0.95],
        weight_decay=0.0,
        grad_clip=1.0,
        num_train_steps=300,
        warmup_steps=30,
        optimizer_type="AdamW",
        micro_batch_size=8,
        grad_accum_steps=1,
        max_seq_len=256,
        mixed_precision="no",
        compile_model=False,
        compile_mode="default",
        enable_gradient_checkpointing=False,
        project_name="tiny-k3-pipeline-test",
        save_interval=100,
        num_eval_steps=5,
        eval_interval=100,
        log_interval=10,
        out_dir=OUT_DIR,
        checkpoint_path="",
        deepspeed_enabled=False,
        wall_clock_breakdown=False,
        zero_stage=0,
        overlap_comm=False,
        contiguous_gradients=False,
        reduce_bucket_size=1,
        allgather_bucket_size=1,
        allgather_partitions=False,
        reduce_scatter=False,
        offload_optimizer=False,
        offload_param=False,
        partition_activations=False,
        cpu_checkpointing=False,
        contiguous_memory_optimization=False,
        num_checkpoints=0,
        loss_scale_window=1000,
        moe_enabled=False,
        ep_size=1,
        moe_param_group=False,
        use_residual=False,
        drop_last=True,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        gen_interval=150,
    )


def main() -> None:
    assert os.path.exists(DATA_FILE), f"missing {DATA_FILE}"

    tokenizer = TokenizerManager(TOKENIZER_NAME, 256)
    print(f"Tokenizer: {TOKENIZER_NAME} | vocab={len(tokenizer.tok)}")

    spec = DatasetSpec("text", data_files={"train": DATA_FILE}, streaming=True)

    cfg = build_cfg()
    model_cfg = build_model_config(len(tokenizer.tok))
    model = TinyK3Model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {model_cfg.num_layers}L hidden={model_cfg.hidden_size} | {n_params:,} params"
    )

    train_ds = build_train_dataset(cfg, tokenizer.tok, specs=[spec], seed=42)
    eval_ds = build_eval_dataset(cfg, tokenizer.tok, specs=[spec], seed=7)
    train_dl = build_dataloader(train_ds, cfg)
    eval_dl = build_dataloader(eval_ds, cfg)

    train(
        model,
        train_dl,
        eval_dl,
        cfg,
        qb_update_interval=25,
        tokenizer=tokenizer.tok,
    )

    final_ckpt = os.path.join(OUT_DIR, f"step_{cfg.num_train_steps}")
    assert os.path.exists(os.path.join(final_ckpt, "model.safetensors"))
    print(f"\nPipeline OK — final checkpoint at {final_ckpt}")


if __name__ == "__main__":
    main()
