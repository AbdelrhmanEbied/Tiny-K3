from __future__ import annotations

import math
import os
import shutil
import time

import bitsandbytes as bnb
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.trainer_config import TrainConfig
from model.moe import MoE
from training.helpers import build_accelerator, copy_checkpoint_from_dataset

_KEEP_CHECKPOINTS = 2


def _wait_everyone(accelerator) -> None:
    # no-op for single-process runs where no process group exists
    if accelerator.num_processes > 1:
        _wait_everyone(accelerator)

_GENERATION_PROMPTS = (
    "The meaning of life is",
    "Once upon a time in a distant kingdom",
    "To solve this math problem, first",
    "A matrix is a ",
)

_GENERATION_CONFIGS: dict[str, dict] = {
    "temp0.8_topk50": {"temperature": 0.8, "top_k": 50},
    "temp1.0_topp0.9": {"temperature": 1.0, "top_p": 0.9},
    "temp1.2_minp0.05": {"temperature": 1.2, "min_p": 0.05},
    "temp0.5_topk30": {"temperature": 0.5, "top_k": 30},
}
_NUM_NEW_TOKENS = 80


def _write_generation_samples(
    accelerator, raw_model: torch.nn.Module, tokenizer, cfg: TrainConfig, step: int
) -> None:
    device = next(raw_model.parameters()).device
    gen_dir = os.path.join(cfg.out_dir, "generations")
    os.makedirs(gen_dir, exist_ok=True)
    path = os.path.join(gen_dir, f"step_{step}.txt")

    was_training = raw_model.training
    raw_model.eval()
    sections: list[str] = [f"=== step {step} ===\n"]
    try:
        with torch.inference_mode():
            for prompt in _GENERATION_PROMPTS:
                input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                    device
                )
                prompt_len = input_ids.shape[1]
                for name, kwargs in _GENERATION_CONFIGS.items():
                    out = raw_model.generate(
                        input_ids,
                        max_new_tokens=_NUM_NEW_TOKENS,
                        eos_token_id=tokenizer.eos_token_id,
                        **kwargs,
                    )
                    text = tokenizer.decode(
                        out[0, prompt_len:], skip_special_tokens=True
                    )
                    sections.append(f"--- {prompt} [{name}] ---\n{text}\n")
    finally:
        if was_training:
            raw_model.train()

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(sections))
    accelerator.print(f"Generation samples written to {path}")


def _build_optimizer(cfg: TrainConfig, model: torch.nn.Module, device: torch.device):
    params = model.parameters()
    fused = device.type == "cuda"
    if cfg.optimizer_type == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=tuple(cfg.betas),
            weight_decay=cfg.weight_decay,
            fused=fused,
        )
    if cfg.optimizer_type == "PagedAdamW":
        return bnb.optim.PagedAdamW(
            params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
        )
    if cfg.optimizer_type == "8bitAdamW":
        return bnb.optim.AdamW8bit(
            params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer_type: {cfg.optimizer_type}")


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(
        1, cfg.num_train_steps - cfg.warmup_steps
    )
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def evaluate(
    model: torch.nn.Module,
    val_dataloader: DataLoader | None,
    accelerator,
) -> tuple[float | None, float | None]:
    if val_dataloader is None:
        return None, None
    was_training = model.training
    model.eval()
    losses: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in val_dataloader:
            input_ids = batch["input_ids"]
            out = model(input_ids, labels=batch.get("labels", input_ids))
            losses.append(accelerator.gather(out.loss.detach()).reshape(-1))
    if was_training:
        model.train()
    if not losses:
        return None, None
    val_loss = torch.cat(losses).mean().item()
    return val_loss, math.exp(min(val_loss, 20))


def _update_qb_bias(accelerator, raw_model: torch.nn.Module) -> None:
    moe_layers = [m for m in raw_model.modules() if isinstance(m, MoE)]
    if not moe_layers:
        return
    for m in moe_layers:
        m.update_bias()
    if dist.is_available() and dist.is_initialized():
        for m in moe_layers:
            dist.all_reduce(m.expert_bias, op=dist.ReduceOp.SUM)
            m.expert_bias /= dist.get_world_size()


def _find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    subdirs = [
        os.path.join(checkpoint_dir, d)
        for d in os.listdir(checkpoint_dir)
        if d.startswith("step_")
    ]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda x: int(x.rsplit("_", 1)[-1]))
    return latest if os.path.exists(os.path.join(latest, "metadata.pt")) else None


def _prune_checkpoints(accelerator, ckpt_dir: str, keep: int = 2) -> None:
    """Keep only the newest `keep` step_* dirs (20GB Kaggle output cap)."""
    dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
    dirs.sort(key=lambda x: int(x.rsplit("_", 1)[-1]))
    for old in dirs[:-keep]:
        accelerator.print(f"Pruning old checkpoint: {old}")
        shutil.rmtree(os.path.join(ckpt_dir, old), ignore_errors=True)


def _save_checkpoint(
    accelerator,
    model,
    raw_model,
    ckpt_dir: str,
    step: int,
    best_step: int,
    best_loss: float,
    current_loss: float,
    tokens_seen: int,
) -> None:
    accelerator.print(f"Saving checkpoint at step {step}...")
    _wait_everyone(accelerator)

    save_path = os.path.join(ckpt_dir, f"step_{step}")
    if accelerator.is_main_process:
        os.makedirs(save_path, exist_ok=True)

    accelerator.save_state(save_path, safe_serialization=True)

    state_dict = accelerator.get_state_dict(model, unwrap=True)

    if accelerator.is_main_process:
        raw_model.config.save_pretrained(save_path)

        torch.save(
            {
                "step": step,
                "best_step": best_step,
                "loss": current_loss,
                "best_loss": best_loss,
                "tokens_seen": tokens_seen,
            },
            os.path.join(save_path, "metadata.pt"),
        )

        clean_state_dict: dict[str, torch.Tensor] = {}
        pointer_map: dict[int, str] = {}
        for key, tensor in state_dict.items():
            clean_key = key.replace("_orig_mod.", "").replace("module.", "")
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            ptr = tensor.data_ptr()
            if ptr in pointer_map:
                clean_state_dict[clean_key] = tensor.clone()
                accelerator.print(f"Untying cloned shared storage for key: {clean_key}")
            else:
                pointer_map[ptr] = clean_key
                clean_state_dict[clean_key] = clean_state_dict.get(clean_key, tensor)

        save_file(clean_state_dict, os.path.join(save_path, "model.safetensors"))
        accelerator.print(f"Checkpoint successfully saved at {save_path}")
        _prune_checkpoints(accelerator, ckpt_dir, keep=_KEEP_CHECKPOINTS)

    _wait_everyone(accelerator)


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None,
    cfg: TrainConfig,
    qb_update_interval: int = 100,
    tokenizer=None,
):
    accelerator = build_accelerator(cfg)
    device = accelerator.device
    raw_model = model

    accelerator.init_trackers(project_name=cfg.project_name, config=vars(cfg))
    _wait_everyone(accelerator)

    if cfg.enable_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            accelerator.print("Warning: Model doesn't support gradient checkpointing")

    optimizer = _build_optimizer(cfg, model, device)
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if cfg.checkpoint_path:
        copy_checkpoint_from_dataset(cfg)
    _wait_everyone(accelerator)

    ckpt_dir = cfg.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    step = 0
    tokens_seen = 0
    best_loss = float("inf")
    best_step = 0
    resume_path = _find_latest_checkpoint(ckpt_dir)
    if resume_path is not None:
        accelerator.print(f"Resuming from: {resume_path}")
        try:
            accelerator.load_state(resume_path)
        except RuntimeError:
            accelerator.print("load_state failed, loading model.safetensors directly")
            weights = load_file(os.path.join(resume_path, "model.safetensors"))
            raw_model.load_state_dict(weights, strict=False)
        meta = torch.load(os.path.join(resume_path, "metadata.pt"), map_location="cpu")
        step = meta["step"]
        tokens_seen = meta["tokens_seen"]
        best_step = meta.get("best_step", step)
        best_loss = meta.get("best_loss", meta.get("loss", float("inf")))
        accelerator.print(
            f"Resumed at step {step} | best_loss {best_loss:.4f} | best_step {best_step} | tokens_seen {tokens_seen:,}"
        )

    if cfg.compile_model:
        accelerator.print(f"Compiling Now... (mode: {cfg.compile_mode})")
        model = torch.compile(model=model, mode=cfg.compile_mode, dynamic=True)

    model.train()
    progress_bar = tqdm(
        total=cfg.num_train_steps, initial=step, disable=not accelerator.is_main_process
    )
    data_iter = iter(train_dataloader)

    current_loss = float("nan")
    lr = _lr_at(step, cfg)
    grad_norm = torch.tensor(0.0)
    step_time = 0.0
    tps = 0.0
    perplexity = float("nan")
    val_loss: float | None = None
    val_ppl: float | None = None
    micro_tokens = 0

    while step < cfg.num_train_steps:
        step_start = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        with accelerator.accumulate(model):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch.get("labels", input_ids).to(device, non_blocking=True)
            position_ids = batch.get("position_ids")
            if position_ids is not None:
                position_ids = position_ids.to(device, non_blocking=True)

            if cfg.compile_model and cfg.compile_mode in (
                "reduce-overhead",
                "max-autotune",
            ):
                torch.compiler.cudagraph_mark_step_begin()

            micro_tokens += input_ids.numel()
            outputs = model(input_ids, labels=labels, position_ids=position_ids)
            current_loss = outputs.loss.detach()
            accelerator.backward(outputs.loss)

            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                token_tensor = torch.tensor(
                    micro_tokens, dtype=torch.long, device=device
                )
                global_step_tokens = accelerator.reduce(
                    token_tensor, reduction="sum"
                ).item()
                micro_tokens = 0
                tokens_seen += global_step_tokens

                lr = _lr_at(step, cfg)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                current_loss = accelerator.gather(current_loss).mean().item()
                perplexity = math.exp(min(current_loss, 20))
                step_time = time.perf_counter() - step_start
                tps = global_step_tokens / max(step_time, 1e-9)

                if qb_update_interval > 0 and step % qb_update_interval == 0:
                    _update_qb_bias(accelerator, raw_model)

                if step % cfg.eval_interval == 0:
                    accelerator.print("Evaluating Now...")
                    _wait_everyone(accelerator)
                    val_loss, val_ppl = evaluate(model, val_dataloader, accelerator)
                    _wait_everyone(accelerator)
                    if accelerator.is_main_process and val_loss is not None:
                        accelerator.log(
                            {"val_loss": val_loss, "val_ppl": val_ppl}, step=step
                        )

                if (
                    cfg.gen_interval > 0
                    and step % cfg.gen_interval == 0
                    and accelerator.is_main_process
                ):
                    if tokenizer is not None:
                        _write_generation_samples(
                            accelerator, raw_model, tokenizer, cfg, step
                        )
                    else:
                        accelerator.print(
                            "gen_interval set but no tokenizer passed to train(); skipping"
                        )

                is_best = current_loss < best_loss
                if is_best:
                    best_loss = current_loss
                    best_step = step

                if step % cfg.save_interval == 0:
                    _save_checkpoint(
                        accelerator,
                        model,
                        raw_model,
                        ckpt_dir,
                        step,
                        best_step,
                        best_loss,
                        current_loss,
                        tokens_seen,
                    )

                if accelerator.is_main_process:
                    progress_bar.update(1)
                    postfix = {
                        "loss": f"{current_loss:.4f}",
                        "sec/step": f"{step_time:.2f}",
                        "Perplex": f"{perplexity:.4f}",
                    }
                    if val_ppl is not None:
                        postfix["val_ppl"] = f"{val_ppl:.4f}"
                    progress_bar.set_postfix(postfix)

                if step % cfg.log_interval == 0:
                    accelerator.print(
                        f"Step {step} | Loss: {current_loss:.4f} | "
                        f"Tokens: {tokens_seen:,} | LR: {lr:.2e} | "
                        f"Perplexity: {perplexity:.4f}"
                        + (f" | Val: {val_loss:.4f}" if val_loss is not None else "")
                    )

                    moe_log: dict[str, object] = {}
                    renames = {"dropped_frac": "overflow_rate"}
                    for (
                        metric_name,
                        metric_tensor,
                    ) in raw_model.aggregated_router_metrics().items():
                        key = renames.get(metric_name, metric_name)
                        moe_log[f"moe/{key}"] = (
                            accelerator.gather(metric_tensor.float()).mean().item()
                        )

                    weight_norm = torch.sqrt(
                        sum(
                            (p.detach().float() ** 2).sum()
                            for p in raw_model.parameters()
                        )
                    ).item()

                    if accelerator.is_main_process:
                        log_data: dict[str, object] = {
                            "train/loss": current_loss,
                            "train/learning_rate": lr,
                            "train/grad_norm": float(grad_norm),
                            "train/tokens_per_sec": tps,
                            "train/step_time": step_time,
                            "train/tokens_seen": tokens_seen,
                            "train/weight_norm": weight_norm,
                            **moe_log,
                        }
                        accelerator.log(log_data, step=step)

    accelerator.print("Saving final checkpoint...")
    _save_checkpoint(
        accelerator,
        model,
        raw_model,
        ckpt_dir,
        step,
        best_step,
        best_loss,
        current_loss,
        tokens_seen,
    )

    progress_bar.close()
    _wait_everyone(accelerator)
    accelerator.end_training()
    accelerator.print("Training complete.")
